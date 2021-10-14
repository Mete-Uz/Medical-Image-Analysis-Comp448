import cv2
import os
import math
import copy
import numpy as np
from libsvm.svmutil import *
from svm import *
from libsvm.svmutil import *
from sklearn import svm
def calculateCooccurrenceMatrix(grayImg, binNumber, di, dj):
    bin_image = copy.deepcopy(grayImg)
    
    for i in range(len(grayImg)):
        for j in range(len(grayImg[0])):
            pixel_bin = int(math.floor(grayImg[i][j])/(len(grayImg[0])/binNumber))
            bin_image[i][j] = pixel_bin
    
    N = [[0 for i in range(binNumber)] for j in range(binNumber)]
    for i in range(binNumber):
        for j in range(binNumber):
            for k in range(len(grayImg)):
                for l in range(len(grayImg[0])):
                    if (k - di > -1) and (k - di < len(grayImg)) and (l - dj > -1) and (l - dj < len(grayImg)) :
                        if (bin_image[k][l] == i) and (bin_image[k-di][l - dj] == j):
                            N[i][j]+=1
    print("OK")
    return N

def calculateAccumulatedCooccurrenceMatrix(grayImg, binNumber, d):
    accM = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, d, 0))
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, d, d))
    accM += temp
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, 0, d))
    accM += temp
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, -d, d))
    accM += temp
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, -d, 0))
    accM += temp
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, -d, -d))
    accM += temp
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, 0, -d))
    accM += temp
    temp = np.array(calculateCooccurrenceMatrix(grayImg, binNumber, d, -d))
    accM += temp
    return accM

def calculateCooccurrenceFeatures(accM):
    normalized = copy.deepcopy(accM).astype(float)
    (a,b) = accM.shape
    sums = []
    for i in range(a):
        sum = 0
        for j in range(b):
            sum += accM[i][j]
        sums.append(sum)
    
    for i in range(a):
        for j in range(b):
            if sums[i] != 0:
                temp = accM[i][j]/sums[i]
                normalized[i][j] = temp
    
    angular_second_moment = 0
    #angular second moment
    max = 0
    inverse_difference_moment = 0
    contrast = 0
    entropy = 0
    correlation = 0
    for i in range(a):
        sum2 = 0
        mean2 = 0
        std2 = 0
        sum3 = 0
        for k in range(b):
            sum2 += normalized[i][k]
        mean2 = sum2 / b 
        for t in range(b):
            sum3 += (normalized[i][t] - mean2)*(normalized[i][t] - mean2)
        std2 = math.sqrt(sum3/b)
        for j in range(b):
            angular_second_moment += normalized[i][j]*normalized[i][j]
            if normalized[i][j] > max:
                max = normalized[i][j]
            inverse_difference_moment += normalized[i][j]/(1 + (i-j)*(i-j))
            contrast += (i-j)*(i-j)*normalized[i][j]
            if normalized[i][j] > 0:
                entropy += normalized[i][j]*math.log(normalized[i][j])
            sum1 = 0
            mean1 = 0
            std1 = 0
            std1 = 0
            sum4 = 0
            for l in range(a):
                sum1 += normalized[l][j]
            mean1 = sum1 / a
            for p in range(a):
                sum4 += (normalized[p][j] - mean1)*(normalized[p][j] - mean1)
            std1 = math.sqrt(sum4/a)
            if (std1*std2) != 0:
                correlation += (i*j*normalized[i][j] - mean1*mean2)/(std1*std2)
    entropy = -entropy       
    
    return angular_second_moment, max, inverse_difference_moment, contrast, entropy, correlation

def writeFeaturestoTxt(data):
    print(len(data))
    with open("C://Users//mete//Desktop//medical//features.txt", 'w') as f:
        for i in range(len(data)):
            print(i)
            accM = calculateAccumulatedCooccurrenceMatrix(data[i], 8, 10)
            angular_second_moment, max, inverse_difference_moment, contrast, entropy, correlation = calculateCooccurrenceFeatures(accM)
            print(accM)
            f.write(str(angular_second_moment))
            print(angular_second_moment)
            f.write(" ")
            f.write(str(max))
            print(max)
            f.write(" ")
            f.write(str(inverse_difference_moment))
            print(inverse_difference_moment)
            f.write(" ")
            f.write(str(contrast))
            print(contrast)
            f.write(" ")
            f.write(str(entropy))
            print(entropy)
            f.write(" ")
            f.write(str(correlation))  
            print(correlation)
            f.write('\n')
        

def loadDataset(directory):
    data = []
    for filename in os.listdir(directory):
        if filename.endswith(".asm") or filename.endswith(".py"): 
            continue
        else:
            if filename.endswith(".jpg"):
                image = cv2.imread(os.path.join(directory, filename))
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                data.append(gray)
    return data

def loadLabels(directory):
    file = open(directory, 'r')
    lines = file.readlines()
    labels = []
    for line in lines:
        labels.append(line.strip())
    for i in range(len(labels)):
        labels[i] = int(labels[i])
    return labels

def svm(data, labels):
    prob = svm_problem(labels, data)
    str = "-t 0 -c 4 -b 1"
    param = svm_parameter(str)
    m = svm_train(prob, param)
    return m
    
def predict(model, test_data, test_labels):
    num_correct = 0
    for i in range(len(test_data)):
        true_label = test_labels[i]
        predicted_label = model.predict(test_data[i])
        print("Prediction: " + str(predicted_label) + "||" + "Correct label: " + str(true_label))
        if true_label == predicted_label:
            num_correct += 1
            print("Correct")
        else:
            print("Incorrect")
    return num_correct/len(test_data)

def readFeatures(txt):
    file = open(txt, 'r')
    lines = file.readlines()
    features = []
    for line in lines:
        features.append(line.split())
    for i in range(len(features)):
        for j in range(len(features[i])):
            features[i][j] = float(features[i][j])
    return features


#load dataset
train_data_directory = "C://Users//mete//Desktop//medical//dataset//dataset//training//" #USE YOUR OWN DIRECTORIES HERE
test_data_directory = "C://Users//mete//Desktop//medical//dataset//dataset//test//"
labels_train = "C://Users//mete//Desktop//medical//dataset//dataset//training_labels.txt"
labels_test = "C://Users//mete//Desktop//medical//dataset//dataset//test_labels.txt"
features_3_train = "C://Users//mete//Desktop//medical//features_3_train.txt"
x_train = loadDataset(train_data_directory)
x_test = loadDataset(test_data_directory)
y_train = loadLabels(labels_train)
y_test = loadLabels(labels_test)
ft_2_tr = readFeatures(features_3_train)
part3_labels = []
#for part3 labels
for i in range(len(y_train)):
    for j in range(16):
        part3_labels.append(y_train[i])
    
print(len(part3_labels))
print(len(ft_2_tr))
#part2 train set 
model = svm_train(part3_labels, ft_2_tr, '-c 1000 -t 2 -g 5') #'-c 10 -t 2 -g 5' '-c 10 -t 2 -g 5'
l, a, v = svm_predict(part3_labels, ft_2_tr, model)
print(a)

#writeFeaturestoTxt(x_train)
#calculate features and write to txt
#N = 4
#sub_size = 64
#cropped_x_train = []
#for k in range(len(x_train)):
#    for i in range(N):
#        lb = i*64
#        for j in range(N):
#            ub = j*64
#            crop_img_f = []
#            for t in range(sub_size):
#                crop_img = x_train[k][lb:lb+64][t][ub:ub+64]
#                if t == 0:
#                    crop_img_f = crop_img
#                    crop_img = [crop_img]
#                else:
#                    crop_img_f.append(crop_img)
#            if k == 0 & i ==0 & j ==0:
#                cropped_x_train = crop_img_f
#                cropped_x_train = [cropped_x_train]
#            else:
#                cropped_x_train.append(crop_img_f)
    
#print(angular_second_moment)
#print(max)
#print(inverse_difference_moment)
#print(contrast)
#print(entropy)
#print(correlation)
#print(y_train)
#print(len(x_train[0][0])) #1. rsim 1. row len 256
#print(x_train[0][0][0]) #0. satir 0. sutun
#print("Hello")
#print(int(math.floor(63/64)))
