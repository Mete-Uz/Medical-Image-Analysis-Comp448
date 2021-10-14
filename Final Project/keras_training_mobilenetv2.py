import keras.backend as K
import os
import math
import cv2
import numpy as np
import tensorflow as tf
import time
import hashlib
import imgaug
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Reshape, Dropout, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.applications import mobilenet_v2
from keras.applications import VGG19
from __future__ import print_function
from keras.backend.tensorflow_backend import set_session
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from albumentations import ( Compose, ShiftScaleRotate, HorizontalFlip, VerticalFlip, RandomBrightness, RandomContrast, HueSaturationValue, GaussNoise, Blur, Flip, JpegCompression, ToFloat, imgaug ) 


    
BATCH_SIZE = 10
INPUT_SIZE_1 = 224
INPUT_SIZE_2 = 224
DATASET_DIR = 'dataset\\'
CLASS_NAMES = ['class-0', 'class-1', 'class-2'] #class names should be put here
EXP_NAME = 'mobilenet' #model name should be put here
MODEL_FNAME = './model_'+EXP_NAME+'.h5'

start_time = time.time()

def step_decay(epoch):
    print('changing learning rate')
    initial_lrate = 0.1
    drop = 0.5
    epochs_drop = 30.0
    lrate = initial_lrate * math.pow(drop, math.floor((1+epoch)/epochs_drop))
    print(lrate)
    return lrate

checkpoint_save = ModelCheckpoint(MODEL_FNAME, save_best_only=True, monitor='val_loss', mode='min')
sched_lr = LearningRateScheduler(step_decay)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
tensorboard = TensorBoard(log_dir='./logs/'+EXP_NAME+'/', histogram_freq=0, write_graph=True, write_images=True)

def strongAug(p=0.5):
    return Compose([
        ShiftScaleRotate(shift_limit=0.01, scale_limit=(-0.1,0.1), rotate_limit=360, border_mode=cv2.BORDER_REPLICATE), HorizontalFlip(), VerticalFlip(), RandomBrightness((0,0.1)), RandomContrast(0.1), HueSaturationValue(hue_shift_limit=20, sat_shift_limit=10, val_shift_limit=0), GaussNoise(var_limit=(10., 100)), Blur(5), imgaug.transforms.IAASharpen(lightness=(0.9, 1.)), ToFloat(max_value=255, always_apply=True) ], p=p)

def augmentSample(img):
    img = np.uint8(img)
    aug = strongAug(1)
    img = aug(image=img)['image']
    return img

base_model = mobilenet_v2.MobileNetV2(input_shape=(INPUT_SIZE_1, INPUT_SIZE_2, 3), include_top=False, weights='imagenet')
x = base_model.output
for layer in base_model.layers:
    layer.trainable = True
x = GlobalAveragePooling2D()(x)
preds = Dense(units=len(CLASS_NAMES), activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

if not os.path.exists(MODEL_FNAME):
    print('Start training...\n')
    train_datagen = ImageDataGenerator(preprocessing_function=augmentSample)

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory( DATASET_DIR + 'train\\', target_size=(INPUT_SIZE_1, INPUT_SIZE_2), batch_size=BATCH_SIZE, classes=CLASS_NAMES, class_mode='categorical', shuffle= True )  
    validation_generator = test_datagen.flow_from_directory( DATASET_DIR + 'test\\', target_size=(INPUT_SIZE_1, INPUT_SIZE_2), batch_size=BATCH_SIZE, classes=CLASS_NAMES, class_mode='categorical')
    for layer in model.layers:
        layer.trainable = True

    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    model.fit_generator(
        train_generator,
        steps_per_epoch=266 // BATCH_SIZE, 
        epochs=120,
        validation_data=validation_generator,
        validation_steps= 51 // BATCH_SIZE,
        callbacks=[checkpoint_save, tensorboard, sched_lr])

    training_time = time.time() - start_time
    print('Training time: ' + str(int(training_time // 60)) + ' minutes '+ str(int(training_time % 60)) + ' seconds')

tf.keras.backend.clear_session()
tf.keras.backend.set_learning_phase(0)

model = load_model(MODEL_FNAME)
session = tf.keras.backend.get_session()
save_pb_dir = 'pb_files\\' 
def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name=EXP_NAME+'.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, save_pb_dir, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen
INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=save_pb_dir)
print('Start evaluation ...\n')

test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory( DATASET_DIR + '/train', target_size=(INPUT_SIZE_1, INPUT_SIZE_2),batch_size=30, classes=CLASS_NAMES, class_mode='categorical')
test_loss = model.evaluate_generator(test_generator, steps=len(test_generator))[0]
print('\n')
print('Test loss: '+str(test_loss))

directory = DATASET_DIR + '/test'
classes = dict(zip(CLASS_NAMES, range(len(CLASS_NAMES))))
correct = 0.
count = 0
OKasNOK = 0
NOKasOK = 0
total_pred_time = 0

for class_dir in os.listdir(directory):
    cls = classes[class_dir]
    for imname in os.listdir(os.path.join(directory, class_dir)):
        if(imname[0:3] == "des"):
            break
        imageSample = image.load_img(os.path.join(directory, class_dir, imname), target_size=(INPUT_SIZE_1,INPUT_SIZE_2))
        startTime = time.time()
        im = image.img_to_array(imageSample)
        im = im.reshape(-1,INPUT_SIZE_1,INPUT_SIZE_2,3)
        out = model.predict(im / 255.)
        endTime = time.time()
        total_pred_time += endTime - startTime

        predicted_cls = np.argmax(np.mean(out, axis=0))
        if predicted_cls == cls:
            correct += 1
        else:
            misclassification_type = CLASS_NAMES[cls] +'as'+ CLASS_NAMES[predicted_cls]
            image.save_img('mispredictions/'+misclassification_type+'_'+imname, imageSample)
        count += 1
        print('Evaluated images: ' + str(count), end='\r')

print('Total number of mispredictions: '+str(int(count - correct)))
print('Evaluated ' + str(count) + ' images with time per frame: ' + str((total_pred_time)/count) + ' s \n')
print('Test Acc. = ' + str(correct / count) + '\n')


