import cv2
import numpy as np
import scipy.ndimage as sn
import copy
from matplotlib import pyplot as plt
from PIL import ImageEnhance, Image
from skimage.feature import peak_local_max

data_mask = np.loadtxt("data\im3_gold_mask.txt")
data_coor = np.loadtxt("data\im3_gold_cells.txt")

# Image enhancement

img = cv2.imread('data\im3.jpg', 2)
dst = cv2.fastNlMeansDenoising(img,None,3,7,21)
dst = Image.fromarray(dst)
contrast = ImageEnhance.Contrast(dst)
dst = contrast.enhance(1.5)
dst = np.asarray(dst)
dst2 = cv2.GaussianBlur(dst,(3,3),0)
rows, col = dst2.shape

# part 1

a, mask = cv2.threshold(dst2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
kernel = np.ones((15,15),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
kernel = np.ones((5,5),np.uint8)
mask = cv2.dilate(mask, kernel)
mask= sn.binary_fill_holes(mask).astype('uint8')
kernel = np.ones((9,9),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
kernel = np.ones((11,11),np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

# Metrics

TP = 0
FP = 0
FN = 0
for row in range(rows):
    for index in range(col):
        if mask[row][index] == 1 and data_mask[row][index] == 1:
            TP += 1
        if mask[row][index] == 1 and data_mask[row][index] == 0:
            FP += 1
        if mask[row][index] == 0 and data_mask[row][index] == 1:
            FN += 1

prec = TP / (TP + FP)
recall = TP / (TP + FN)
Fscore = TP / (TP + (1/2 *(FP + FN)))

print('Part 1:')
print("Precision: " + "{:.2f}".format(prec))
print("Recall: " + "{:.2f}".format(recall))
print("Fscore: " + "{:.2f}".format(Fscore) + '\n')

# plot

plt.xticks([]), plt.yticks([])
plt.imshow(mask, cmap = 'gray')
plt.show()

# part 2
for row in range(rows):
    for index in range(col):
        if dst2[row][index] > 195:
            dst2[row][index] = 255
        else:
            dst2[row][index] = 0

kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(dst2, kernel)

new = copy.deepcopy(mask)

for row in range(rows):
    for index in range(col):
        if erosion[row][index] == 255:
            new[row][index] = 0


dist = cv2.distanceTransform(new, cv2.DIST_L2, 3)
cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
coordinates = peak_local_max(dist, min_distance=14)

cells = []
for x, y in coordinates:
    cells = np.append(cells, data_coor[x][y])


# Plot
plt.xticks([]), plt.yticks([])
plt.plot(coordinates[:,1], coordinates[:,0], 'ro', markersize=3)
plt.imshow(dist, cmap = 'gray')
plt.show()

# Metrics

val = 999
TP_2 = 0
dup = 0
cells = np.sort(cells)
for c in range(cells.size):
    if val != cells[c] and 0 != cells[c]:
        TP_2 -= dup
        dup = 0
        TP_2 += 1
    if val == cells[c] and 0 != cells[c]:
        dup = 1
    val = cells[c]

TP_2 -= dup

prec = TP_2 / cells.size
recall = TP_2 / data_coor.max()
Fscore = (2 * prec * recall) / (prec + recall)

print('Part 2:')
print("Precision: " + "{:.2f}".format(prec))
print("Recall: " + "{:.2f}".format(recall))
print("Fscore: " + "{:.2f}".format(Fscore) + '\n')

# Part 3


def neighbors(coor, cell):
    neighbor = [[cell,int(coor[0]), int(coor[1])]]
    for c in range(-1, 2, 1):
        for d in range(-1, 2, 1):
            neighbor.append([cell, int(coor[0]) + c * 3, int(coor[1]) + d * 3])
    return neighbor


part_3 = copy.deepcopy(new)
part_3 = part_3.astype(int)

for row in range(rows):
    for index in range(col):
        if part_3[row][index] == 1:
            part_3[row][index] = -1


regions = []
cell_tot = coordinates.shape[0]
for cell_id in range(coordinates.shape[0]):
    regions.extend(neighbors(coordinates[cell_id], cell_id + 1))


checked = []

part_3 = np.pad(part_3, ((3,3),(3,3)), 'constant')

while regions:
    e = regions[0][0]
    x = regions[0][1]
    y = regions[0][2]
    checked.append(regions[0])
    if part_3[x+3][y+3] == -1:
        for c in range(-2,3,1):
            for d in range(-2, 3, 1):
                part_3[x+c+3][y+d+3] = e
        neighs = neighbors([x,y], e)
        for e in neighs:
            if e not in checked:
                regions.append(e)
    regions.pop(0)

part_3 = part_3[3:-3, 3:-3]

for row in range(rows):
    for index in range(col):
        if part_3[row][index] == -1:
            part_3[row][index] = 0

# Metrics

scores_dice = []
scores_iou = []

for c in range(cell_tot):
    cell = -1
    TP_3 = 0
    FP_3 = 0
    FN_3 = 0
    for row in range(rows):
        for index in range(col):
            if cell == -1 and data_coor[row][index] != 0 and part_3[row][index] == c + 1:
                cell = data_coor[row][index]
            if part_3[row][index] == c + 1 and cell == data_coor[row][index]:
                TP_3 += 1
            elif part_3[row][index] != c + 1 and cell == data_coor[row][index]:
                FN_3 += 1
            elif part_3[row][index] == c + 1 and cell != data_coor[row][index]:
                FP_3 += 1
    scores_dice = np.append(scores_dice,(2 * TP_3)/((2*TP_3) + FN_3 + FP_3))
    scores_iou = np.append(scores_iou, TP_3/(TP_3 + FN_3 + FP_3))


thresh1d = 0
thresh2d = 0
thresh3d = 0

thresh1i = 0
thresh2i = 0
thresh3i = 0

for c in range(scores_dice.size):
    if scores_dice[c] > 0.5:
        thresh1d += 1
    if scores_dice[c] > 0.75:
        thresh2d += 1
    if scores_dice[c] > 0.9:
        thresh3d += 1
    if scores_iou[c] > 0.5:
        thresh1i += 1
    if scores_iou[c] > 0.75:
        thresh2i += 1
    if scores_iou[c] > 0.9:
        thresh3i += 1

print('Part 3:')
print("Threshold 0.5: \n")
print("Dice: " + "{:.2f}".format(thresh1d/scores_dice.size))
print("IoU: " + "{:.2f}".format(thresh1i/scores_dice.size) + '\n')
print("Threshold 0.75: \n")
print("Dice: " + "{:.2f}".format(thresh2d/scores_dice.size))
print("IoU: " + "{:.2f}".format(thresh2i/scores_dice.size) + '\n')
print("Threshold 0.9: \n")
print("Dice: " + "{:.2f}".format(thresh3d/scores_dice.size))
print("IoU: " + "{:.2f}".format(thresh3i/scores_dice.size) + '\n')

plt.imshow(part_3, cmap = 'prism')
plt.xticks([]), plt.yticks([])

plt.show()

