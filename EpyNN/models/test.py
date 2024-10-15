import cv2 as cv
import numpy as np

path = 'models\\Images\\number3.png'
path1 = 'models\\Images\\dog.jpeg'
img = cv.imread(path1)
#img = cv.resize(img, (100, 100))
print(img.shape)
b = True
for i in range(28):
    for j in range(28):
        b = b & (img[i][j] == img[i][j][0]).all() 
print(b)
#img = np.moveaxis(img, 2, 0)
gray_img = cv.cvtColor(img, code = cv.COLOR_RGB2GRAY)
for i in range(28):
    for j in range(28):
        b = b & (img[i][j] == gray_img[i][j]).all() 
print(gray_img.shape)
print(b)
cv.imshow('NUMBER', gray_img)
cv.waitKey(0)