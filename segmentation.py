import time
from os import listdir
from numpy import asarray
import numpy as np
import cv2
from matplotlib import image
from matplotlib import pyplot
import re
from skimage import filters
from sklearn import metrics

import widthMeasures

def preprocessing(img):
    """Green channel extraction"""
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = backgroundMask(img_grey)

    """MEAN FILTER"""
    # kernel = np.ones((11,11), np.float32)/121
    # conv = cv2.filter2D(img_grey, -1, kernel, borderType=cv2.BORDER_ISOLATED)

    """GAUSSIAN FILTER"""
    # kernel = cv2.getGaussianKernel(11, 11)
    # conv = cv2.filter2D(img_grey, -1, kernel)
    # conv = cv2.GaussianBlur(img_grey, (11, 11), 0)
    """MEDIAN FILTER"""
    conv = cv2.medianBlur(img_grey, 15)


    diff = cv2.subtract(conv, img_grey)
    masked_img = cv2.bitwise_and(diff, diff, mask=mask)
    normalized = normalization(masked_img)
    """pyplot.imshow(masked_img)
    pyplot.show()
    mask0 = np.ones((3, 3), np.uint8)
    opened = cv2.morphologyEx(normalized, cv2.MORPH_OPEN, mask0)
    pyplot.imshow(opened)
    pyplot.show()"""

    return normalized, img_grey.shape


def normalization(img):
    """normalization"""
    pixels = asarray(img)
    norm_dst = np.zeros(img.shape)
    norm = cv2.normalize(pixels, norm_dst, 0, 255, cv2.NORM_MINMAX)
    normalized = norm.reshape((-1, 1))
    return normalized


def backgroundMask(inputImg):

    #mask0 = np.ones((7, 7), np.uint8)
    #morphClose = cv2.morphologyEx(inputImg, cv2.MORPH_CLOSE, mask0)

    ret, th = cv2.threshold(inputImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #mask = cv2.bitwise_or(morphClose, th)
    #mask1 = np.ones((7, 7), np.uint8)
    #morphDilate = cv2.erode(mask, mask1)

    """pyplot.imshow(th)
    pyplot.show()"""
    return th


def kmeans(img, grey_shape):
    """kmeans"""
    vectorized = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1.0)
    k = 5
    attempts = 1
    retval, labels, centers = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((grey_shape))
    return segmented_image

def postprocessing(img):
    """Post processing"""
    segmented_image1 = cv2.medianBlur(img, 3)
    segmented_image2 = cv2.GaussianBlur(segmented_image1, (5, 5), 0)
    mask0 = np.ones((3, 3), np.uint8)
    final = cv2.morphologyEx(segmented_image2, cv2.MORPH_OPEN, mask0)
    #final = cv2.erode(segmented_image2, mask0, iterations=0)
    #final1 = cv2.dilate(final, mask0, iterations=0)
    return final

def sliding_window(img, stepSize, windowSize):
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield(x, y, img[y:y+windowSize[1], x:x+windowSize[0]])

def classification(img):
    (winW, winH) = (12,12)
    highest = 0

    for(x, y, window) in sliding_window(img, stepSize=9, windowSize=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        """here should be classification instead"""
        """clone = img.copy()
        cv2.rectangle(clone, (x,y), (x+ winW, y+winH), (255,255,255), 2)
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)"""
        clone = img[x:x+winW, y:y+winH]
        #print(clone)
        cnt = 0
        for index in clone:
            for pixel in index:
                if pixel > 5:
                    cnt += 1
        threshold = (winW*winH) * 0.3
        if cnt > threshold:
            loopX = 0
            for index in clone:
                loopY = 0
                for pixel in index:
                    if pixel > 5:
                        cnt += 1
                        img[x + loopX, y + loopY] = 255
                    else:
                        img[x + loopX, y + loopY] = 0
                    loopY += 1
                loopX += 1
        else:
            loopX = 0
            for index in clone:
                loopY = 0
                for pixel in index:
                    img[x + loopX, y + loopY] = 0
                    loopY += 1
                loopX += 1



    #print(highest)
    return(img)

path = '/Users/nemcekova/Documents/School/BP/databases/DRIVE/training/images'
pathMasks = '/Users/nemcekova/Documents/School/BP/databases/DRIVE/training/mask'
pathManual = '/Users/nemcekova/Documents/School/BP/databases/DRIVE/training/1st_manual'
for filename in listdir(path):
    img_data = image.imread(path + '/' + filename)

    """get correct mask and manual image"""
    img_number = re.findall(r'\d+',filename)
    manual_name = [name for name in listdir(pathManual) if img_number[0] in name]
    manual = image.imread(pathManual + '/' + manual_name[0])

    prepocessed, grey_shape = preprocessing(img_data)
    segmented_image = kmeans(prepocessed, grey_shape)
    final = postprocessing(segmented_image)
    #final1 = classification(final)

    widthMeasures.main(final)

    pyplot.imshow(final)
    pyplot.title(filename)
    pyplot.show()
    #print(final)
    #break
