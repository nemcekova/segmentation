from os import listdir
from numpy import asarray
import numpy as np
import cv2
from matplotlib import image
from matplotlib import pyplot
import re
from skimage import filters
from sklearn import preprocessing
from skimage import morphology
from skimage import exposure
import os
import math

import widthMeasures
import classification

def my_preprocessing(img):
    """Green channel extraction"""
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pyplot.imshow(img_grey)
    pyplot.show()
    img_grey = img[:,:,1]
    pyplot.imshow(img_grey)
    pyplot.show()
    mask = backgroundMask(img_grey)

    """MEDIAN FILTER"""
    conv = cv2.medianBlur(img_grey, 15)

    diff = cv2.subtract(conv, img_grey)
    masked_img = cv2.bitwise_and(diff, diff, mask=mask)
    normalized = normalization(masked_img)

    #opened = morphology.opening(normalized, morphology.disk(3))
    #tophat = morphology.white_tophat(normalized, morphology.disk(9))
    """    pyplot.imshow(opened)
    pyplot.show()"""
    return normalized, img_grey.shape

def preprocessingMorphology(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = backgroundMask(img_grey)
    img_grey = cv2.bitwise_and(img_grey, mask)
    #clahe = exposure.equalize_adapthist(img_grey)
    median = filters.median(img_grey)
    preprocessed = morphology.opening(median, morphology.square(5))
    diff = cv2.subtract(median, img_grey)
    preprocessed = morphology.black_tophat(preprocessed, morphology.disk(12))

    normalized = normalization(preprocessed)

    pyplot.imshow(preprocessed)
    pyplot.show()

    return normalized, img_grey.shape

def normalization(img):
    """normalization"""
    pixels = asarray(img)
    norm_dst = np.zeros(img.shape)
    norm = cv2.normalize(pixels, norm_dst, 0, 255, cv2.NORM_MINMAX)
    normalized = norm.reshape((-1, 1))
    return normalized


def backgroundMask(inputImg):

    ret, th = cv2.threshold(inputImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = morphology.closing(th, morphology.square(20))

    """pyplot.imshow(th)
    pyplot.show()"""

    result_dir = "/results/DRIVE/masks"
    if result_dir not in os.getcwd():
        os.chdir("." + result_dir)
    cv2.imwrite(filename, th)
    return th


def kmeans(img, grey_shape):
    """kmeans"""
    vectorized = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1.0)
    k = 4
    attempts = 1
    retval, labels, centers = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((grey_shape))
    return segmented_image, centers

def postprocessing(img, centers):
    """Post processing"""
    """ T = int(sorted(centers)[1])
    print(T)
    ret, th = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)
    #th = morphology.opening(th, morphology.square(4))"""
    segmented_image1 = cv2.medianBlur(img, 3)
    segmented_image2 = cv2.GaussianBlur(segmented_image1, (5, 5), 0)
    mask0 = np.ones((3, 3), np.uint8)
    final = morphology.opening(segmented_image2, mask0)
    pyplot.imshow(final)
    pyplot.show()

    #th = cv2.adaptiveThreshold(final,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,5,2)
    #th = morphology.closing(th)

    T = math.ceil(np.mean(final))
    print(T)
    ret, th = cv2.threshold(final, T, 255, cv2.THRESH_BINARY)
    th = morphology.opening(th)
    return th



path = './data/DRIVE/test/images/'

for filename in listdir(path):
    try:
        img_data = image.imread(path + filename)
    except:
        os.chdir("../../../")
        img_data = image.imread(path + filename)



    prepocessed, grey_shape = my_preprocessing(img_data)
    segmented_image, centers = kmeans(prepocessed, grey_shape)
    final = postprocessing(segmented_image, centers)
    #final1 = classification(final)

    #widthMeasures.main(final)

    #clahe = exposure.equalize_adapthist(final)
    #clahe = clahe.astype(np.uint8)
    #clahe = cv2.adaptiveThreshold(clahe,255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,5,2 )
    pyplot.imshow(final)
    pyplot.title(filename)
    pyplot.show()

    result_dir = "/images"
    if result_dir not in os.getcwd():
        os.chdir(".." + result_dir)
    cv2.imwrite(filename, final)

    #break
