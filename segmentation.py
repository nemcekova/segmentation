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
    #img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_grey = img[:,:,1]
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
    img_grey = img[:,:,1]
    mask = backgroundMask(img_grey)
    img_grey = cv2.bitwise_and(img_grey, mask)
    clahe = exposure.equalize_adapthist(img_grey)
    """closed = morphology.closing(clahe, morphology.disk(8))
    open = morphology.opening(closed, morphology.disk(8))
    preprocessed = cv2.subtract(clahe, open)"""
    preprocessed = morphology.black_tophat(clahe, morphology.disk(12))
    normalized = normalization(preprocessed)
    """median = filters.median(img_grey)
    preprocessed = morphology.opening(median, morphology.square(5))
    diff = cv2.subtract(median, img_grey)
    preprocessed = morphology.black_tophat(preprocessed, morphology.disk(12))

    normalized = normalization(preprocessed)"""

    pyplot.imshow(clahe)
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


def kmeans(img, grey_shape, k):
    """kmeans"""
    vectorized = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1.0)
    k = k
    attempts = 1
    retval, labels, centers = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((grey_shape))
    return segmented_image, centers

def postprocessing(img, centers):
    """Post processing"""
    segmented_image1 = cv2.medianBlur(img, 3)
    segmented_image2 = cv2.GaussianBlur(segmented_image1, (5, 5), 0)
    mask0 = np.ones((3, 3), np.uint8)
    final = morphology.opening(segmented_image2, mask0)
    """    pyplot.imshow(final)
    pyplot.show()"""

    T = math.ceil(np.mean(final))
    ret, th = cv2.threshold(final, T, 255, cv2.THRESH_BINARY)
    th = morphology.opening(th)
    return final

def thickVessels(img):
    img_grey = img[:, :, 1]
    mask = backgroundMask(img_grey)
    img_grey = cv2.bitwise_and(img_grey, mask)
    clahe = exposure.equalize_adapthist(img_grey)

    preprocessed = morphology.black_tophat(clahe, morphology.disk(12))
    #preprocessed = filters.median(preprocessed, morphology.square(5))
    normalized = normalization(preprocessed)

    segmented_image, centers = kmeans(normalized, grey_shape, 3)

    centers = np.sort(centers.flatten())

    T = (int(centers[-2]) + int(centers[-1])) / 2
    print(T)
    print(int(centers[-2]))
    ret, th = cv2.threshold(segmented_image, T, 255, cv2.THRESH_BINARY)
    pyplot.imshow(th)
    pyplot.show()
    #opened = morphology.opening(th, morphology.square(2))
    clean = classification.reduceNoise(th)
    clean = morphology.dilation(clean)
    pyplot.imshow(clean)
    pyplot.show()
    return clean

path = './data/DRIVE/test/images/'

for filename in listdir(path):
    try:
        img_data = image.imread(path + filename)
    except:
        os.chdir("../../../")
        img_data = image.imread(path + filename)



    prepocessed, grey_shape = my_preprocessing(img_data)
    segmented_image, centers = kmeans(prepocessed, grey_shape, 5)
    final = postprocessing(segmented_image, centers)
    #final1 = classification(final)

    #widthMeasures.main(final)

    pyplot.imshow(segmented_image)
    pyplot.title(filename)
    pyplot.show()

    thick = thickVessels(img_data)
    #thin = cv2.subtract(final, thick)

    result_dir = "/images"
    if result_dir not in os.getcwd():
        os.chdir(".." + result_dir)
    cv2.imwrite(filename, segmented_image)

    break
