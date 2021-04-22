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
from skimage import restoration
import os
import math

import widthMeasures
import classification

def my_preprocessing(img):
    """Green channel extraction"""
    img_grey = img[:, :, 1]
    img_mask = img[:, :, 0]
    mask = backgroundMask(img_mask)
    masked = cv2.bitwise_and(img_grey, mask)
    clahe = exposure.equalize_adapthist(masked)
    tophat = morphology.black_tophat(clahe, morphology.disk(8))
    tophat = matchedFilter(tophat)
    #normalized = normalization(tophat)

    return tophat, img_grey.shape

def preprocessingMorphology(img):
    img_grey = img[:,:,1]
    #mask = backgroundMask(img_grey)
    #img_grey = cv2.bitwise_and(img_grey, mask)
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

    """pyplot.imshow(clahe)
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
    img = normalization(img)
    vectorized = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 1.0)
    k = k
    attempts = 1
    retval, labels, centers = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape((grey_shape))
    return segmented_image, centers, labels

def postprocessing(img):
    """Post processing"""
    segmented_image1 = cv2.medianBlur(img, 3)
    segmented_image2 = cv2.GaussianBlur(segmented_image1, (5, 5), 0)
    mask0 = np.ones((3, 3), np.uint8)
    final = morphology.opening(segmented_image2, mask0)
    """pyplot.imshow(final)
    pyplot.show()"""

    T = math.ceil(np.mean(final))
    ret, th = cv2.threshold(final, T, 255, cv2.THRESH_BINARY)
    th = morphology.opening(th)
    return th

def matchedFilter(img):
    s = 1.5 #sigma
    L = 7 #lenght of piecewise element

    out = np.zeros(img.shape)

    m = max(math.ceil(3*s), (L-1)/2)
    x_val = np.arange(-m, m)
    y_val = np.arange(-m, m)
    [x,y] = np.meshgrid(x_val, y_val)

    for l in range(1, 13):
        theta = 15 * (l - 1)
        angle = theta / 180 * math.pi #in radians
        u = math.cos(angle)*x - math.sin(angle)*y
        v = math.sin(angle)*x + math.cos(angle)*y
        N = (abs(u) <= 3*s) & (abs(v) <= L/2) #domain
        k = np.exp((-u ** 2) / (2 * s ** 2)) #kernel
        k = k - np.mean(k[N]) #y K:(x, y) = K,(x, y) - m, m, =sum K,(x, y)/A
        #k[not N] = 0 # set kernel outside of domain to 0
        #print(k)
        res = cv2.filter2D(img, ddepth=-1, kernel=k, borderType=cv2.BORDER_CONSTANT)
        out = np.maximum(out, res)

    return out

def thickVessels(img, centers, shape):

    sort = np.sort(centers.flatten())
    """T = int(centers[-2]) - 1
    ret, th = cv2.threshold(img, T, 255, cv2.THRESH_BINARY)"""

    senior_class1 = sort[-1]
    thick = np.where(segmented_image == senior_class1, 255, segmented_image)
    thick = np.where(thick == 255, thick, 0)

    #clean = classification.reduceNoise(thick)
    clean = morphology.dilation(thick)
    opened = morphology.opening(clean, morphology.square(3))
    #opened = morphology.erosion(opened)

    return opened

def thinVessels(img, centers, shape):
    frangis = filters.frangi(img, sigmas=(0.5, 2.5, 0.5), alpha=0.5, beta=15, black_ridges=False)
    frangis = normalization(frangis)
    frangis = frangis.reshape(shape)
    frangis = frangis.astype("uint8")
    ret, th = cv2.threshold(frangis, 1, 255, cv2.THRESH_BINARY)

    pyplot.imshow(th)
    pyplot.show()

    image = morphology.closing(th)
    image = morphology.opening(image)
    pyplot.imshow(image)
    pyplot.show()

    return image


path = './data/DRIVE/test/images/'

for filename in listdir(path):
    try:
        img_data = image.imread(path + filename)
    except:
        os.chdir("../../../")
        img_data = image.imread(path + filename)

    prepocessed, grey_shape = my_preprocessing(img_data)
    segmented_image, centers, labels = kmeans(prepocessed, grey_shape, 3)
    thick = thickVessels(segmented_image, centers, grey_shape)
    segmented_image1, centers1, labels1 = kmeans(prepocessed, grey_shape, 10)
    final = thinVessels(segmented_image1, centers1, grey_shape)

    thin = cv2.bitwise_xor(final, thick)
    thin = morphology.opening(thin)
    """pyplot.imshow(thick)
    pyplot.show()
    pyplot.imshow(thin)
    pyplot.show()"""
    """pyplot.imshow(final)
    pyplot.title(filename)
    pyplot.show()"""


    result_dir = "/images"
    if result_dir not in os.getcwd():
        os.chdir(".." + result_dir)
    cv2.imwrite(filename, final)


    #break
