import re
import sys
import getopt
from os import listdir
from numpy import asarray
import numpy as np
import cv2
from matplotlib import image
from skimage import filters
from skimage import morphology
from skimage import exposure

import os


def my_preprocessing(img):
    """Green channel extraction"""
    img_grey = img[:, :, 1]
    img_mask = img[:, :, 0]

    h, w = img_grey.shape

    """resizing images with width over 2000"""
    if w > 2000:
        img_grey, img_mask = resize(img_grey, img_mask)

    mask = backgroundMask(img_mask)

    if w < 900:
        kernel_blur = (9, 9)
        kernel_tophat = 8
    elif w < 1400:
        kernel_blur = (13, 13)
        kernel_tophat = 8
    else:
        kernel_blur = (19,19)
        kernel_tophat = 8

    """preprocessing"""
    blured = cv2.GaussianBlur(img_grey, kernel_blur, 0)
    masked = cv2.bitwise_and(blured, mask)
    clahe = exposure.equalize_adapthist(masked, clip_limit=0.01)
    tophat = morphology.black_tophat(clahe, morphology.disk(kernel_tophat))

    return tophat, img_grey.shape

def resize( img_grey, img_mask):
    """Resize big image to half of its size"""
    h = int(img_grey.shape[0] * 0.5)
    w = int(img_grey.shape[1] * 0.5)

    img_grey = cv2.resize(img_grey, (w, h), interpolation=cv2.INTER_AREA)
    img_mask = cv2.resize(img_mask, (w, h), interpolation=cv2.INTER_AREA)

    return img_grey, img_mask

def normalization(img):
    """normalization"""
    pixels = asarray(img)
    norm_dst = np.zeros(img.shape)
    norm = cv2.normalize(pixels, norm_dst, 0, 255, cv2.NORM_MINMAX)
    normalized = norm.reshape((-1, 1))
    return normalized


def backgroundMask(inputImg):
    """mask of region of interest"""
    ret, th = cv2.threshold(inputImg, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    th = morphology.closing(th, morphology.square(20))

    """result_dir = "/results/DRIVE/masks"
    if result_dir not in os.getcwd():
        os.chdir("." + result_dir)
    cv2.imwrite(filename, th)"""
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
    segmented = segmented_data.reshape(grey_shape)

    return segmented, centers, labels



def thickVessels(img, centers, shape):
    """extraction of thin vessels"""
    sort = np.sort(centers.flatten())
    senior_class1 = sort[-1]
    thick = np.where(img == senior_class1, 255, img)
    thick = np.where(thick == 255, thick, 0)

    opened = morphology.opening(thick, morphology.square(3))

    return opened

def segmentation(img, shape):
    """segmentation of all vessels"""
    img = morphology.opening(img, morphology.square(3))
    frangis = filters.frangi(img, sigmas=(0.2, 2.7, 0.5), alpha=0.5, beta=15, gamma=15, black_ridges=False)
    frangis = normalization(frangis)
    frangis = frangis.reshape(shape)
    frangis = frangis.astype("uint8")

    t = filters.threshold_mean(frangis)
    ret, th = cv2.threshold(frangis, t, 255, cv2.THRESH_BINARY)

    image = morphology.opening(th, morphology.square(3))
    image = morphology.closing(image, morphology.square(3))
    return image

def creteSubDirs(outputfile):
    try:
        os.mkdir(outputfile + "images")
        os.mkdir(outputfile + "thin")
        os.mkdir(outputfile + "thick")
    except OSError:
        print("Creation of result dir failed")
        exit(-1)
def createResultDir(outputfile):
    try:
        os.mkdir(outputfile)
    except OSError:
        print("Creation of result dir failed")
        exit(-1)
    creteSubDirs(outputfile)

def main(argv):
    inputfile = ''
    outputfile = ''
    try:
        opts, args = getopt.getopt(argv, "hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print('segmentation.py -i <inputDir> -o <outputDir>')
        print("<inputDir> - contains images of retina")
        print("<outputDir> - will contain directories with results")
        print("if <outputDir> is not provided, \"results\" directory will be created in the <inputDir> directory")
        exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('test.py -i <inputDir> -o <outputDir>')
            print("<inputDir> - contains images of retina")
            print("<outputDir> - will contain directories with results")
            print("if <outputDir> is not provided, \"results\" directory will be created in the <inputDir> directory")
            sys.exit()

        elif opt in ("-i", "--ifile"):
            inputfile = arg
            if inputfile[0] == ".":
                inputfile = os.getcwd() + inputfile[1:]

            if not os.path.exists(inputfile):
                print("Input directory does not exist")
                exit(-1)

            if not os.path.isdir(inputfile):
                print("Input directory is not a dir")
                exit(-1)

            if inputfile[-1] != "/":
                inputfile = inputfile + "/"

        elif opt in ("-o", "--ofile"):
            outputpath = arg
            if outputpath[0] == ".":
                outputpath = os.getcwd() + outputpath[1:]

            if not os.path.exists(outputpath):
                print("Output directory does not exist")
                exit(-1)

            if not os.path.isdir(outputpath):
                print("Output directory is not a dir")
                exit(-1)

            if outputpath[-1] != "/":
                outputpath = outputpath + "/"

            outputfile = outputpath + "results/"
            if not os.path.exists(outputfile):
                createResultDir(outputfile)

    if not inputfile:
        print('segmentation.py -i <inputDir> -o <outputDir>')
        print("<inputDir> - contains images of retina")
        print("<outputDir> - will contain directories with results")
        print("if <outputDir> is not provided, \"results\" directory will be created in the <inputDir> directory")
        exit(1)

    if not outputfile:
        os.chdir(inputfile)
        outputpath = os.getcwd()
        outputfile = outputpath + "/results/"
        if not os.path.exists(outputfile):
            createResultDir(outputfile)


    for filename in listdir(inputfile):
        if os.path.isdir(inputfile + filename):
            continue
        try:
            img_data = image.imread(inputfile + filename)
        except:
            print("Cannot load image in this directory: " + inputfile + filename)
            exit(-1)

        prepocessed, grey_shape = my_preprocessing(img_data)
        segmented_image, centers, labels = kmeans(prepocessed, grey_shape, 3)
        thick = thickVessels(segmented_image, centers, grey_shape)
        segmented_image1, centers1, labels1 = kmeans(prepocessed, grey_shape, 10)
        final = segmentation(segmented_image1, grey_shape)

        thin = cv2.bitwise_xor(final, thick)
        thin = morphology.opening(thin)

        result_dir = outputfile + "images"
        thick_dir = "thick"
        thin_dir = "thin"

        os.chdir(result_dir)
        cv2.imwrite(filename, final) #+ ".png"
        img_number = re.findall(r'\d+', filename)

        if thick_dir not in os.getcwd():
            os.chdir("..")
            os.chdir(thick_dir)
        cv2.imwrite(img_number[0] + "_thick.png", thick)

        if thin_dir not in os.getcwd():
            os.chdir("..")
            os.chdir(thin_dir)
        cv2.imwrite(img_number[0] + "_thin.png", thin)
        #break

if __name__ == "__main__":
    main(sys.argv[1:])