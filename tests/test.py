from os import listdir
import numpy as np
import cv2
import re
import os
import sys
import getopt
from matplotlib import image
from sklearn.metrics._classification import confusion_matrix
from sklearn.preprocessing import binarize
from skimage.color import rgb2gray
from statistics import mean

def main(argv):
    fileManual = ''
    fileResults = ''
    fileSegmented = ''
    fileSegmentedThick = ''

    try:
        opts, args = getopt.getopt(argv, "hr:m:", ["rdir=", "mdir="])
    except getopt.GetoptError:
        print('test.py -r <resultsDir> -m <manualImagesDir>')
        print("<resultsDir> - contains directories \"images\" and \"thick\" from segmentation")
        print("<manualImagesDir> - contains manually segmented images")
        exit(2)

    for opt, arg in opts:
        if opt == '-h':
            print('test.py -r <resultsDir> -m <manualImagesDir>')
            print("<resultsDir> - contains directories \"images\" and \"thick\" from segmentation")
            print("<manualImagesDir> - contains manually segmented images")
            sys.exit()

        elif opt in ("-r", "--rdir"):
            fileResults = arg
            if fileResults[0] == ".":
                fileResults = os.getcwd() + fileResults[1:]

            if not os.path.exists(fileResults):
                print("Result directory does not exist")
                exit(-1)

            if not os.path.isdir(fileResults):
                print("Result directory is not a dir")
                exit(-1)

            if fileResults[-1] != "/":
                fileResults = fileResults + "/"

            fileSegmented = fileResults + "images/"
            fileSegmentedThick = fileResults + "thick/"
            if not os.path.exists(fileSegmented):
                print("Result image directory does not exist")
                exit(-1)

            if not os.path.isdir(fileSegmented):
                print("Result image directory is not a dir")
                exit(-1)

            if not os.path.exists(fileSegmentedThick):
                print("Result thick directory does not exist")
                exit(-1)

            if not os.path.isdir(fileSegmentedThick):
                print("Result thick directory is not a dir")
                exit(-1)

        elif opt in ("-m", "--mdir"):
            fileManual = arg
            if fileManual[0] == ".":
                fileManual = os.getcwd() + fileManual[1:]

            if not os.path.exists(fileManual):
                print("Manual directory does not exist")
                exit(-1)

            if not os.path.isdir(fileManual):
                print("Manual directory is not a dir")
                exit(-1)

            if fileManual[-1] != "/":
                fileManual = fileManual + "/"

    if not fileManual or not fileSegmented:
        print('test.py -r <resultsDir> -m <manualImagesDir>')
        print("<resultsDir> - contains directories \"images\" and \"thick\" from segmentation")
        print("<manualImagesDir> - contains manually segmented images")
        exit(-1)


    all_specifity = []
    all_sensitivity = []
    all_accuracy = []
    all_percentage = []

    for img in listdir(fileSegmented):
        if img == ".DS_Store": continue

        vessels_pred = image.imread(fileSegmented + img)
        """get correct manual image"""
        img_number = re.findall(r'\d+', img)
        img_name_thick = [name for name in listdir(fileSegmentedThick) if img_number[0] in name]
        vessels_thick = image.imread(fileSegmentedThick + img_name_thick[0])
        img_name = [name for name in listdir(fileManual) if img_number[0] in name]
        vessels_gt = image.imread(fileManual + img_name[0])

        if vessels_gt.shape != vessels_pred.shape:
            h = vessels_pred.shape[0]
            w = vessels_pred.shape[1]
            vessels_gt = cv2.resize(vessels_gt, (w, h), interpolation=cv2.INTER_AREA)
        if vessels_thick.shape != vessels_pred.shape:
            h = vessels_pred.shape[0]
            w = vessels_pred.shape[1]
            vessels_thick = cv2.resize(vessels_thick, (w, h), interpolation=cv2.INTER_AREA)


        vessels_pred = binarize(vessels_pred)
        vessels_gt = binarize(vessels_gt)
        vessels_thick = binarize(vessels_thick)

        cm = np.array([[0,0],[0,0]])
        accuracy = []
        for truth, pred in zip(vessels_gt, vessels_pred):
             c = confusion_matrix(truth, pred)
             #tn, fp, fn, tp
             cm = np.add(cm, c)

        tn, fp, fn, tp = cm.ravel()
        accuracy =  (tp + tn) / (tp + fp + fn + tn)
        print(img_name)
        print("Accuracy: ", accuracy)
        all_accuracy.append(accuracy)
        specificity = tn / (tn + fp)
        print("Specificity: ", specificity)
        all_specifity.append(specificity)
        sensitivity = tp / (tp + fn)
        print("Sensitivity: ", sensitivity)
        all_sensitivity.append(sensitivity)


        nonzero_all = cv2.countNonZero(vessels_pred)
        nonzero_thick = cv2.countNonZero(vessels_thick)
        percentage = float(nonzero_thick)/float(nonzero_all) * 100
        print("Thick vessels: ", percentage)
        print("Thin vessels: ", (100 - percentage))
        all_percentage.append(percentage)

    print("\nOverall results for dataset:")
    print("Accuracy: ", mean(all_accuracy))
    print("Specificity: ", mean(all_specifity))
    print("Sensitivity: ", mean(all_sensitivity))
    print("Thick vessels: ", mean(all_percentage) )
    print("Thin vessels: ", (100 - mean(all_percentage)))

if __name__ == "__main__":
    main(sys.argv[1:])


