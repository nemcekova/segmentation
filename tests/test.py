from os import listdir
import numpy as np
import cv2
import re

from matplotlib import image
from sklearn.metrics import roc_curve, precision_recall_curve, accuracy_score
from sklearn.metrics._classification import confusion_matrix
from sklearn.preprocessing import binarize
from skimage.color import rgb2gray
import os
from statistics import mean

import measure

comparison_out = "./evaluation/{}/comparison/{}"
vessels_out = "./evaluation/{}/vessels/{}"
curves_out = "./evaluation/{}/measures"
testdata_manual = "../data/{}/test/manual"
testmask = "../data/{}/test/masks"


results_dir = "../results"
datasets = os.listdir(results_dir)
datasets = ["DRIVE"]

for dataset in datasets:
    print(dataset)

    all_results_img = os.listdir(results_dir + "/" + dataset + "/images/")
    all_results_masks = os.listdir(results_dir + "/" + dataset + "/masks/")
    print(all_results_img)
    #if os.path.exists(testmask.format(dataset)):
    all_specifity = []
    all_sensitivity = []
    all_accuracy = []
    for img in all_results_img:

        vessels_pred = rgb2gray(image.imread(results_dir + "/" + dataset + "/images/" + img))
        """get correct mask and manual image"""
        img_number = re.findall(r'\d+', img)
        img_name = [name for name in listdir(testdata_manual.format(dataset)) if img_number[0] in name]
        vessels_gt = rgb2gray(image.imread(testdata_manual.format(dataset) + '/' + img_name[0]))
        print(img_name)
        vessels_pred = binarize(vessels_pred)
        vessels_gt = binarize(vessels_gt)
        cm = np.array([[0,0],[0,0]])
        accuracy = []
        for truth, pred in zip(vessels_gt, vessels_pred):
            c = confusion_matrix(truth, pred)
            #tn, fp, fn, tp
            cm = np.add(cm, c)
            #accuracy.append(accuracy_score(truth, pred))

        tn, fp, fn, tp = cm.ravel()
        accuracy =  (tp + tn) / (tp + fp + fn + tn)
        print("Accuracy: ", accuracy)
        all_accuracy.append(accuracy)
        specificity = tn / (tn + fp)
        print("Specificity: ", specificity)
        all_specifity.append(specificity)
        sensitivity = tp / (tp + fn)
        print("Sensitivity: ", sensitivity)
        all_sensitivity.append(sensitivity)
    print("Accuracy: ", mean(all_accuracy))
    print("Specificity: ", mean(all_specifity))
    print("Sensitivity: ", mean(all_sensitivity))

