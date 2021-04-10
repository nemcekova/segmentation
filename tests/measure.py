import numpy as np
import cv2
import os

def confusion_matrix(img_gt, img_pred):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    print(img_gt)
    for gt, pred in zip(img_gt, img_pred):
        print(gt)
        for gt_i, pred_i in zip(gt, pred):
            if gt_i == 0 and pred_i == 0:
                tn += 1
            elif gt_i != 0 and pred_i != 0:
                tp += 1
            elif gt_i == 0 and pred_i != 0:
                fp += 1
            elif gt_i != 0 and pred_i == 0:
                fn += 0

    print(tp)
    print(tn)
    print(fp)
    print(fn)