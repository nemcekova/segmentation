import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import skeletonize, thin
from skimage import filters

def edges(img):
    #laplacian = cv2.Laplacian(img, cv2.CV_8U)
    laplacian = cv2.Canny(img, 5, 200)
    return laplacian

def skeleton(img):
    kernel = np.ones((7, 7), np.uint8) #might use 5x5
    dilate = cv2.dilate(img, kernel, iterations=1)
    binary = dilate > filters.threshold_otsu(dilate)
    thinned = skeletonize(binary)

    #plt.imshow(thinned)
    #plt.show()
    return thinned

"""def sliding_window(img, stepSize, windowSize):
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield(x, y, img[y:y+windowSize[1], x:x+windowSize[0]])

def distance(img):
    (winW, winH) = (50, 50)
    highest = 0
    for (x, y, window) in sliding_window(img, stepSize=50, windowSize=(winW, winH)):
        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = img[x:x + winW, y:y + winH]
        cnt = 0
        loopX = 0
        vessel = False
        edge = False

        #prechadzanie riadkov
        for index in clone:
           #prechadzanie stlpcov v riadku
            loopY = 0
            width = 0
            for pixel in index:
                # v prvom riadku, cize odtial vychadza cieva
                if loopX == 0:
                    if pixel == 255 and vessel == False and edge == False:
                        edge = True
                    elif edge == True and pixel == 255:
                        continue
                    elif edge == True and pixel == 0 and width == 0:
                        vessel = True
                        edge = False
                        width += 1
                    elif vessel == True and pixel == 0:
                        width += 1
                    elif vessel == True and pixel == 255:
                        print(width)
                        vessel = False
                        plt.imshow(clone)
                        plt.show()


                loopY += 1
            loopX += 1

"""
def main(img):
    borderline = edges(img)
    #innerline = np.uint8(skeleton(img))
    #final = cv2.bitwise_or(borderline, cv2.UMat(innerline))
    #distance(borderline)
    plt.imshow(borderline)
    plt.show()