import cv2
from matplotlib import pyplot as plt
from skimage import filters

def sliding_window(img, stepSize, windowSize):
    for y in range(0, img.shape[0], stepSize):
        for x in range(0, img.shape[1], stepSize):
            yield(x, y, img[y:y+windowSize[1], x:x+windowSize[0]])

def classification(img):
    (winW, winH) = (50,50)
    highest = 0

    for(x, y, window) in sliding_window(img, stepSize=50, windowSize=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

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

def localOtsu(img):
    (winW, winH) = (40, 40)

    for (x, y, window) in sliding_window(img, stepSize=40, windowSize=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        #clone = img[x:x + winW, y:y + winH]
        ret, img[x:x + winW, y:y + winH] = cv2.threshold(img[x:x + winW, y:y + winH], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img



def reduceNoise(img):
    (winW, winH) = (20, 20)
    highest = 0

    for (x, y, window) in sliding_window(img, stepSize=16, windowSize=(winW, winH)):

        if window.shape[0] != winH or window.shape[1] != winW:
            continue

        clone = img[x:x + winW, y:y + winH]
        # print(clone)
        cnt = 0
        for index in clone:
            for pixel in index:
                if pixel > 5:
                    cnt += 1
        threshold = (winW * winH) * 0.05
        if cnt < threshold:
            loopX = 0
            for index in clone:
                loopY = 0
                for pixel in index:
                    img[x + loopX, y + loopY] = 0
                    loopY += 1
                loopX += 1

    #print(highest)
    return(img)

"""li_t = filters.threshold_li(frangis)
    ret, li = cv2.threshold(frangis, li_t, 255, cv2.THRESH_BINARY)
    mean_t = filters.threshold_mean(frangis)
    ret, mean = cv2.threshold(frangis, mean_t, 255, cv2.THRESH_BINARY)
    triangle_t = filters.threshold_triangle(frangis)
    ret, triangle = cv2.threshold(frangis, triangle_t, 255, cv2.THRESH_BINARY)
    
 fig = pyplot.figure(figsize=(10,7))
    rows = 2
    cols = 2
    fig.add_subplot(rows, cols, 1)

    # showing image
    pyplot.imshow(th)
    pyplot.axis('off')
    pyplot.title("My thresh")

    # Adds a subplot at the 2nd position
       fig.add_subplot(rows, cols, 2)

    # showing image
    pyplot.imshow(li)
    pyplot.axis('off')
    pyplot.title("Li")

    # Adds a subplot at the 3rd position
    fig.add_subplot(rows, cols, 3)

    # showing image
    pyplot.imshow(mean)
    pyplot.axis('off')
    pyplot.title("Mean")

    # Adds a subplot at the 4th position
    fig.add_subplot(rows, cols, 4)

    # showing image
    pyplot.imshow(triangle)
    pyplot.axis('off')
    pyplot.title("Triangel")

    pyplot.show()"""