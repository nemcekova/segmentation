

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

def reduceNoise(img):
    (winW, winH) = (12, 12)
    highest = 0

    for (x, y, window) in sliding_window(img, stepSize=9, windowSize=(winW, winH)):

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