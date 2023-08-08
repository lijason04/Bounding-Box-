import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import matplotlib
from matplotlib import pyplot as plt
import os
import logging

import threading
from threading import Thread
import time



import time

def timeit(f):

    def timed(*args, **kw):

        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        #print('func:%r args:[%r, %r] took: %2.4f sec' % \
        #  (f.__name__, args, kw, te-ts) )
        return result

    return timed



modelResNet50 = tf.keras.applications.resnet50.ResNet50(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000, classifier_activation = 'softmax')

#@timeit
def processImg(image, minDimensions=(2,2), maxDimensions=(-1,-1)):
    print("processImg has started!")

    #@timeit
    def getAllRectFromImgArr(image):
        print("getAllRectFromImgArr has started!")
        maxX = maxDimensions[0]
        maxY = maxDimensions[1]

        rectArr = []
        for i in range(image.shape[1]):
            for j in range(image.shape[0]):
                for k in range(i+minDimensions[0], image.shape[1]):
                    for l in range(j+minDimensions[1], image.shape[0]):
                        rectArr.append(  (i,j,k,l) )  #x1,y1, x2,y2
        return rectArr
    def verifySinglePrediction(prediction):
        k = prediction[281] + prediction[282] + prediction[283] + prediction[284] + prediction[285] + prediction[286] + prediction[287] + prediction[299] + prediction[289] + prediction[290] + prediction[291] + prediction[292] + prediction[293]
        max = 0
        maxIndx = 0
        for i in range(len(prediction)):
            if max < prediction[i]:
                max = prediction[i]
                maxIndx = i
        if (maxIndx > 280) and (maxIndx < 294):
            return True
        return False

    def rectangleAssistant(image, rectList, batchSize, startingIndex, appendSet):
        # print("rectangleAssistant is starting!")
        if startingIndex > len(rectList):
            return 0
        rawImgBuffer = []
        a = startingIndex + batchSize
        if (a > len(rectList)):
            a = len(rectList)
        for i in range(startingIndex, a):
            img = np.array(image[ rectList[i][0] : rectList[i][2], rectList[i][1] : rectList[i][3]])
            padded = tf.image.resize(img, (224, 224))
            rawImgBuffer.append(padded)
        
        predictions = modelResNet50.predict(tf.keras.applications.resnet50.preprocess_input( np.array(rawImgBuffer), data_format=None), verbose=True )
        for predIndex in range(len(predictions)):
            if verifySinglePrediction(predictions[predIndex]):
                appendSet.append(predIndex)
        return 1


    #@timeit
    def processRectangle(image, rectangles, batchSize=1000, threadcount=2):
        print(len(rectangles))
        print("processRectangle has started!")
        results = []
        for i in range(len(rectangles) // (batchSize * threadcount)):
            threads = []
            for j in range(threadcount):
                #print(j)
                
                startingIndex = i + j * batchSize
                threads.append(threading.Thread(target=rectangleAssistant, args=(image, rectangles, batchSize, startingIndex, results) ))
                threads[j].start()
            print()
            for thread in threads:
                thread.join()
                #print("One of the threads is done!")
            #print(len(results))
        return results


    #@timeit
    def filter(rectangleArray, resultIndexArray):
        print("filter has started!")
        ret = set()

        for i in resultIndexArray:
            currentRect = rectangleArray[i]
            for j in resultIndexArray:
                if ( (rectangleArray[j][0] > currentRect[0]) and (rectangleArray[j][1] > currentRect[1]) ) and ( (rectangleArray[j][2] < currentRect[2]) and (rectangleArray[j][3] < currentRect[3])):
                    currentRect = rectangleArray[j]
            ret.add(currentRect)
        return ret
    rectArr = getAllRectFromImgArr(image=image)
    finalResults = processRectangle(image=image, rectangles=rectArr, batchSize=256, threadcount=4)
    return filter(rectArr, finalResults)


                    

    
def plotResults(image, boundingBoxes):
    print(len(boundingBoxes))
    print("plotResults has started!")
    data = plt.imshow(image)
    print(boundingBoxes)
    for rect in boundingBoxes:
        c1 = [rect[0], rect[1]] #top left
        c2 = [rect[2], rect[1]] #top right
        c3 = [rect[2], rect[3]] #bottom right
        c4 = [rect[0], rect[3]] #bottom left
        plt.plot(c1, c2, color="red")
        plt.plot(c2, c3, color="red")
        plt.plot(c3, c4, color="red")
        plt.plot(c4, c1, color="red")
    plt.show()
def runStuff(imagePath, downscaleSize=32):

    imgRaw = PIL.Image.open(imagePath)
    imgArr = np.asarray(imgRaw)
    kys = (imgArr.shape[0]//downscaleSize, imgArr.shape[1]//downscaleSize)
    imgRaw = imgRaw.resize(kys)
    imgArr = np.asarray(imgRaw)
    results = processImg(imgArr, (410,410))
    plotResults(imgArr, results)

imgPathWin ="C:\\Project23\\waitIm\\IMG_7986.jpg"
imgPathWSL = "waitIm/cat.jpg"
runStuff(imgPathWin, downscaleSize=2)