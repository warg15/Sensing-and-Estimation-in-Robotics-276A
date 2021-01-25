2# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os, cv2, math
from skimage.measure import label, regionprops
import numpy as np


def sigmoid(z):
    if z < -20:
        return 0
    else:
        return 1/(1+ math.exp(-z))

if __name__ =="__main__":
    TrainingData = np.load('TrainingData.npy')
    TrainingDataLabels = np.load('TrainingDataLabels.npy')
    WNew =np.array([0, 0, 0])
    WCurrent = np.array([0, 0, 0])
    a = 0.01
    numIterations = 1
    
    storeIterations = np.zeros((numIterations,3))
    
    for i in range (numIterations):
        result=0
        WCurrent = WNew
        for k in range (0,  5582780):
            j=1*k
            y=TrainingDataLabels[j]
            x=TrainingData[j,:]
            xT=[TrainingData[j,0], TrainingData[j,1], TrainingData[j,2]]
            z= y*np.dot(xT,WCurrent)
            toSum=y*x*(1-sigmoid(z))
            result = result + toSum
        WNew = WCurrent+ a*result
        storeIterations[i,:] = WNew
    print("Hello World!")
