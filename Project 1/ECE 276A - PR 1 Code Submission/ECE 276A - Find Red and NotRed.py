#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:30:21 2020

@author: Will Argus
ECE 276A 
Find Red and NotRed
"""

import os, cv2
import numpy as np

redPixelDir = "/Users/femw90/Google Drive/SECKSY (1)/Grad School/Winter Quarter 2020/ECE 276A/stopsigns/trainset/RedPixels"
redPixels = np.empty([1, 3], dtype = int)
os.chdir(redPixelDir)
folder = "/Users/femw90/Google Drive/SECKSY (1)/Grad School/Winter Quarter 2020/ECE 276A/stopsigns/trainset/RedPixels"

for file in os.listdir(folder):
    if (file.endswith(".jpg")):
        img = cv2.imread(file)
        redLocations = np.where(np.all(img >= [0, 0, 55], axis = -1))
        redPixels = np.append(redPixels, img[redLocations], axis = 0)
redPixels = redPixels[1:]

redSize = np.shape(redPixels)
redLabels = np.empty([redSize[0]])
redLabels[:] =  1


notRedPixelDir = "/Users/femw90/Google Drive/SECKSY (1)/Grad School/Winter Quarter 2020/ECE 276A/stopsigns/trainset/notRedPixels"
notRedPixels = np.empty([1, 3], dtype = int)
os.chdir(notRedPixelDir)
folder = "/Users/femw90/Google Drive/SECKSY (1)/Grad School/Winter Quarter 2020/ECE 276A/stopsigns/trainset/notRedPixels"

for file in os.listdir(folder):
    if (file.endswith(".jpg")):
        img = cv2.imread(file)
        notRedLocations = np.where(np.all(img >= [0, 0, 55], axis = -1))
        notRedPixels = np.append(notRedPixels, img[notRedLocations], axis = 0)
notRedPixels = notRedPixels[1:]

notRedSize = np.shape(notRedPixels)
notRedLabels = np.empty([notRedSize[0]])
notRedLabels[:] =  -1


TrainingData = np.concatenate(redPixels, notRedPixels)
TrainingDataLabels = np.concatenate(redLabels, notRedLabels)
np.save('TrainingData', TrainingData) 
np.save('TrainingDataLabels', TrainingDataLabels) 









