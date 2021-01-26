#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 14:07:35 2020

@author: Will Argus
ECE276A WI20 HW1
Stop Sign Detector
"""
import os, cv2, os.path
from skimage.measure import label, regionprops
from matplotlib import pyplot as plt
from matplotlib.path import Path
import numpy as np

class StopSignDetector():
    def __init__(self):
        '''
            Initilize your stop sign detector with the attributes you need,
            e.g., parameters of your classifier
        '''

    def segment_image(self, img):
        '''
            Obtain a segmented image using a color classifier,
            e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture, 
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
            mask_img - a binary image with 1 if the pixel in the original image is red and 0 otherwise
        '''
        # YOUR CODE HERE
        size=np.shape(img)
        height=size[0]
        width = size[1]
        new_img = np.zeros([height*width, 3])
        new_img = np.reshape(img, (img.shape[0]*img.shape[1], img.shape[2]))
        wOptimized = ([-1474210], [-1942180], [1982300])
        vectorMask = np.zeros([height*width, 1])
        for i in range (height*width):
            xT = new_img[i, :]
            testValue = np.dot(xT, wOptimized)
            if testValue >= 0:
                vectorMask[i,0] = 1
        mask_img = np.reshape(vectorMask, [img.shape[1], img.shape[0]], 1)
        mask_img = np.transpose(mask_img)
        return mask_img

    def get_bounding_box(self, img):
        '''
            Find the bounding box of the stop sign
            call other functions in this class if needed
            
            Inputs:
                img - original image
            Outputs:
                boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2] 
                where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively. The order of bounding boxes in the list
                is from left to right in the image.
                
            Our solution uses xy-coordinate instead of rc-coordinate. More information: http://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions
        '''
        # YOUR CODE HERE
        mask_img = self.segment_image(img)
        blur_mask_img = cv2.GaussianBlur(mask_img, (3,3),cv2.BORDER_DEFAULT)
        ret,th1 = cv2.threshold(blur_mask_img,0,255,cv2.THRESH_BINARY)
        blurred_img = th1.astype(np.uint8)
        rgb_blurred_img = cv2.cvtColor(blurred_img, cv2.COLOR_GRAY2RGB)
        contours,k = cv2.findContours(blurred_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        newContours=sorted(contours, key = cv2.contourArea, reverse = True)[:3]
        size=np.shape(img)
        height=size[0]
        width = size[1]
        Rects = []
        
        for cnt in newContours:
            epsilon = 0.0247*cv2.arcLength(cnt,True)  
            poly = cv2.approxPolyDP(cnt,epsilon,True)
            polyLength = len(poly)
            if polyLength >= 6 and polyLength <= 14:
                ratioCheck=cv2.fitEllipse(poly)
                trash1,lengths,trash2=ratioCheck
                bigLength=max(lengths)
                smolLength=min(lengths)
                ratioHelp = smolLength/bigLength
                ratio = np.sqrt(1-(ratioHelp**2))
                if ratioHelp > 0.75993421:  #ratio <0.65:
                    (a1,b1,c1,d1) = cv2.boundingRect(poly)
                    if c1*d1 >= 0.0001*height*width:
                        crnrs=[a1,height-b1-d1,a1+c1,height-b1]
                        Rects.append(crnrs)
                        Rects.sort(key=lambda x: x[0])
        return Rects

    def sigmoid(self, z):
        zArray = [z]
        return 1/(1+ np.exp(zArray))

if __name__ == '__main__':
    folder = "/Users/femw90/Google Drive/SECKSY (1)/Grad School/Winter Quarter 2020/ECE 276A/PR1/TestFolder"
    my_detector = StopSignDetector()
    #print("Hello World!")
    for filename in os.listdir(folder):
        # read one test image
        img = cv2.imread(os.path.join(folder,"19.jpg"))
        boxes = []
        boxes = my_detector.get_bounding_box(img)
        print (boxes)
        
        
        
        
        
        
        
        #boxes = np.int0(boxes)
        #im = cv2.drawContours(img,[boxes],0,(0,0,255),2)
        #cv2.imshow('image', im)
        cv2.waitKey(10)
        #cv2.destroyAllWindows()
        #cv2.imshow('image', img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        #Display results:
        #(1) Segmented images
        #     mask_img = my_detector.segment_image(img)
        #(2) Stop sign bounding box
        #    boxes = my_detector.get_bounding_box(img)
        #The autograder checks your answers to the functions segment_image() and get_bounding_box()
        #Make sure your code runs as expected on the testset before submitting to Gradescope

