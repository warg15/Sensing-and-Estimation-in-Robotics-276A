#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:26:09 2020

@author: Will Argus
"""
import numpy as np
from scipy import io
from scipy.special import softmax
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import os
import cv2
import math
import utils
import matplotlib.pyplot as plt
from transforms3d.euler import mat2euler
from utils import *

def vect2hat(a): #a is vector with 3 elements
    hat = np.array([[   0,  -a[2],  a[1]], [a[2],      0, -a[0]], [-a[1],  a[0],    0]])
    #hat is the linear map used for computing cross product
    return hat

def IMU2world(car_Mean, time):
    #find pose (described by matrix in SE(3) group)
    imuPose = np.zeros((4,4))
    imuPose[3,3] =1
    imuPose[0:3,0:3] = np.transpose(car_Mean[0:3,0:3])
    imuPose[0:3,3] = -np.transpose(car_Mean[0:3,0:3])@car_Mean[0:3,3]
    return imuPose

def createM(b,K):
    return np.array([[K[0,0],0,K[0,2],0], [0,K[1,1],K[1,2],0], [K[0,0],0,K[0,2],-b*K[0,0]], [0,K[1,1],K[1,2],0]])

def piProjMatrix (qVect):
    return qVect/qVect[2]

def dPidQ (qVect):
    matrix = np.array([[1, 0,-qVect[0]/qVect[2],0],[0,1,-qVect[1]/qVect[2],0],[0,0,0,0],[0,0,-qVect[3]/qVect[2],0]])
    matrix = (1/qVect[2])*(matrix)
    return matrix

def GetHzHat(k, car_Mean, marx_Mean, M, P):
    world2camera = np.dot(cam_T_imu, car_Mean) #get world2camera transform matrix
    markInCam = np.dot(world2camera, marx_Mean[:,k]) #get the coords of landmark in camera frame
    #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
    zHat = np.dot(M,piProjMatrix(markInCam)) #compute zhat (lec 13,slide 19)
    jacob = M@dPidQ(markInCam)@world2camera@P #compute jacobian (lec 13,slide 19)
    return jacob, zHat


if __name__ == '__main__':
    ###### Import data with provided function ########
    file = "./data/0022.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(file)
    # t: time stamps
    #pixel coordinates zt in R(4x M) of detected visual features with precomputed 
    #        correspondences between the left and the right camera frames
    #        4 (x1,y1,x2,y2) by 3220 (number of features), by 800 (number of time stamps)
    # linear velocity vt in R3 and angular velocity wt in R3 measured in the body frame of the IMU
    # stereo baseline b and camera calibration matrix, K
    # the transformation CTI 2 SE(3) from the IMU to left camera frame
    a=t[0,1]-t[0,0]
    
    ###### Initialize Variables ########
    car_Mean = np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]], dtype = float)
    initCov = 0.01
    car_Cov = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    car_Cov = car_Cov*initCov
    
    numFeat = features.shape[1]
    marx_Cov = np.zeros((3,3,numFeat))
    k=0
    while k < numFeat:
        np.fill_diagonal(marx_Cov[:,:,k], initCov)
        k=k+1
    marx_Mean = np.ones((4,numFeat))
    marx_Mean = marx_Mean*-1 #-1 means the feature has not been initialized yet

    #need to save  car trajectory (save all the )
    numTime = t.shape[1]
    car_StateSave = np.zeros((4,4,numTime))
    marx_StateSave = np.zeros((4,numFeat, numTime))
    marx_StateSave[:,:,0] = marx_Mean
    
    skipSave = 1 #how often to save the current progress
    movNoise = 100
    W = movNoise*np.array([[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    pMat = np.array([[1,0,0],[0,1,0], [0,0,1],[0,0,0]], dtype = float)
    M = createM(b,K)
    vI = movNoise*np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]])
    identity = np.eye(3)
    
    #loop through all time
    for j in range(1,5): #numTime
        
        dt = t[0,j] - t[0,j-1]
        #find mean and cov, lec13, slide 16
        
        
        ############## (a) IMU Localization via EKF Prediction ###################
        
        ##### find variables for predicting car_mean and car_cov ########
        uHat = np.zeros((4,4))
        w = rotational_velocity[:,j]
        w_hat = vect2hat(w)
        v = linear_velocity[:,j]
        v_hat = vect2hat(v)
        uHat[0:3,0:3] = w_hat
        uHat[0:3,3] = v
        uHatPrime=np.zeros((6,6))
        uHatPrime[0:3,0:3] = w_hat
        uHatPrime[3:6,3:6] = w_hat
        uHatPrime[0:3,3:6] = v_hat
        #################                         ###############
        
        ##### Predict car_mean and car_cov ########
        car_Mean=np.dot(expm(-dt*uHat),car_Mean)
        emap =expm(-dt*uHatPrime)
        #car_Cov=np.dot(emap, car_Cov@np.transpose(emap)))+W
        car_Cov=emap@car_Cov@np.transpose(emap) +W
        #################                         ###############
        
        ##### Save car state ########
        imuPose = IMU2world(car_Mean, t)
        car_StateSave[:,:,-1+j] = imuPose
        #################                         ###############
        
        
        
        ############## (b) Landmark Mapping via EKF Update ###################
        
        for k in range (numFeat):
            featNow = features[:,k,j]
            
            #all -1 means the mark isnt in current time instance
            if (featNow[0] == -1 and featNow[1] == -1 and featNow[2] == -1 and featNow[3] == -1 ): 
                continue
            
            #initialize landmark if it hasn't been yet
            if (marx_Mean[0,k] == -1 and marx_Mean[1,k] == -1 and marx_Mean[2,k] == -1 and marx_Mean[3,k] == -1 ): 
                #mean2world = IMU2world(car_Mean, t)
                camera2world = np.dot(imuPose,np.linalg.inv(cam_T_imu)) #transform camera to world frame
                zinit = ((K[0,0])*b)/(featNow[0]-featNow[2])
                #featCoords = np.ones((4,1)) # feature coords in in camera frame
                getFeatCoords = np.ones((3,1)) #pull coords from featNow
                getFeatCoords[0:2,0] = featNow[0:2]#pull coords from featNow
                temp = zinit*(np.linalg.inv(K)@getFeatCoords) #get coordinates of features
                featCoords = np.append(temp,[1])
                featCords2World = camera2world@featCoords #convert the coords to world frame from camera frame
                marx_Mean[:,k] = featCords2World
                continue
            
            #if landmark has already been initialized, and is in current time, must update it 
            
            #world2camera = np.dot(cam_T_imu, car_Mean)
            #markInCam = np.dot(world2camera, marx_Mean[:,k])
            #zHat = np.dot(M,piProjMatrix(markInCam))
            #jacob = M@dPidQ(markInCam)@world2camera@P
            #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
            jacob, zHat = GetHzHat(k, car_Mean, marx_Mean, M, pMat)
            
            #k = jacob @ marx_Cov[:,:,k]@np.transpose(H) 
            kal = marx_Cov[:,:,k] @ np.transpose(jacob) @ np.linalg.inv(jacob @ marx_Cov[:,:,k]@np.transpose(jacob) + vI)
            #update the mean and cov of the landmark (lec 13, slide 19)
            #marx_Mean[:,k] = marx_Mean[:,k] + pMat @ kal @ (featNow-zHat)
            #marx_Cov[:,:,k] = (np.eye(3) - kal @ jacob) @ marx_Cov[:,:,k]
            marx_Mean[:,k] = marx_Mean[:,k] + pMat @ kal @ (featNow - zHat)
            marx_Cov[:,:,k] = (np.eye(3) - kal @ jacob) @ marx_Cov[:,:,k]
        # record current pose
        #Landmarks['trajectory'][:, :, i] = Landmarks['mean'][:]
        
    # (a) IMU Localization via EKF Prediction
    # (b) Landmark Mapping via EKF Update
    # (c) Visual-Inertial SLAM
    # You can use the function below to visualize the robot pose over time
    #visualize_trajectory_2d(world_T_imu,show_ori=True)
    
    
'''
def EKF_visual_update(Car, Landmarks, tau, curr_features, K, b, cam_T_imu, weight = 100):
    # covariance for measurement noise
    V = weight * np.eye(4)
    
    P = np.eye(3, 4)
    M = stereo_camera_model(K, b)
    
    for i in range(curr_features.shape[1]):
        
        z = curr_features[:, i][:]
        
        # only operate for landmarks present in current timestep
        if (np.all(z == -1)):
            continue
        
        # else if we make it here, that means the current landmark is present in the camera frame.
        # if, in the previous timestep, the landmark wasn't present, initialize the landmark now
        # using the car's pose
        if (np.all(np.isnan(Landmarks['mean'][:, i]))):
            d = (z[0] - z[2])
            Z_0 = (K[0, 0] * b) / d
            
            world_T_cam = world_T_imu(Car['mean']) @ np.linalg.inv(cam_T_imu)
            
            camera_frame_coords = np.hstack((Z_0 * np.linalg.inv(K) @ np.hstack((z[:2], 1)), 1))
            
            Landmarks['mean'][:, i] = world_T_cam @ camera_frame_coords
            
            continue 
        
            # else if landmark is present in the current timestamp, and has been seen before
            # create predicted z_hat from previous z (in camera-frame coordinates)
            cam_T_world = cam_T_imu @ Car['mean']
            curr_landmark = cam_T_world @ Landmarks['mean'][:, i]
            z_hat = M @ projection(curr_landmark) # remove depth information via projection, and project to pixels
            
            # form H; the Jacobian of z_hat w.r.t. current feature m evaluated at car's current position
            H = M @ projection_derivative(curr_landmark) @ cam_T_world @ P.T
            
            # perform the EKF update
            KG = Landmarks['covariance'][:, :, i] @ H.T @ np.linalg.inv(H @ Landmarks['covariance'][:, :, i] @ H.T + V)
            
            Landmarks['mean'][:, i] = Landmarks['mean'][:, i] + P.T @ KG @ (z - z_hat)
            Landmarks['covariance'][:, :, i] = (np.eye(3) - KG @ H) @ Landmarks['covariance'][:, :, i]
'''












