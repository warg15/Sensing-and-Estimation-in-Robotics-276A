#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 16:26:09 2020

@author: Will Argus
"""
from scipy.linalg import inv
from scipy.linalg import expm
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

def load_data(file_name):
  '''
  function to read visual features, IMU measurements and calibration parameters
  Input:
      file_name: the input data file. Should look like "XXX_sync_KLT.npz"
  Output:
      t: time stamp
          with shape 1*t
      features: visual feature point coordinates in stereo images, 
          with shape 4*n*t, where n is number of features
      linear_velocity: IMU measurements in IMU frame
          with shape 3*t
      rotational_velocity: IMU measurements in IMU frame
          with shape 3*t
      K: (left)camera intrinsic matrix
          [fx  0 cx
            0 fy cy
            0  0  1]
          with shape 3*3
      b: stereo camera baseline
          with shape 1
      cam_T_imu: extrinsic matrix from IMU to (left)camera, in SE(3).
          close to 
          [ 0 -1  0 t1
            0  0 -1 t2
            1  0  0 t3
            0  0  0  1]
          with shape 4*4
  '''
  with np.load(file_name) as data:
      t = data["time_stamps"] # time_stamps
      features = data["features"] # 4 x num_features : pixel coordinates of features
      linear_velocity = data["linear_velocity"] # linear velocity measured in the body frame
      rotational_velocity = data["rotational_velocity"] # rotational velocity measured in the body frame
      K = data["K"] # intrindic calibration matrix
      b = data["b"] # baseline
      cam_T_imu = data["cam_T_imu"] # Transformation from imu to camera frame
  return t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu


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

def piProjMatrix (qVect): #remove z axis (depth information)
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
    file = "./data/0027.npz"
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
    vNoise = 10000
    skipSave = 100 #how often to save the current progress
    movNoise = 100
    W = movNoise*np.array([[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]])
    pMat = np.array([[1,0,0],[0,1,0], [0,0,1],[0,0,0]], dtype = float)
    M = createM(b,K)
    vI = vNoise*np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]])
    identity = np.eye(3)
    
    #loop through all time
    for j in range(1,numTime): #numTime
        
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
        
        ##### Predict car_mean and car_cov ########
        car_Mean=np.dot(expm(-dt*uHat),car_Mean)
        emap =expm(-dt*uHatPrime)
        car_Cov=emap@car_Cov@np.transpose(emap) +W
        
        ##### Save car state ########
        imuPose = IMU2world(car_Mean, t)
        car_StateSave[:,:,-1+j] = imuPose
        
        #################                         ###############
        
        
        
        ############## (b) Landmark Mapping via EKF Update ###################
        
        for k in range (numFeat):
            
            #all -1 means the mark isnt in current time instance
            featNow = features[:,k,j]
            if (featNow[0] == -1 and featNow[1] == -1 and featNow[2] == -1 and featNow[3] == -1 ): 
                continue
            
            #initialize landmark if it hasn't been yet
            if (marx_Mean[0,k] == -1 and marx_Mean[1,k] == -1 and marx_Mean[2,k] == -1 and marx_Mean[3,k] == -1 ): 
                #mean2world = IMU2world(car_Mean, t)
                camera2world = np.dot(imuPose,inv(cam_T_imu)) #transform camera to world frame
                zinit = ((K[0,0])*b)/(featNow[0]-featNow[2])
                #featCoords = np.ones((4,1)) # feature coords in in camera frame
                getFeatCoords = np.ones((3,1)) #pull coords from featNow
                getFeatCoords[0:2,0] = featNow[0:2]#pull coords from featNow
                temp = zinit*(inv(K)@getFeatCoords) #get coordinates of features
                featCoords = np.append(temp,[1])
                featCords2World = camera2world@featCoords #convert the coords to world frame from camera frame
                marx_Mean[:,k] = featCords2World
                continue
            
            #if landmark has already been initialized, and is in current time, must update it 
            #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
            jacob, zHat = GetHzHat(k, car_Mean, marx_Mean, M, pMat)
            #equations, lec 13, slide 19
            kal = marx_Cov[:,:,k] @ np.transpose(jacob) @ inv(jacob @ marx_Cov[:,:,k]@np.transpose(jacob) + vI)
            marx_Mean[:,k] = marx_Mean[:,k] + pMat @ kal @ (featNow - zHat)
            marx_Cov[:,:,k] = (np.eye(3) - kal @ jacob) @ marx_Cov[:,:,k]
        # save landmark trajectories pose
        #Landmarks['trajectory'][:, :, i] = Landmarks['mean'][:]
        marx_StateSave[:,:,j] = marx_Mean[:]
        
        
        
        # (c) Visual-Inertial SLAM

        if (j%skipSave == 0):
            utils.visualize_trajectory_2d(car_StateSave, marx_Mean, path_name=filename[7:-4],timestamp = str(j), show_ori=False)
            #visualize_trajectory_2d(Car['trajectory'], Landmarks['mean'], Car['trajectory_vi'], Landmarks['mean_vi'], timestamp = str(i), path_name = filename[7:-4], show_ori = True, show_grid = True)
            print(j)

    utils.visualize_trajectory_2d(car_StateSave, marx_Mean, path_name=filename[7:-4], timestamp = str(j), show_ori=False)





    # You can use the function below to visualize the robot pose over time
    #visualize_trajectory_2d(world_T_imu,show_ori=True)
    































