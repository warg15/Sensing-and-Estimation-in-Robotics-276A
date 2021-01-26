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

####### #use 10500 to get straight result?
#used 9800, 10000, 10500, 110000
#0.000001/05 on the linear noise
########################IMPORTANT####################################
#Due to memory problems, when running this code, either run it without 
#SLAM (by commenting out Part C in the “j” loop of the main function, or 
#run it only with SLAM (by commenting out the parts A and B). 
#If both are tried to run at the same time, the user will not get desired output as a result of memory issues due to the large amount of data being inputted.
#######

def load_data(file_name): #function given in utils
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

def visualize_trajectory_2d(pose,landmarks,path_name="Unknown",show_ori=False): #function given in utils
  '''
  function to visualize the trajectory in 2D
  Input:
      pose:   4*4*N matrix representing the camera pose, 
              where N is the number of pose, and each
              4*4 matrix is in SE(3)
  '''
  fig,ax = plt.subplots(figsize=(5,5))
  n_pose = pose.shape[2]
  ax.plot(landmarks[0,:],landmarks[1,:],'b.',markersize=1.5,label='Landmarx') #added to also plot landmarks
  ax.plot(pose[0,3,:],pose[1,3,:],'r-',label=path_name)
  ax.scatter(pose[0,3,0],pose[1,3,0],marker='s',label="start")
  ax.scatter(pose[0,3,-1],pose[1,3,-1],marker='o',label="end")
  if show_ori:
      select_ori_index = list(range(0,n_pose,max(int(n_pose/50), 1)))
      yaw_list = []
      for i in select_ori_index:
          _,_,yaw = mat2euler(pose[:3,:3,i])
          yaw_list.append(yaw)
      dx = np.cos(yaw_list)
      dy = np.sin(yaw_list)
      dx,dy = [dx,dy]/np.sqrt(dx**2+dy**2)
      ax.quiver(pose[0,3,select_ori_index],pose[1,3,select_ori_index],dx,dy,\
          color="b",units="xy",width=1)
  ax.set_xlabel('x')
  ax.set_ylabel('y')
  ax.axis('equal')
  ax.grid(False)
  ax.legend()
  plt.show(block=True)
  return fig, ax

def vect2hat(a): #a is vector with 3 elements
    hat = np.array([[   0,  -a[2],  a[1]], [a[2],      0, -a[0]], [-a[1],  a[0],    0]])
    #hat is the linear map used for computing cross product
    return hat

def IMU2world(Mean, time):
    #find pose (described by matrix in SE(3) group)
    imuPose = np.zeros((4,4))
    imuPose[3,3] =1
    imuPose[0:3,0:3] = np.transpose(Mean[0:3,0:3])
    imuPose[0:3,3] = -np.transpose(Mean[0:3,0:3])@Mean[0:3,3]
    return imuPose

def createM(b,K): #create the M matrix using given K and b
    return np.array([[K[0,0],0,K[0,2],0], [0,K[1,1],K[1,2],0], [K[0,0],0,K[0,2],-b*K[0,0]], [0,K[1,1],K[1,2],0]])

def piProjMatrix (qVect): #remove z axis (depth information)
    return qVect/qVect[2]

def dPidQ (qVect): #find the derivative of the projection function
    matrix = np.array([[1, 0,-qVect[0]/qVect[2],0],[0,1,-qVect[1]/qVect[2],0],[0,0,0,0],[0,0,-qVect[3]/qVect[2],1]])
    matrix = (1/qVect[2])*(matrix)
    return matrix

def GetHzHat(k, car_Mean, marx_Mean, M, P): #get and return the jacobian and z-Hat
    world2camera = np.dot(cam_T_imu, car_Mean) #get world2camera transform matrix
    markInCam = np.dot(world2camera, marx_Mean[:,k]) #get the coords of landmark in camera frame
    #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
    zHat = np.dot(M,piProjMatrix(markInCam)) #+ vm #compute zhat (lec 13,slide 19)
    jacob = M@dPidQ(markInCam)@world2camera@P #compute jacobian (lec 13,slide 19)
    return jacob, zHat

def UpdateMarxMean(featNow, imuPose, cam_T_imu): #update the 
    #mean2world = IMU2world(car_Mean, t)
    camera2world = np.dot(imuPose,inv(cam_T_imu)) #transform camera to world frame
    zinit = ((K[0,0])*b)/(featNow[0]-featNow[2])
    #featCoords = np.ones((4,1)) # feature coords in in camera frame
    getFeatCoords = np.ones((3,1)) #pull coords from featNow
    getFeatCoords[0:2,0] = featNow[0:2]#pull coords from featNow
    temp = zinit*(inv(K)@getFeatCoords) #get coordinates of features
    featCoords = np.append(temp,[1])
    featCords2World = camera2world@featCoords #convert the coords to world frame from camera frame
    #marx_Mean[:,k] = featCords2World
    return featCords2World

def getUHat(j,rotational_velocity, linear_velocity): #get the value of uHat
    uHat = np.zeros((4,4)) #create
    w = rotational_velocity[:,j] #make vector
    w_hat = vect2hat(w) #calc W hat
    v = linear_velocity[:,j] #linear velocity
    v_hat = vect2hat(v) #create v-hat
    uHat[0:3,0:3] = w_hat #populate the uHat matrix
    uHat[0:3,3] = v #populate the uHat matrix
    uHatPrime=np.zeros((6,6))  #populate the uHatPrime  matrix
    uHatPrime[0:3,0:3] = w_hat
    uHatPrime[3:6,3:6] = w_hat
    uHatPrime[0:3,3:6] = v_hat
    return uHat, uHatPrime

def vect2hat46(a): #a is vector with 6 elements
    hat = np.array([[ 0, -a[5], a[4], -a[0]], [a[5], 0, -a[3], -a[1]], [-a[4], a[3], 0, -a[2]], [0,0,0,0]])
    #hat is the linear map used for computing cross product, 6 elements
    return hat

if __name__ == '__main__':
    ###### Import data with provided function ########
    file = "./data/0027.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(file) #create data
    # t: time stamps
    #pixel coordinates zt in R(4x M) of detected visual features with precomputed 
    #        correspondences between the left and the right camera frames
    #        4 (x1,y1,x2,y2) by 3220 (number of features), by 800 (number of time stamps)
    # linear velocity vt in R3 and angular velocity wt in R3 measured in the body frame of the IMU
    # stereo baseline b and camera calibration matrix, K
    # the transformation CTI 2 SE(3) from the IMU to left camera frame
    
    
    
    
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
    
    skipSave = 100 #how often to save the current progress
    vNoise = 1345#10000#3500 #movement noise
    WNoise = 100 #w noise
    W = WNoise*np.array([[1,0,0,0,0,0],[0,1,0,0,0,0], [0,0,1,0,0,0], [0,0,0,1,0,0], [0,0,0,0,1,0], [0,0,0,0,0,1]]) #noise to add to car_Cov
    pMat = np.array([[1,0,0],[0,1,0], [0,0,1],[0,0,0]], dtype = float)
    M = createM(b,K) #M matrix, used in finding jacobian, lecture 13, slide 19
    vI = vNoise*np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]]) #identity matrix times noise, for calculating kalman gain, observation noise
    movNoise = 234#100
    vm = movNoise*np.array([[1,0,0,0],[0,1,0,0], [0,0,1,0], [0,0,0,1]])
    identity = np.eye(3) #identity Matrix
    identityC = np.eye(6) #identity Matrix
    
    #Variables specific to Part C
    car_MeanC = car_Mean[:] #car mean for SLAM
    car_CovC = car_Cov[:] #car cov for SLAM
    car_StateSaveC = car_StateSave[:] #save car traj. during SLAM
    marx_MeanC = marx_Mean[:] #landmarks mean during SLAM
    marx_CovC = marx_Cov[:] #landmarks cov during SLAM
    marx_StateSaveC = marx_StateSave[:] #save landmarks states during SLAM
    
    ################### loop through all time ###################
    for j in range(1,numTime): #numTime
        
        dt = np.abs(t[0,j] - t[0,j-1])
        '''
        ############## (a) IMU Localization via EKF Prediction ###################
        
        ##### find variables for predicting car_mean and car_cov ########
        uHat, uHatPrime = getUHat(j,rotational_velocity, linear_velocity)
        
        ##### Predict car_mean and car_cov ########
        car_Mean=np.dot(expm(-dt*uHat),car_Mean)
        emap =expm(-dt*uHatPrime)
        car_Cov=emap@car_Cov@np.transpose(emap) +W
        
        ##### Save car state ########
        imuPose = IMU2world(car_Mean, t)
        car_StateSave[:,:,j] = imuPose
        marx_StateSave[:,:,-1+j] = marx_Mean[:] #not necessary here but may be used in part C
        
        #imuPose = IMU2world(car_MeanC, t)
        #car_StateSaveC[:,:,j] = imuPose
        #marx_StateSaveC[:,:,-1+j] = marx_MeanC[:] #not necessary here but may be used in part C
        
        ############## (b) Landmark Mapping via EKF Update ###################
        
        for k in range (numFeat):
            #all -1 means the mark isnt in current time instance
            featNow = features[:,k,j]
            if (featNow[0] == -1 and featNow[1] == -1 and featNow[2] == -1 and featNow[3] == -1 ): 
                continue
            
            #initialize landmark if it hasn't been yet
            if (marx_Mean[0,k] == -1 and marx_Mean[1,k] == -1 and marx_Mean[2,k] == -1 and marx_Mean[3,k] == -1 ): 
                marx_Mean[:,k] = UpdateMarxMean(featNow, imuPose, cam_T_imu)
                continue
            
            #if landmark has already been initialized, and is in current time, must update it 
            #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
            jacob, zHat = GetHzHat(k, car_Mean, marx_Mean, M, pMat)
            #equations, lec 13, slide 19
            kal = marx_Cov[:,:,k] @ np.transpose(jacob) @ inv(jacob @ marx_Cov[:,:,k]@np.transpose(jacob) + vI)
            marx_Mean[:,k] = marx_Mean[:,k] + pMat @ kal @ (featNow - zHat)
            marx_Cov[:,:,k] = (np.eye(3) - kal @ jacob) @ marx_Cov[:,:,k]
        # save landmark trajectories pose
        #marx_StateSave[:,:,j] = marx_Mean[:] #not necessary here but may be used in part C
        
        '''
        
        ############## (c) Visual-Inertial SLAM ###################
        #Part 1: visual-inertial SLAM prediction
        v_value = 0.00007#0.00001 #move motion noise
        o_value = 0.000002 #rot motion noise
        WPrime = np.zeros((6,6))
        WPrime[0,0], WPrime[1,1], WPrime[2,2], WPrime[3,3], WPrime[4,4], WPrime[5,5] = v_value,v_value,v_value,o_value,o_value,o_value
        uHat, uHatPrime = getUHat(j,rotational_velocity, linear_velocity)
        
        ##### Predict car_mean and car_cov ########
        car_MeanC=np.dot(expm(-dt*uHat),car_MeanC)
        emap =expm(-dt*uHatPrime)
        car_CovC=emap@car_CovC@np.transpose(emap) + WPrime
        
        ##### Save car state ########
        #imuPose = IMU2world(car_MeanC, t)
        #car_StateSaveC[:,:,-1+j] = imuPose
        imuPose = IMU2world(car_MeanC, t)
        car_StateSaveC[:,:,j] = imuPose
        marx_StateSaveC[:,:,-1+j] = marx_MeanC[:] #not necessary here but may be used in part C
        
        
        #Part 2: visual-inertial SLAM Update
        
        for k in range (numFeat):
            #all -1 means the mark isnt in current time instance
            featNowC = features[:,k,j]
            if (featNowC[0] == -1 and featNowC[1] == -1 and featNowC[2] == -1 and featNowC[3] == -1 ): 
                continue
            
            #initialize landmark if it hasn't been yet
            if (marx_MeanC[0,k] == -1 and marx_MeanC[1,k] == -1 and marx_MeanC[2,k] == -1 and marx_MeanC[3,k] == -1 ): 
                marx_MeanC[:,k] = UpdateMarxMean(featNowC, imuPose, cam_T_imu)
                continue
            #print(j)
            #if landmark has already been initialized, and is in current time, must update it 
            #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
            jacobC, zHatC = GetHzHat(k, car_MeanC, marx_MeanC, M, pMat)
            #equations, lec 13, slide 19
            kalC = marx_CovC[:,:,k] @ np.transpose(jacobC) @ inv(jacobC @ marx_CovC[:,:,k]@np.transpose(jacobC) + vI)
            marx_MeanC[:,k] = marx_MeanC[:,k] + pMat @ kalC @ (featNowC - zHatC)
            marx_CovC[:,:,k] = (np.eye(3) - kalC @ jacobC) @ marx_CovC[:,:,k]
            
            
            #now update the car_mean
            #perform kalman update on the mean 1 landmark at a time
            #find Jacobian of zt+1 w/ respect to Ut+1, evaluated at ut+1|t
            temp = np.zeros((4,6))
            temp[0,0], temp[1,1], temp[2,2] = 1,1,1
            temp[0:3,3:6] = -1*vect2hat(car_MeanC@marx_MeanC[:,k])
            jacobC = M@dPidQ(cam_T_imu@ car_MeanC @ marx_MeanC[:,k])@cam_T_imu@temp
            
            kalC = car_CovC @ np.transpose(jacobC) @ inv(jacobC @ car_CovC @ np.transpose(jacobC) + vI)
            car_MeanC = expm(vect2hat46(kalC@(featNowC-zHatC)))@car_MeanC
            #car_MeanC = kalC@(featNowC-zHatC)
            car_CovC = (identityC - kalC@jacobC)@car_CovC

            
        if (j%skipSave == 0):
            #utils.visualize_trajectory_2d(car_StateSave, marx_Mean, path_name="Unknown",show_ori=False)
            utils.visualize_trajectory_2d(car_StateSaveC, marx_MeanC, timestamp = str(j),path_name=file[7:-4], show_ori=False)
            print(j)

    #visualize_trajectory_2d(car_StateSave, marx_Mean, path_name="PartA/B",show_ori=False)
    #visualize_trajectory_2d(car_StateSaveC, marx_MeanC, path_name="SLAM",show_ori=False)
    utils.visualize_trajectory_2d(car_StateSaveC, marx_MeanC, timestamp = str(j),path_name=file[7:-4], show_ori=False)































