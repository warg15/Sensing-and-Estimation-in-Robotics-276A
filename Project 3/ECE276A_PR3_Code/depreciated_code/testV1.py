#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 13:15:23 2020

@author: femw90
"""
import numpy as np
from p3_utils import *

def EKF_imu_prediction(Car, v, omega, tau, weight = 100):
    # covariance for movement noise
    W = weight * np.eye(6)
    
    tau = -(tau)
    
    #u = np.vstack((v, omega)) # control input
    u_hat = np.vstack((np.hstack((hat_map(omega), v.reshape(3, 1))), np.zeros((1, 4))))    
    u_curlyhat = np.block([[  hat_map(omega),     hat_map(v)], 
                           [np.zeros((3, 3)), hat_map(omega)]])
    
    Car['mean'] = expm(tau * u_hat) @ Car['mean']
    Car['covariance'] = expm(tau * u_curlyhat) @ Car['covariance'] @ np.transpose(expm(tau * u_curlyhat)) + W

if __name__ == '__main__':
    filename = "./data/0022.npz"
    t, features, linear_velocity, rotational_velocity, K, b, cam_T_imu = load_data(filename)

    # initialize robots and landmarks
    Car = initCar(t.shape[1])
    Landmarks = initLandmarks(features.shape[1], t.shape[1])
    
    for i in range(1,3): #t.shape[1]
        print(i)
        if i == 0:
            continue
        
        tau = np.abs(t[0, i] - t[0, i - 1])
        
    	# (a) IMU Localization via EKF Prediction
        EKF_imu_prediction(Car, linear_velocity[:, i], rotational_velocity[:, i], tau)
        
        mean_pose = Car['mean']
        R_T = np.transpose(mean_pose[:3, :3])
        p = mean_pose[:3, 3].reshape(3, 1)
        U_inv = np.vstack((np.hstack((R_T, -np.dot(R_T, p))), np.array([0, 0, 0, 1])))
        
        
        # record current pose
        Car['trajectory'][:, :, i] = U_inv  #world_T_imu(Car['mean']) # inv(inv pose)
        Landmarks['trajectory'][:, :, i - 1] = Landmarks['mean'][:]
        
        
        #EKF_visual_update(Car, Landmarks, tau, features[:, :, i], K, b, cam_T_imu)
            
        #def EKF_visual_update(Car, Landmarks, tau, curr_features, K, b, cam_T_imu, weight = 100):
        weight = 100
        curr_features = features[:, :, i]
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



