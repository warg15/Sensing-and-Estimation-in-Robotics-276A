#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thurs Feb 13 14:48:45 2020

@author: Will Argus
UCSD ECE 276A PR 2
With functions adapted from ECE276A PR2 package
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
import load_data
import p2_utils

def lidar2cartesian (t): #pass this the time instance we want to get the scan from
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T #array for angles, from load_data.py
    ranges = np.zeros((1081,1)) #array for ranges
    ranges[:,0] = np.double(l0[t]['scan'][0,:]).T #get scans range values
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1)) #limit scans to valid ranges
    ranges = ranges[indValid] #only keep range of valid scans 
    angles = angles[indValid] #only keep angles of valid scans 
    
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)]) #convert x scans to cartesian
    ys0 = np.array([ranges*np.sin(angles)]) #convert y scans to cartesian

    # convert position in the map frame here 
    Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0)  #put x and y coords in array together
    lenY = Y.shape #get length (number of valid coordinate pairs)
    addToY = np.ones((1, lenY[1])) #create array of 1's
    lidarState = np.zeros((4,lenY[1]))
    lidarState[0:3, :] = Y #create lidar state and assign x and y coords
    lidarState[3, :] = addToY #add a column of 1's for future transforations 
    return lidarState #returns lidar state in cartesian coords and lidar frame
    


def lidar2head (lidarState): #pass a numpy in lidar frame and get it back, converted to head frame
    hTl = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,.15],[0,0,0,1]]) #define transformation matrix
    h2l = np.matmul(hTl,lidarState) #apply transformation to lidar coords
    return h2l #return lidar coords in head frame

def findR (psi, theta, phi): #take in angles to make rotation matrix #yaw pitch roll, z-y-x
    rpsi = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]]) #create psi matrix
    rtheta = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]]) #create theta matrix
    rphi = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]]) #create phi matrix
    temp = np.matmul(rtheta, rphi) #multiply first 2 of the matrices
    R = np.matmul(rpsi, temp) #multiply the result with the third
    return R #return rotation matrix

def head2body (index, h2l): # pass a lidar scans in matrix in head frame and current index, converts to body frame
    idx = (np.abs(j0['ts'] - l0[index]['t'])).argmin() #get the closest time stamp for the joint angles
    if (j0['ts'][0,idx] > l0[index]['t']): #if the closest head angle time stamp is greater than the lidar time stamp
        idx = idx-1 #take the head angle time stamp just before the lidar time stamp
    psi = j0['head_angles'][0,idx] # get neck angle, yaw
    theta = -j0['head_angles'][1,idx] # get head angle, pitch
    phi  = 0 #set theta = 0
    r = findR(psi, theta, phi) #get rotation matrix
    bTh = np.zeros((4,4)) #create transformation matrix
    bTh[0:3,0:3] = r #add in rotation matrix
    lastRow = np.transpose([0, 0, .33, 1]) # createthe last row
    bTh[0:4, 3] = lastRow #add in the last row
    b2h = np.matmul(bTh, h2l) #transform to body
    return b2h #return lidar scans in body frame


def body2world (index, b2h, robot_state_meters): #pass in time index, lidar scans in body frame, and robot state
    x = robot_state_meters[0] #set x to robot state x
    y = robot_state_meters[1] #set y to robot state y 
    theta = robot_state_meters[2] #set theta
    r = findR(theta,0,0) #get rotation matrix
    wTb = np.zeros((4,4)) #create transformation matrix
    wTb[0:3,0:3] = r #add in rotation matrix
    lastRow = np.transpose([x, y, .93, 1]) #create last row
    wTb[0:4, 3] = lastRow #add in last row
    w2b = np.matmul(wTb, b2h) #perform transformation
    indValid = w2b[2,:] > 0.1 #get all indices of scans that dont hit the floor
    w2b = w2b[:,indValid] #keep only scans that dont hit the floor
    return w2b #return lidar scans in world frame
    

def meters2cells (coords, x_im, y_im): #input distance in coords, will convert it to distance in cells
    x = coords[0] #set x coords
    y = coords[1] #set y coords
    cX = (np.abs(x - x_im)).argmin() #find the closest cell to x coords
    cY = (np.abs(y - y_im)).argmin() #find the closest cell to y coords
    cells = np.zeros((1,2)) #create array for cells
    cells[0,0] = cX #set x cell
    cells[0,1] = cY #set y cell
    return cells #return the cell 

def meters2cellsVector (coords, x_im, y_im): #input distance in coords, will convert it to distance in cells
    xs0 = np.transpose(coords[0,:]) #get x coords
    ys0 = np.transpose(coords[1,:]) #get y coords
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1 #find closest cell for all x coords
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1 #find closest cell for all y coords
    cells = np.zeros((coords.shape[1],2)) #create array for all cell coords
    cells[:,0] = xis  #set x coords
    cells[:,1] = yis #set y coords
    return cells # return array of cells


def cells2meters (cells, x_im, y_im): #input coordinates of cell, will return distance from origin in meters
    x = x_im[np.int64(cells[0,0])] #set x to the value of meters in the given cell
    y = y_im[np.int64(cells[0,1])] #set y to the value of meters in the given cell
    coords = np.array([x,y]) #set the coords
    return coords #return the coords

def robotState2Cells (robot_state_meters): #takes robot state in meters and converts it to cells
    robot_xy_coords = np.array([robot_state_meters[0], robot_state_meters[1]]) #create array of the x y coords
    robot_coords_cells = meters2cells(robot_xy_coords, x_im, y_im) # convert to cells
    robot_state_cells = np.array([robot_coords_cells[0,0], robot_coords_cells[0,1], robot_state_meters[2]]) #re-assign cell coords into state array
    return robot_state_cells #return robot state

def robotUpdateState (i, robot_state_meters): #feed in current time index and robot state, returns updated state
    xRobot = l0[i]['delta_pose'][0,0] #find delta pose x
    yRobot = l0[i]['delta_pose'][0,1] #find delta pose y
    theta = l0[i]['delta_pose'][0,2] #find delta pose theta
    robot_delta_meters = np.array([xRobot, yRobot, theta]) #create array of delta pose
    robot_state_meters = np.add(robot_state_meters, robot_delta_meters)  #add delta pose
    return robot_state_meters #return updated robot state in meters

def MappingSmart (index, robot_state_meters): #pass in current time index and robot state, this will update the map
    robot_state_cells = robotState2Cells(robot_state_meters) #convert robot state
    lidarState = lidar2cartesian(index) #get lidar state
    #then convert to world frame
    h2l = lidar2head(lidarState) #transform lidar state
    b2h = head2body(index,h2l) #transform lidar state
    w2b = body2world(index, b2h,robot_state_meters) #transform lidar state

    cells = meters2cellsVector(w2b, x_im, y_im) #convert all scan coords to cells
    x = np.int32(cells[:,0]) #set x cells to type int32
    y = np.int32(cells[:,1]) #set y cells to type int32
    cells = np.array([x,y]) # make array of cells coords
    #Create Psuedo-Map
    MapSize = len(MAP['map']) #get size of map
    tempMap = np.zeros((MapSize,MapSize), dtype=np.float64) #create temp map
    
    cellsT = np.transpose(cells) #transpose cells
    szCells = cellsT.shape[0] #get size of cells 
    cellsC = np.zeros((szCells+1, 2), dtype=np.int32) #cerate int32 array to hold cells coords
    cellsC[0:szCells, 0:2] = cellsT #fill in array
    robotX = np.int32(robot_state_cells[0]) #get x cells
    robotY = np.int32(robot_state_cells[1]) #get y cells
    cellsC[szCells, 0:2] = [robotX, robotY] #fill array with x and y cell
    trash = np.zeros((1,2), dtype=np.int32) #create 2x1 array for contours list
    
    contours = list([trash]) #create list of arrays of ints for contours
    contours[0] = cellsC #add cells coords array to contours list
    logProb = math.log(10*lidarTrust) #calc log prob
    tempMap = cv2.drawContours(tempMap, contours, -1, -logProb, -1) #set entire area inside robot scan on temp map to -logProb
    tempMap = cv2.drawContours(tempMap, contours, -1, logProb, 1) #set extact scan coordinates (obstacles) to logProb
    MAP['map'] = np.add(MAP['map'], tempMap) #add temp map to log odds map
    return

def updateParticles (particles, i, numParticles): #pass in particles, currnet time index, number of particles update particles new movement and add noise
    xRobot, yRobot, theta = 0,0,0 #initialize variables
    xRobot = l0[i]['delta_pose'][0,0] #set x delta pose
    yRobot = l0[i]['delta_pose'][0,1] #set y delta pose
    theta = l0[i]['delta_pose'][0,2] #set theta delta pose
    robot_delta_meters = np.array([xRobot, yRobot, theta, 0]) #create delta pose array
    particles[:,:] = np.add(particles[:,:], robot_delta_meters) #add delta pose to all particles
    noise = np.zeros((numParticles, 4)) #create noise array
    if (xRobot != 0 or yRobot != 0): #only add noise if robot moves
        #mu, sigmax, sigmay, sigmatheta = 0, 0.002, 0.002, 0.005
        #mu, sigmax, sigmay, sigmatheta = 0, 0.005, 0.005, 0.005 # worked best so far 
        mu, sigmax, sigmay, sigmatheta = 0, 0.01, 0.01, 0.03 #set noise parameters
        #mu, sigmax, sigmay, sigmatheta = 0, 0.07, 0.07, 0.09
        noise[:,0] = np.random.normal(mu, sigmax, numParticles) #generate array of random x noise
        noise[:,1] = np.random.normal(mu, sigmay, numParticles) #generate array of random y noise
        noise[:,2] = np.random.normal(mu, sigmatheta, numParticles) #generate array of random theta noise
        particles[:,:] = np.add(particles, noise) #add the noise to all the particles
    return particles


if __name__ == '__main__':
    j0 = load_data.get_joint("joint/train_joint0") #load joint data
    l0 = load_data.get_lidar("lidar/train_lidar0") #load lidar data
    # init MAP
    MAP = {} #create map dict
    MAP['res']   = 0.05 #set resolution
    MAP['xmin']  = -60  #set x min meters
    MAP['ymin']  = -60 #set y min
    MAP['xmax']  =  60 #set xmax
    MAP['ymax']  =  60 #set ymax
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #get number of x cells
    MAP['sizey']= int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1)) #get number of y cells
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #create a map
    
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    #need coords of a cell? the cell equals the x and y indicies, and the value at these indices is the meters
    #need the cell number for your meter coords? call find nearest to get the index
    
    MapSize = len(MAP['map']) #get map size
    MAP['map'] = np.zeros((MapSize,MapSize)) #make map all zeros
    robot_state_meters = np.array([0,0,0]) #set iniital state to 0
    robot_state_cells = robotState2Cells(robot_state_meters) #convert state to cells
    robot_state_storep = np.zeros((len(l0),3)) #store state in cells

    #initialize variables
    lidarTrust = 0.9 #initialize lidar trust
    numParticles = 30 #initialize number of particles
    particles = np.zeros((numParticles, 4)) #each particle is its own row #goes [x, y, yaw, weight], IN METERS
    particles[:,3] = 1/numParticles #initialize particles weight
    maxWeight = np.argmax(particles, axis =0) #find max weight of each column
    wIndex = maxWeight[3] #get particle index with max weight
    robot_state_meters = [particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]] #set robot state as max weight particle
    MappingSmart(0, robot_state_meters) #update map
    particleCorrelationsGet = np.zeros((numParticles, len(l0))) #initialize particle correlations
    countResample = 0 #initialize particle resample counter
    skipScan = 1 #number of scans to move forward each time (1 means we do every scans)
    picSave = 1000 #how often save the map image (every 1000 scans)

    for i in range(1,len(l0)): # for all scans
        #=================save map perdiodically as it updates ======
        if (i%picSave) == 0 : #if its time to save map
            indPos = MAP['map'][:,:] > 0 #get positive indices
            indNeg = MAP['map'][:,:] < 0 #get negative indices
            MapSize = len(MAP['map']) #get map size
            ColorMap = np.zeros((MapSize,MapSize)) #create binary map
            ColorMap[indPos] = 1 #get positive cells to 1
            ColorMap[indNeg] = -1 #set negative cells to -1
            robot_state_int = robot_state_storep.astype(int) #make robot state an int
            ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2 #set all cells robot has been in to 2
            plt.imshow(ColorMap) #show binary map
            plt.savefig('ECE276A_PR2_Lidar0_PIC_'+str(i)+'.png', dpi=200) #save binary map
        
        #================= update the particles =================
        particles = updateParticles (particles, i, numParticles) #update particles


        if (i%skipScan) == 0 : #if we are using this scan
            #================= update the correlations =================
            particleCorrelations = np.zeros((numParticles)) #initialize array for correlations
            #create binary max to pass 
            indPos = MAP['map'][:,:] > 0 #get positive indices
            indNeg = MAP['map'][:,:] < 0 #get negative indices
            MapSize = len(MAP['map']) #get map size
            binaryMap = np.zeros((MapSize,MapSize)) #create binary occupany map
            binaryMap[indPos] = 1 #set positive cels to 1
            binaryMap[indNeg] = 0 #set negative cells to -1
            
            #get the lidar state
            lidarState = lidar2cartesian(i) #get the lidar scans in cartesian
            h2l = lidar2head(lidarState) #convert to head frame
            b2h = head2body(i,h2l) #convert to body frame
            
            for j in range(numParticles): #for all particles
                currentParticle = particles[j,:] #set current particle
                w2b = body2world(i, b2h,currentParticle[0:3]) #transform lidar to world w/ current particle
                cells = meters2cellsVector(w2b, x_im, y_im) #convert lidar to cells
                x = np.int32(cells[:,0]) #convert x cells coords to int32
                y = np.int32(cells[:,1]) #convert y cells coords to int32
                cells = np.array([x,y]) #put cell coords in array
                particleCorrelationsGet[j,i] = binaryMap[cells[1,:],cells[0,:]].sum() #sum all points where lidar scan hits obstacle
            particleCorrelations = softmax(particleCorrelationsGet[:,i]) #get softmax of all correlations
            particles[:,3] = particleCorrelations #set all correlations
                
            #================= choose max weight particle =================
            maxWeight = np.argmax(particles, axis =0) #find max weight of each column
            wIndex = maxWeight[3] #get particel indice of max weight
            robot_state_meters = np.array([particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]]) #set robot state to largest weight particle
            robot_state_cells = robotState2Cells(robot_state_meters) #convert to cells
            robot_state_storep[i,:] = robot_state_cells #save state
            
            #================= update the map =================
            MappingSmart(i, robot_state_meters) #map with new state
            
            #================= particle resampling =================
            nEff = 1/sum(np.square(particles[:,3])) #find the one over the sum of the squares of the weight
            nThresh = .1*numParticles #set threshold
            if nEff <= nThresh: #if it is time to resample
                countResample = countResample+1 #increment the count
                #This is the stratified resampling algorithm in the lecture slides
                j, c = 1, particles[0,3] #set j and c
                particlesNew = np.zeros((numParticles,4)) #create array to store new particles
                for k in range(numParticles): #for all particles
                    b = np.random.uniform(0, 1/numParticles) + (k)/numParticles # initialize b
                    while b > c: #while b more than c
                        j = j+1 #increment j
                        c = c + particles[j-1,3] #add particles correlation to c
                    particlesNew[k,0] = particles[j-1,0] #set x on new particle
                    particlesNew[k,1] = particles[j-1,1] #set y on new particle
                    particlesNew[k,2] = particles[j-1,2] #set theta on new particle
                    particlesNew[k,3] = 1/numParticles #set weight

                particles = particlesNew #set particles as the resampled particles
                
    indPos = MAP['map'][:,:] > 0 #get positive indices
    indNeg = MAP['map'][:,:] < 0 #get negative indices
    MapSize = len(MAP['map']) #get map size
    ColorMap = np.zeros((MapSize,MapSize)) #create binary map
    ColorMap[indPos] = 1 #set positive indicies to 1
    ColorMap[indNeg] = -1 #set negative indices to -1
    robot_state_int = robot_state_storep.astype(int) #set state as an int
    ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2 #set all robot's states as 2
    plt.imshow(ColorMap) #show the mape
    plt.colorbar() #add colorbar
    plt.savefig('Lidar0,1.png', dpi=500) #save map
    plt.show() #show in console
    
    plt.imshow(MAP['map']) #show log odds map
    plt.colorbar() #add colorbar
    plt.savefig('Lidar0,2.png', dpi=500) #save it
    plt.show() #show in console
    np.save('Lidar0', MAP['map']) #save map numpy array
