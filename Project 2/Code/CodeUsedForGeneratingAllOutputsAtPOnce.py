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

    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.zeros((1081,1))
    ranges[:,0] = np.double(l0[t]['scan'][0,:]).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])

    # convert position in the map frame here 
    Y = np.concatenate([np.concatenate([xs0,ys0],axis=0),np.zeros(xs0.shape)],axis=0)
    lenY = Y.shape
    addToY = np.ones((1, lenY[1]))
    lidarState = np.zeros((4,lenY[1]))
    lidarState[0:3, :] = Y
    lidarState[3, :] = addToY
    return lidarState
    


def lidar2head (lidarState): #pass a numpy in lidar frame and get it back, converted to head frame
    hTl = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,.15],[0,0,0,1]])
    h2l = np.matmul(hTl,lidarState)
    return h2l

def findR (psi, theta, phi): #yaw pitch roll, z-y-x
    rpsi = np.array([[math.cos(psi), -math.sin(psi), 0], [math.sin(psi), math.cos(psi), 0], [0, 0, 1]])
    rtheta = np.array([[math.cos(theta), 0, math.sin(theta)], [0, 1, 0], [-math.sin(theta), 0, math.cos(theta)]])
    rphi = np.array([[1, 0, 0], [0, math.cos(phi), -math.sin(phi)], [0, math.sin(phi), math.cos(phi)]])
    temp = np.matmul(rtheta, rphi)
    R = np.matmul(rpsi, temp)
    return R

def head2body (index, h2l): # pass a lidar state in head frame and current index, converts to body frame
    idx = (np.abs(j0['ts'] - l0[index]['t'])).argmin()
    if (j0['ts'][0,idx] > l0[index]['t']): #if the closest head angle time stamp is greater than the lidar time stamp
        idx = idx-1 #take the head angle time stamp just before the lidar time stamp
    psi = j0['head_angles'][0,idx] # neg neck angle, yaw
    theta = -j0['head_angles'][1,idx] # neg head angle, pitch
    phi = 0
    r = findR(psi, theta, phi)
    bTh = np.zeros((4,4))
    bTh[0:3,0:3] = r
    lastRow = np.transpose([0, 0, .33, 1])
    bTh[0:4, 3] = lastRow
    b2h = np.matmul(bTh, h2l)
    return b2h


def body2world (index, b2h, robot_state_meters):
    x = robot_state_meters[0]
    y = robot_state_meters[1]
    theta = robot_state_meters[2]
    r = findR(theta,0,0)
    wTb = np.zeros((4,4))
    wTb[0:3,0:3] = r
    lastRow = np.transpose([x, y, .93, 1])
    wTb[0:4, 3] = lastRow
    w2b = np.matmul(wTb, b2h)
    indValid = w2b[2,:] > 0.1
    w2b = w2b[:,indValid]
    return w2b
    

def meters2cells (coords, x_im, y_im): #input distance in coords, will convert it to distance in cells
    #how to use:     
        #coords = np.array([0,19.99])
        #cells = meters2cells(coords, x_im, y_im)
    x = coords[0]
    y = coords[1]
    cX = (np.abs(x - x_im)).argmin()
    cY = (np.abs(y - y_im)).argmin()
    cells = np.zeros((1,2))
    cells[0,0] = cX 
    cells[0,1] = cY
    return cells

def meters2cellsVector (coords, x_im, y_im): #input distance in coords, will convert it to distance in cells
    #how to use:     
        #coords = np.array([0,19.99])
        #cells = meters2cells(coords, x_im, y_im)
    xs0 = np.transpose(coords[0,:])
    ys0 = np.transpose(coords[1,:])
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    cells = np.zeros((coords.shape[1],2))
    cells[:,0] = xis 
    cells[:,1] = yis
    return cells


def cells2meters (cells, x_im, y_im): #input coordinates of cell, will return distance from origin in meters
    x = x_im[np.int64(cells[0,0])]
    y = y_im[np.int64(cells[0,1])]
    coords = np.array([x,y])
    return coords

def robotState2Cells (robot_state_meters):
    robot_xy_coords = np.array([robot_state_meters[0], robot_state_meters[1]])
    robot_coords_cells = meters2cells(robot_xy_coords, x_im, y_im)
    robot_state_cells = np.array([robot_coords_cells[0,0], robot_coords_cells[0,1], robot_state_meters[2]])
    return robot_state_cells

def robotUpdateState (i, robot_state_meters):
    xRobot = l0[i]['delta_pose'][0,0]
    yRobot = l0[i]['delta_pose'][0,1]
    theta = l0[i]['delta_pose'][0,2]
    robot_delta_meters = np.array([xRobot, yRobot, theta])
    robot_state_meters = np.add(robot_state_meters, robot_delta_meters) #important to keep
    return robot_state_meters #return updated robot state in meters

def MappingSmart (index, robot_state_meters):
    robot_state_cells = robotState2Cells(robot_state_meters)
    lidarState = lidar2cartesian(index)
    #then convert to world frame
    h2l = lidar2head(lidarState)
    b2h = head2body(index,h2l)
    w2b = body2world(index, b2h,robot_state_meters)
    '''
    objects = np.zeros((1,2))  #set()
    free_spaces = np.zeros((1,2)) #set()
    len_w2b = w2b.shape[1]
    #obstacle = np.zeros((1,2))
    '''
    cells = meters2cellsVector(w2b, x_im, y_im)
    
    x = np.int32(cells[:,0])
    y = np.int32(cells[:,1])
    cells = np.array([x,y])
    
    #Create Psuedo-Map
    MapSize = len(MAP['map'])
    tempMap = np.zeros((MapSize,MapSize), dtype=np.float64)
    logProb = math.log(10*lidarTrust)
    
    cellsT = np.transpose(cells)
    szCells = cellsT.shape[0]
    cellsC = np.zeros((szCells+1, 2), dtype=np.int32)
    cellsC[0:szCells, 0:2] = cellsT
    robotX = np.int32(robot_state_cells[0])
    robotY = np.int32(robot_state_cells[1])

    cellsC[szCells, 0:2] = [robotX, robotY]
    trash = np.zeros((1,2), dtype=np.int32)
    contours = list([trash])
    contours[0] = cellsC

    tempMap = cv2.drawContours(tempMap, contours, -1, -1, -1)
    tempMap = cv2.drawContours(tempMap, contours, -1, 1, 1)
    #tempMap[cells[:,0],cells[:,1]]= logProb
    MAP['map'] = np.add(MAP['map'], tempMap)
    #threshold = 90
    #indNeg = MAP['map'][:,:] < -threshold
    #MAP['map'][indNeg] = -threshold
    #indPos = MAP['map'][:,:] > threshold
    #MAP['map'][indPos] = threshold
    #indDecay = MAP['map'][:,:] <= 0
    #MAP['map'][indDecay] = MAP['map'][indDecay]*.3
    #indDecay = MAP['map'][:,:] > 0
    #MAP['map'][indDecay] = MAP['map'][indDecay]*.3
    return

def updateParticles (particles, i, numParticles, deltaPoseSave): #update them with new movement and add noise
    #update particles (particles, i, numParticles)
    xRobot, yRobot, theta = 0,0,0
    '''
    for k in range (i,i+skipScan ):
        xRobot = xRobot + l0[k]['delta_pose'][0,0]
        yRobot = yRobot + l0[k]['delta_pose'][0,1]
        theta = theta + l0[k]['delta_pose'][0,2]
    '''
    xRobot = l0[i]['delta_pose'][0,0]
    yRobot = l0[i]['delta_pose'][0,1]
    theta = l0[i]['delta_pose'][0,2]
    robot_delta_meters = np.array([xRobot, yRobot, theta, 0])
    particles[:,:] = np.add(particles[:,:], robot_delta_meters) #important to keep
    #deltaPoseSave = np.append(deltaPoseSave, robot_delta_meters)
    
    #particles[0,0]=particles[0,0]+xRobot
    #particles[0,1]=particles[0,1]+yRobot
    #particles[0,2]=particles[0,2]+theta
    #add noise
    #sigma x = 0.02, y = 0.017, theta = 0.03 #.0005
    noise = np.zeros((numParticles, 4))
    #mu, sigmax, sigmay, sigmatheta = 0, 0.002, 0.0017, 0.003 # mean and standard deviation
    #maybe even less noise
    #mu, sigmax, sigmay, sigmatheta = 0, .0005, .0005, .0005# mean and standard deviation
    if (xRobot != 0 or yRobot != 0):
        #mu, sigmax, sigmay, sigmatheta = 0, 0.002, 0.002, 0.005
        #mu, sigmax, sigmay, sigmatheta = 0, 0.005, 0.005, 0.005 # worked best so far 
        # not much different. mu, sigmax, sigmay, sigmatheta = 0, 0.01, 0.01, 0.05# mean and standard deviation
        #mu, sigmax, sigmay, sigmatheta = 0, .05*abs(xRobot) +.005, .05*abs(yRobot) +.005, .05*abs(theta) +.005
        # failed mu, sigmax, sigmay, sigmatheta = 0, 0.1, 0.1, 0.3# mean and standard deviation
        #best for lidar4 so far mu, sigmax, sigmay, sigmatheta = 0, 0.07, 0.07, 0.09
        #used for the 500 particles full run mu, sigmax, sigmay, sigmatheta = 0, 0.01, 0.01, 0.03
        mu, sigmax, sigmay, sigmatheta = 0, 0.01, 0.01, 0.03
        #mu, sigmax, sigmay, sigmatheta = 0, 0.07, 0.07, 0.09
        noise[:,0] = np.random.normal(mu, sigmax, numParticles)
        noise[:,1] = np.random.normal(mu, sigmay, numParticles)
        noise[:,2] = np.random.normal(mu, sigmatheta, numParticles)
        particles[:,:] = np.add(particles, noise) #important to keep
    return particles


if __name__ == '__main__':
    #######################################################
    ######################  Lidar 0
    #######################################################
    np.random.seed(0)
    j0 = load_data.get_joint("joint/train_joint0")
    l0 = load_data.get_lidar("lidar/train_lidar0")
    dataIn = io.loadmat("lidar/train_lidar0.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -60  #meters
    MAP['ymin']  = -60
    MAP['xmax']  =  60
    MAP['ymax']  =  60 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']= int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    #need coords of a cell? the cell equals the x and y indicies, and the value at these indices is the meters
    #need the cell number for your meter coords? call find nearest to get the index
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    
    MapSize = len(MAP['map'])
    MAP['map'] = np.zeros((MapSize,MapSize))
    
    deltaPoseSave = np.zeros((1,4))
    
    robot_state_meters = np.array([0,0,0])
    robot_state_cells = robotState2Cells(robot_state_meters)
    robot_state_storep = np.zeros((len(l0),3))
    
    lidarTrust = 0.9
    numParticles = 100#40
    particles = np.zeros((numParticles, 4)) #each particle is its own row
    #goes [x, y, yaw, weight], IN METERS
    particles[:,3] = 1/numParticles #initialize particles weight
    
    maxWeight = np.argmax(particles, axis =0)
    wIndex = maxWeight[3]
    robot_state_meters = [particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]]
    
    #initialize variables
    MappingSmart(0, robot_state_meters) 
    particleCorrelationsGet = np.zeros((numParticles, len(l0)))
    particleCorrelations = np.zeros((numParticles, len(l0)))
    countResample = 0
    countSinceLast = 0
    skipScan = 1
    picSave = 500
    a = 1
    #for i in range(a,(a+998)): #len(l0)
    for i in range(1,len(l0)): #len(l0)
        
        #=================save map perdiodically as it updates ======
        if (i%picSave) == 0 :
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            ColorMap = np.zeros((MapSize,MapSize))
            ColorMap[indPos] = 1
            ColorMap[indNeg] = -1
            robot_state_int = robot_state_storep.astype(int)
            positionMap = np.zeros((MapSize,MapSize))
            ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2
            plt.imshow(ColorMap)
            plt.savefig('ECE276A_PR2_Lidar0_PIC_'+str(i)+'.png', dpi=1000)
        #================= update the particles =================
        particles = updateParticles (particles, i, numParticles, deltaPoseSave)

        countSinceLast = countSinceLast +1
        #if(l0[i]['delta_pose'][0,2] != 0  or countSinceLast >50):
        if (i%skipScan) == 0 :
            print(i)
            countSinceLast = 0
            #================= update the correlations =================
            #getCorrelation(particles, i, numParticles, MAP, x_im, y_im) #return array of correlations
            particleCorrelations = np.zeros((numParticles))
            #particleCorrelationsGet = np.zeros((numParticles))
            #create binary max to pass 
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            binaryMap = np.zeros((MapSize,MapSize))
            binaryMap[indPos] = 1
            binaryMap[indNeg] = 0
            
            #get the lidar state
            lidarState = lidar2cartesian(i)
            h2l = lidar2head(lidarState)
            b2h = head2body(i,h2l)
            
            for j in range(numParticles):
                currentParticle = particles[j,:]
                w2b = body2world(i, b2h,currentParticle[0:3])
                
                cells = meters2cellsVector(w2b, x_im, y_im)
                
                x = np.int32(cells[:,0])
                y = np.int32(cells[:,1])
                cells = np.array([x,y])

                particleCorrelationsGet[j,i] = binaryMap[cells[1,:],cells[0,:]].sum()
                #return particleCorrelations
                #get softmax, update correlations
            particleCorrelations = softmax(particleCorrelationsGet[:,i])
            particles[:,3] = particleCorrelations
                
                
            #================= choose max weight particle =================
            maxWeight = np.argmax(particles, axis =0)
            wIndex = maxWeight[3]
            robot_state_meters = np.array([particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]])
            #robot_state_meters = robotUpdateState(i, robot_state_meters)
            robot_state_cells = robotState2Cells(robot_state_meters)
            robot_state_storep[i,:] = robot_state_cells
            
            #================= update the map =================
            MappingSmart(i, robot_state_meters)
            #decay = .9
            #MAP['map'] = numpy.multiply(decay, MAP['map'])
            
            #================= particle resampling =================
            nEff = 1/sum(np.square(particles[:,3])) #find the sum of the squares of the weight
            #nEff is smaller for more particles
            nThresh = .1*numParticles
            if nEff <= nThresh:
                countResample = countResample+1
                j, c = 1, particles[0,3]
                particlesNew = np.zeros((numParticles,4))
                for k in range(numParticles):
                    b = np.random.uniform(0, 1/numParticles) + (k)/numParticles
                    while b > c:
                        j = j+1
                        c = c + particles[j-1,3]
                    particlesNew[k,0] = particles[j-1,0]
                    particlesNew[k,1] = particles[j-1,1]
                    particlesNew[k,2] = particles[j-1,2]
                    particlesNew[k,3] = 1/numParticles

                particles = particlesNew
                
            
    
    indPos = MAP['map'][:,:] > 0
    indNeg = MAP['map'][:,:] < 0
    MapSize = len(MAP['map'])
    ColorMap = np.zeros((MapSize,MapSize))
    ColorMap[indPos] = 1
    ColorMap[indNeg] = -1
    plt.imshow(ColorMap)
    robot_state_int = robot_state_storep.astype(int)
    positionMap = np.zeros((MapSize,MapSize))
    ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2

    plt.imshow(ColorMap)
    plt.colorbar()
    plt.savefig('Lidar0,1.png', dpi=500)
    plt.show()
    
    plt.imshow(MAP['map'])
    plt.colorbar()
    plt.savefig('Lidar0,2.png', dpi=500)
    plt.show()
    np.save('Lidar0', MAP['map'])
    
    #######################################################
    ######################  Lidar 1
    #######################################################
    np.random.seed(0)
    j0 = load_data.get_joint("joint/train_joint1")
    l0 = load_data.get_lidar("lidar/train_lidar1")
    dataIn = io.loadmat("lidar/train_lidar1.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -60  #meters
    MAP['ymin']  = -60
    MAP['xmax']  =  60
    MAP['ymax']  =  60 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']= int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    #need coords of a cell? the cell equals the x and y indicies, and the value at these indices is the meters
    #need the cell number for your meter coords? call find nearest to get the index
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    
    MapSize = len(MAP['map'])
    MAP['map'] = np.zeros((MapSize,MapSize))
    
    deltaPoseSave = np.zeros((1,4))
    
    robot_state_meters = np.array([0,0,0])
    robot_state_cells = robotState2Cells(robot_state_meters)
    robot_state_storep = np.zeros((len(l0),3))
    
    lidarTrust = 0.9
    numParticles = 100#40
    particles = np.zeros((numParticles, 4)) #each particle is its own row
    #goes [x, y, yaw, weight], IN METERS
    particles[:,3] = 1/numParticles #initialize particles weight
    
    maxWeight = np.argmax(particles, axis =0)
    wIndex = maxWeight[3]
    robot_state_meters = [particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]]
    
    #initialize variables
    MappingSmart(0, robot_state_meters) 
    particleCorrelationsGet = np.zeros((numParticles, len(l0)))
    particleCorrelations = np.zeros((numParticles, len(l0)))
    countResample = 0
    countSinceLast = 0
    skipScan = 1
    
    a = 1
    #for i in range(a,(a+998)): #len(l0)
    for i in range(1,len(l0)): #len(l0)
        
        #=================save map perdiodically as it updates ======
        if (i%picSave) == 0 :
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            ColorMap = np.zeros((MapSize,MapSize))
            ColorMap[indPos] = 1
            ColorMap[indNeg] = -1
            robot_state_int = robot_state_storep.astype(int)
            positionMap = np.zeros((MapSize,MapSize))
            ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2
            plt.imshow(ColorMap)
            plt.savefig('ECE276A_PR2_Lidar1_PIC_'+str(i)+'.png', dpi=1000)
        
        #================= update the particles =================
        particles = updateParticles (particles, i, numParticles, deltaPoseSave)

        countSinceLast = countSinceLast +1
        #if(l0[i]['delta_pose'][0,2] != 0  or countSinceLast >50):
        if (i%skipScan) == 0 :
            print(i)
            countSinceLast = 0
            #================= update the correlations =================
            #getCorrelation(particles, i, numParticles, MAP, x_im, y_im) #return array of correlations
            particleCorrelations = np.zeros((numParticles))
            #particleCorrelationsGet = np.zeros((numParticles))
            #create binary max to pass 
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            binaryMap = np.zeros((MapSize,MapSize))
            binaryMap[indPos] = 1
            binaryMap[indNeg] = 0
            
            #get the lidar state
            lidarState = lidar2cartesian(i)
            h2l = lidar2head(lidarState)
            b2h = head2body(i,h2l)
            
            for j in range(numParticles):
                currentParticle = particles[j,:]
                w2b = body2world(i, b2h,currentParticle[0:3])
                
                cells = meters2cellsVector(w2b, x_im, y_im)
                
                x = np.int32(cells[:,0])
                y = np.int32(cells[:,1])
                cells = np.array([x,y])

                particleCorrelationsGet[j,i] = binaryMap[cells[1,:],cells[0,:]].sum()
                #return particleCorrelations
                #get softmax, update correlations
            particleCorrelations = softmax(particleCorrelationsGet[:,i])
            particles[:,3] = particleCorrelations
                
                
            #================= choose max weight particle =================
            maxWeight = np.argmax(particles, axis =0)
            wIndex = maxWeight[3]
            robot_state_meters = np.array([particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]])
            #robot_state_meters = robotUpdateState(i, robot_state_meters)
            robot_state_cells = robotState2Cells(robot_state_meters)
            robot_state_storep[i,:] = robot_state_cells
            
            #================= update the map =================
            MappingSmart(i, robot_state_meters)
            #decay = .9
            #MAP['map'] = numpy.multiply(decay, MAP['map'])
            
            #================= particle resampling =================
            nEff = 1/sum(np.square(particles[:,3])) #find the sum of the squares of the weight
            #nEff is smaller for more particles
            nThresh = .1*numParticles
            if nEff <= nThresh:
                countResample = countResample+1
                j, c = 1, particles[0,3]
                particlesNew = np.zeros((numParticles,4))
                for k in range(numParticles):
                    b = np.random.uniform(0, 1/numParticles) + (k)/numParticles
                    while b > c:
                        j = j+1
                        c = c + particles[j-1,3]
                    particlesNew[k,0] = particles[j-1,0]
                    particlesNew[k,1] = particles[j-1,1]
                    particlesNew[k,2] = particles[j-1,2]
                    particlesNew[k,3] = 1/numParticles

                particles = particlesNew
                
            
    
    indPos = MAP['map'][:,:] > 0
    indNeg = MAP['map'][:,:] < 0
    MapSize = len(MAP['map'])
    ColorMap = np.zeros((MapSize,MapSize))
    ColorMap[indPos] = 1
    ColorMap[indNeg] = -1
    plt.imshow(ColorMap)
    robot_state_int = robot_state_storep.astype(int)
    positionMap = np.zeros((MapSize,MapSize))
    ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2

    plt.imshow(ColorMap)
    plt.colorbar()
    plt.savefig('Lidar1,1.png', dpi=500)
    plt.show()
    
    plt.imshow(MAP['map'])
    plt.colorbar()
    plt.savefig('Lidar1,2.png', dpi=500)
    plt.show()
    np.save('Lidar1', MAP['map'])
    
    
    #######################################################
    ######################  Lidar 2
    #######################################################
    np.random.seed(0)
    j0 = load_data.get_joint("joint/train_joint2")
    l0 = load_data.get_lidar("lidar/train_lidar2")
    dataIn = io.loadmat("lidar/train_lidar2.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -60  #meters
    MAP['ymin']  = -60
    MAP['xmax']  =  60
    MAP['ymax']  =  60 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']= int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    #need coords of a cell? the cell equals the x and y indicies, and the value at these indices is the meters
    #need the cell number for your meter coords? call find nearest to get the index
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    
    MapSize = len(MAP['map'])
    MAP['map'] = np.zeros((MapSize,MapSize))
    
    deltaPoseSave = np.zeros((1,4))
    
    robot_state_meters = np.array([0,0,0])
    robot_state_cells = robotState2Cells(robot_state_meters)
    robot_state_storep = np.zeros((len(l0),3))
    
    lidarTrust = 0.9
    numParticles = 100#40
    particles = np.zeros((numParticles, 4)) #each particle is its own row
    #goes [x, y, yaw, weight], IN METERS
    particles[:,3] = 1/numParticles #initialize particles weight
    
    maxWeight = np.argmax(particles, axis =0)
    wIndex = maxWeight[3]
    robot_state_meters = [particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]]
    
    #initialize variables
    MappingSmart(0, robot_state_meters) 
    particleCorrelationsGet = np.zeros((numParticles, len(l0)))
    particleCorrelations = np.zeros((numParticles, len(l0)))
    countResample = 0
    countSinceLast = 0
    skipScan = 1
    
    a = 1
    #for i in range(a,(a+998)): #len(l0)
    for i in range(1,len(l0)): #len(l0)
        
        #=================save map perdiodically as it updates ======
        if (i%picSave) == 0 :
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            ColorMap = np.zeros((MapSize,MapSize))
            ColorMap[indPos] = 1
            ColorMap[indNeg] = -1
            robot_state_int = robot_state_storep.astype(int)
            positionMap = np.zeros((MapSize,MapSize))
            ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2
            plt.imshow(ColorMap)
            plt.savefig('ECE276A_PR2_Lidar2_PIC_'+str(i)+'.png', dpi=1000)
        
        #================= update the particles =================
        particles = updateParticles (particles, i, numParticles, deltaPoseSave)

        countSinceLast = countSinceLast +1
        #if(l0[i]['delta_pose'][0,2] != 0  or countSinceLast >50):
        if (i%skipScan) == 0 :
            print(i)
            countSinceLast = 0
            #================= update the correlations =================
            #getCorrelation(particles, i, numParticles, MAP, x_im, y_im) #return array of correlations
            particleCorrelations = np.zeros((numParticles))
            #particleCorrelationsGet = np.zeros((numParticles))
            #create binary max to pass 
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            binaryMap = np.zeros((MapSize,MapSize))
            binaryMap[indPos] = 1
            binaryMap[indNeg] = 0
            
            #get the lidar state
            lidarState = lidar2cartesian(i)
            h2l = lidar2head(lidarState)
            b2h = head2body(i,h2l)
            
            for j in range(numParticles):
                currentParticle = particles[j,:]
                w2b = body2world(i, b2h,currentParticle[0:3])
                
                cells = meters2cellsVector(w2b, x_im, y_im)
                
                x = np.int32(cells[:,0])
                y = np.int32(cells[:,1])
                cells = np.array([x,y])

                particleCorrelationsGet[j,i] = binaryMap[cells[1,:],cells[0,:]].sum()
                #return particleCorrelations
                #get softmax, update correlations
            particleCorrelations = softmax(particleCorrelationsGet[:,i])
            particles[:,3] = particleCorrelations
                
                
            #================= choose max weight particle =================
            maxWeight = np.argmax(particles, axis =0)
            wIndex = maxWeight[3]
            robot_state_meters = np.array([particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]])
            #robot_state_meters = robotUpdateState(i, robot_state_meters)
            robot_state_cells = robotState2Cells(robot_state_meters)
            robot_state_storep[i,:] = robot_state_cells
            
            #================= update the map =================
            MappingSmart(i, robot_state_meters)
            #decay = .9
            #MAP['map'] = numpy.multiply(decay, MAP['map'])
            
            #================= particle resampling =================
            nEff = 1/sum(np.square(particles[:,3])) #find the sum of the squares of the weight
            #nEff is smaller for more particles
            nThresh = .1*numParticles
            if nEff <= nThresh:
                countResample = countResample+1
                j, c = 1, particles[0,3]
                particlesNew = np.zeros((numParticles,4))
                for k in range(numParticles):
                    b = np.random.uniform(0, 1/numParticles) + (k)/numParticles
                    while b > c:
                        j = j+1
                        c = c + particles[j-1,3]
                    particlesNew[k,0] = particles[j-1,0]
                    particlesNew[k,1] = particles[j-1,1]
                    particlesNew[k,2] = particles[j-1,2]
                    particlesNew[k,3] = 1/numParticles

                particles = particlesNew
                
            
    
    indPos = MAP['map'][:,:] > 0
    indNeg = MAP['map'][:,:] < 0
    MapSize = len(MAP['map'])
    ColorMap = np.zeros((MapSize,MapSize))
    ColorMap[indPos] = 1
    ColorMap[indNeg] = -1
    plt.imshow(ColorMap)
    robot_state_int = robot_state_storep.astype(int)
    positionMap = np.zeros((MapSize,MapSize))
    ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2

    plt.imshow(ColorMap)
    plt.colorbar()
    plt.savefig('Lidar2,1.png', dpi=500)
    plt.show()
    
    plt.imshow(MAP['map'])
    plt.colorbar()
    plt.savefig('Lidar2,2.png', dpi=500)
    plt.show()
    np.save('Lidar2', MAP['map'])
    
    
    
    #######################################################
    ######################  Lidar 3
    #######################################################
    np.random.seed(0)
    j0 = load_data.get_joint("joint/train_joint3")
    l0 = load_data.get_lidar("lidar/train_lidar3")
    dataIn = io.loadmat("lidar/train_lidar3.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -60  #meters
    MAP['ymin']  = -60
    MAP['xmax']  =  60
    MAP['ymax']  =  60 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']= int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    #need coords of a cell? the cell equals the x and y indicies, and the value at these indices is the meters
    #need the cell number for your meter coords? call find nearest to get the index
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    
    MapSize = len(MAP['map'])
    MAP['map'] = np.zeros((MapSize,MapSize))
    
    deltaPoseSave = np.zeros((1,4))
    
    robot_state_meters = np.array([0,0,0])
    robot_state_cells = robotState2Cells(robot_state_meters)
    robot_state_storep = np.zeros((len(l0),3))
    
    lidarTrust = 0.9
    numParticles = 100#40
    particles = np.zeros((numParticles, 4)) #each particle is its own row
    #goes [x, y, yaw, weight], IN METERS
    particles[:,3] = 1/numParticles #initialize particles weight
    
    maxWeight = np.argmax(particles, axis =0)
    wIndex = maxWeight[3]
    robot_state_meters = [particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]]
    
    #initialize variables
    MappingSmart(0, robot_state_meters) 
    particleCorrelationsGet = np.zeros((numParticles, len(l0)))
    particleCorrelations = np.zeros((numParticles, len(l0)))
    countResample = 0
    countSinceLast = 0
    skipScan = 1
    
    a = 1
    #for i in range(a,(a+998)): #len(l0)
    for i in range(1,len(l0)): #len(l0)
        
        #=================save map perdiodically as it updates ======
        if (i%picSave) == 0 :
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            ColorMap = np.zeros((MapSize,MapSize))
            ColorMap[indPos] = 1
            ColorMap[indNeg] = -1
            robot_state_int = robot_state_storep.astype(int)
            positionMap = np.zeros((MapSize,MapSize))
            ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2
            plt.imshow(ColorMap)
            plt.savefig('ECE276A_PR2_Lidar3_PIC_'+str(i)+'.png', dpi=1000)
        
        #================= update the particles =================
        particles = updateParticles (particles, i, numParticles, deltaPoseSave)

        countSinceLast = countSinceLast +1
        #if(l0[i]['delta_pose'][0,2] != 0  or countSinceLast >50):
        if (i%skipScan) == 0 :
            print(i)
            countSinceLast = 0
            #================= update the correlations =================
            #getCorrelation(particles, i, numParticles, MAP, x_im, y_im) #return array of correlations
            particleCorrelations = np.zeros((numParticles))
            #particleCorrelationsGet = np.zeros((numParticles))
            #create binary max to pass 
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            binaryMap = np.zeros((MapSize,MapSize))
            binaryMap[indPos] = 1
            binaryMap[indNeg] = 0
            
            #get the lidar state
            lidarState = lidar2cartesian(i)
            h2l = lidar2head(lidarState)
            b2h = head2body(i,h2l)
            
            for j in range(numParticles):
                currentParticle = particles[j,:]
                w2b = body2world(i, b2h,currentParticle[0:3])
                
                cells = meters2cellsVector(w2b, x_im, y_im)
                
                x = np.int32(cells[:,0])
                y = np.int32(cells[:,1])
                cells = np.array([x,y])

                particleCorrelationsGet[j,i] = binaryMap[cells[1,:],cells[0,:]].sum()
                #return particleCorrelations
                #get softmax, update correlations
            particleCorrelations = softmax(particleCorrelationsGet[:,i])
            particles[:,3] = particleCorrelations
                
                
            #================= choose max weight particle =================
            maxWeight = np.argmax(particles, axis =0)
            wIndex = maxWeight[3]
            robot_state_meters = np.array([particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]])
            #robot_state_meters = robotUpdateState(i, robot_state_meters)
            robot_state_cells = robotState2Cells(robot_state_meters)
            robot_state_storep[i,:] = robot_state_cells
            
            #================= update the map =================
            MappingSmart(i, robot_state_meters)
            #decay = .9
            #MAP['map'] = numpy.multiply(decay, MAP['map'])
            
            #================= particle resampling =================
            nEff = 1/sum(np.square(particles[:,3])) #find the sum of the squares of the weight
            #nEff is smaller for more particles
            nThresh = .1*numParticles
            if nEff <= nThresh:
                countResample = countResample+1
                j, c = 1, particles[0,3]
                particlesNew = np.zeros((numParticles,4))
                for k in range(numParticles):
                    b = np.random.uniform(0, 1/numParticles) + (k)/numParticles
                    while b > c:
                        j = j+1
                        c = c + particles[j-1,3]
                    particlesNew[k,0] = particles[j-1,0]
                    particlesNew[k,1] = particles[j-1,1]
                    particlesNew[k,2] = particles[j-1,2]
                    particlesNew[k,3] = 1/numParticles

                particles = particlesNew
                
            
    
    indPos = MAP['map'][:,:] > 0
    indNeg = MAP['map'][:,:] < 0
    MapSize = len(MAP['map'])
    ColorMap = np.zeros((MapSize,MapSize))
    ColorMap[indPos] = 1
    ColorMap[indNeg] = -1
    plt.imshow(ColorMap)
    robot_state_int = robot_state_storep.astype(int)
    positionMap = np.zeros((MapSize,MapSize))
    ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2

    plt.imshow(ColorMap)
    plt.colorbar()
    plt.savefig('Lidar3,1.png', dpi=500)
    plt.show()
    
    plt.imshow(MAP['map'])
    plt.colorbar()
    plt.savefig('Lidar3,2.png', dpi=500)
    plt.show()
    np.save('Lidar3', MAP['map'])
    
    
    
    
    #######################################################
    ######################  Lidar 4
    #######################################################
    np.random.seed(0)
    j0 = load_data.get_joint("joint/train_joint4")
    l0 = load_data.get_lidar("lidar/train_lidar4")
    dataIn = io.loadmat("lidar/train_lidar4.mat")
    angles = np.array([np.arange(-135,135.25,0.25)*np.pi/180.]).T
    ranges = np.double(dataIn['lidar'][0][110]['scan'][0][0]).T
    # take valid indices
    indValid = np.logical_and((ranges < 30),(ranges> 0.1))
    ranges = ranges[indValid]
    angles = angles[indValid]
    # init MAP
    MAP = {}
    MAP['res']   = 0.05 #meters
    MAP['xmin']  = -60  #meters
    MAP['ymin']  = -60
    MAP['xmax']  =  60
    MAP['ymax']  =  60 
    MAP['sizex']  = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1)) #cells
    MAP['sizey']= int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))
    MAP['map'] = np.zeros((MAP['sizex'],MAP['sizey']),dtype=np.float64) #DATA TYPE: char or int8
    # xy position in the sensor frame
    xs0 = np.array([ranges*np.cos(angles)])
    ys0 = np.array([ranges*np.sin(angles)])
    # convert from meters to cells
    xis = np.ceil((xs0 - MAP['xmin']) / MAP['res'] ).astype(np.int16)-1
    yis = np.ceil((ys0 - MAP['ymin']) / MAP['res'] ).astype(np.int16)-1
    # build an arbitrary map 
    indGood = np.logical_and(np.logical_and(np.logical_and((xis > 1), (yis > 1)), (xis < MAP['sizex'])), (yis < MAP['sizey']))
    MAP['map'][xis[0][indGood[0]],yis[0][indGood[0]]]=1
    x_im = np.arange(MAP['xmin'],MAP['xmax']+MAP['res'],MAP['res']) #x-positions of each pixel of the map
    y_im = np.arange(MAP['ymin'],MAP['ymax']+MAP['res'],MAP['res']) #y-positions of each pixel of the map
    #need coords of a cell? the cell equals the x and y indicies, and the value at these indices is the meters
    #need the cell number for your meter coords? call find nearest to get the index
    x_range = np.arange(-0.2,0.2+0.05,0.05)
    y_range = np.arange(-0.2,0.2+0.05,0.05)
    
    
    MapSize = len(MAP['map'])
    MAP['map'] = np.zeros((MapSize,MapSize))
    
    deltaPoseSave = np.zeros((1,4))
    
    robot_state_meters = np.array([0,0,0])
    robot_state_cells = robotState2Cells(robot_state_meters)
    robot_state_storep = np.zeros((len(l0),3))
    
    lidarTrust = 0.9
    numParticles = 100#40
    particles = np.zeros((numParticles, 4)) #each particle is its own row
    #goes [x, y, yaw, weight], IN METERS
    particles[:,3] = 1/numParticles #initialize particles weight
    
    maxWeight = np.argmax(particles, axis =0)
    wIndex = maxWeight[3]
    robot_state_meters = [particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]]
    
    #initialize variables
    MappingSmart(0, robot_state_meters) 
    particleCorrelationsGet = np.zeros((numParticles, len(l0)))
    particleCorrelations = np.zeros((numParticles, len(l0)))
    countResample = 0
    countSinceLast = 0
    skipScan = 1
    
    a = 1
    #for i in range(a,(a+998)): #len(l0)
    for i in range(1,len(l0)): #len(l0)
        
        #=================save map perdiodically as it updates ======
        if (i%picSave) == 0 :
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            ColorMap = np.zeros((MapSize,MapSize))
            ColorMap[indPos] = 1
            ColorMap[indNeg] = -1
            robot_state_int = robot_state_storep.astype(int)
            positionMap = np.zeros((MapSize,MapSize))
            ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2
            plt.imshow(ColorMap)
            plt.savefig('ECE276A_PR2_Lidar4_PIC_'+str(i)+'.png', dpi=1000)
        
        #================= update the particles =================
        particles = updateParticles (particles, i, numParticles, deltaPoseSave)

        countSinceLast = countSinceLast +1
        #if(l0[i]['delta_pose'][0,2] != 0  or countSinceLast >50):
        if (i%skipScan) == 0 :
            print(i)
            countSinceLast = 0
            #================= update the correlations =================
            #getCorrelation(particles, i, numParticles, MAP, x_im, y_im) #return array of correlations
            particleCorrelations = np.zeros((numParticles))
            #particleCorrelationsGet = np.zeros((numParticles))
            #create binary max to pass 
            indPos = MAP['map'][:,:] > 0
            indNeg = MAP['map'][:,:] < 0
            MapSize = len(MAP['map'])
            binaryMap = np.zeros((MapSize,MapSize))
            binaryMap[indPos] = 1
            binaryMap[indNeg] = 0
            
            #get the lidar state
            lidarState = lidar2cartesian(i)
            h2l = lidar2head(lidarState)
            b2h = head2body(i,h2l)
            
            for j in range(numParticles):
                currentParticle = particles[j,:]
                w2b = body2world(i, b2h,currentParticle[0:3])
                
                cells = meters2cellsVector(w2b, x_im, y_im)
                
                x = np.int32(cells[:,0])
                y = np.int32(cells[:,1])
                cells = np.array([x,y])

                particleCorrelationsGet[j,i] = binaryMap[cells[1,:],cells[0,:]].sum()
                #return particleCorrelations
                #get softmax, update correlations
            particleCorrelations = softmax(particleCorrelationsGet[:,i])
            particles[:,3] = particleCorrelations
                
                
            #================= choose max weight particle =================
            maxWeight = np.argmax(particles, axis =0)
            wIndex = maxWeight[3]
            robot_state_meters = np.array([particles[wIndex,0], particles[wIndex,1], particles[wIndex,2]])
            #robot_state_meters = robotUpdateState(i, robot_state_meters)
            robot_state_cells = robotState2Cells(robot_state_meters)
            robot_state_storep[i,:] = robot_state_cells
            
            #================= update the map =================
            MappingSmart(i, robot_state_meters)
            #decay = .9
            #MAP['map'] = numpy.multiply(decay, MAP['map'])
            
            #================= particle resampling =================
            nEff = 1/sum(np.square(particles[:,3])) #find the sum of the squares of the weight
            #nEff is smaller for more particles
            nThresh = .1*numParticles
            if nEff <= nThresh:
                countResample = countResample+1
                j, c = 1, particles[0,3]
                particlesNew = np.zeros((numParticles,4))
                for k in range(numParticles):
                    b = np.random.uniform(0, 1/numParticles) + (k)/numParticles
                    while b > c:
                        j = j+1
                        c = c + particles[j-1,3]
                    particlesNew[k,0] = particles[j-1,0]
                    particlesNew[k,1] = particles[j-1,1]
                    particlesNew[k,2] = particles[j-1,2]
                    particlesNew[k,3] = 1/numParticles

                particles = particlesNew
                
            
    
    indPos = MAP['map'][:,:] > 0
    indNeg = MAP['map'][:,:] < 0
    MapSize = len(MAP['map'])
    ColorMap = np.zeros((MapSize,MapSize))
    ColorMap[indPos] = 1
    ColorMap[indNeg] = -1
    plt.imshow(ColorMap)
    robot_state_int = robot_state_storep.astype(int)
    positionMap = np.zeros((MapSize,MapSize))
    ColorMap[robot_state_int[:,1], robot_state_int[:,0]] = 2

    plt.imshow(ColorMap)
    plt.colorbar()
    plt.savefig('Lidar4,1.png', dpi=500)
    plt.show()
    
    plt.imshow(MAP['map'])
    plt.colorbar()
    plt.savefig('Lidar4,2.png', dpi=500)
    plt.show()
    np.save('Lidar4', MAP['map'])
    