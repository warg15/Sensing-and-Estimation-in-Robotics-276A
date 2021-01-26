# Sensing-and-Estimation-in-Robotics-276A
Git Repo of Project work from UCSD ECE276A.

Project 1: Used Logistic Regression machine learning to train a stop sign detector to detect color, which was used in conjuction with OpenCV libraries to 
identify object shape and locate stop signs, if any existed, in a given image. 
 
Project 2: Took LiDAR, Actuator, IMU, and Camera data from THOR walking robot to perform Simultaneous Localization and Mapping (SLAM) as the robot moved 
through an unknown enviornment. Implementation included a particle filter to stochastically predict a large number of potential state deviations from the 
state determined by Dead Reckoning via the THOR robot's movement actuator inputs. At any given time the best state was chosen with respect to current 
sensor data. 
 
Project 3: Performed Visual-Inertial SLAM with an Extended Kalman Filter. The EKF used IMU-based Localization as state prediction and landmark mapping from robot's
onboard stereo camera as EKF Update. Combining the two measurements allowed a more accurate prediction of the robot's state as well as a mapping of the
robot's enviornment.
