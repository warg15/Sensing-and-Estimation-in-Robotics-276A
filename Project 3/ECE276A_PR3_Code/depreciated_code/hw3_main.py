import numpy as np
from utils import *


if __name__ == '__main__':
	filename = "./data/0027.npz"
	t,features,linear_velocity,rotational_velocity,K,b,cam_T_imu = load_data(filename)

	# (a) IMU Localization via EKF Prediction

	# (b) Landmark Mapping via EKF Update

	# (c) Visual-Inertial SLAM

	# You can use the function below to visualize the robot pose over time
	#visualize_trajectory_2d(world_T_imu,show_ori=True)



"""
        v = linear_velocity[:, i]
        omega = rotational_velocity[:, i]
        weight = 100
            # covariance for movement noise
        W = weight * np.eye(6)
    
        tau = -(tau)
        
        #u = np.vstack((v, omega)) # control input
        u_hat = np.vstack((np.hstack((hat_map(omega), v.reshape(3, 1))), np.zeros((1, 4))))    
        u_curlyhat = np.block([[  hat_map(omega),     hat_map(v)], [np.zeros((3, 3)), hat_map(omega)]])
        
        Car['mean'] = expm(tau * u_hat) @ Car['mean']
        Car['covariance'] = expm(tau * u_curlyhat) @ Car['covariance'] @ np.transpose(expm(tau * u_curlyhat)) + W
        """"""