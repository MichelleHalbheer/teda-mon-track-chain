# Define constants

import numpy as np
import os

### ------------------------------ ###
### -------- DIRECTORIES --------- ###
### ------------------------------ ###

dir_src = os.path.dirname(__file__)
dir_root = os.path.dirname(dir_src)
dir_data = os.path.join(dir_root, 'data')
dir_plots = os.path.join(dir_root, 'plots')


### ------------------------------ ###
### ----------- VALUES ----------- ###
### ------------------------------ ###

rho_gon = np.pi / 200 # Conversion factor from gon to radians
rho_deg = np.pi / 180 # Conversion factor from degree to radians

sigma_dist = 50 # [mm]
sigma_tilt = 0.072 # [deg]
sigma_tilt_rad = sigma_tilt * rho_deg

num_sensors = 100 # Number of sensors in track chain
num_points = num_sensors + 1 # Number of measurement points
dist_tot = 120 # [m] nominal distance between any two sensors

num_reps = int(1e6) # Number of repetitions for Monte Carlo simulation

h0 = 0 # Height level of first point