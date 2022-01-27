# Program to calculate a preadjustment for a traverse to simulate the precision of the TrackChain

import numpy as np
from matplotlib import pyplot as plt
from const import *

def main():
    
    ### Define starting values for height ###

    height_0 = np.zeros(num_sensors) ## Assumed to be zero for all points, as the tilt of all sensors is set to 0

    
    # Get pairwose height differences of the two points
    pairwise_deltas = (height_0 - np.insert(height_0[:-1], 0, 0, axis=0))[1:]
    distances = np.ones(num_sensors) * dist


if __name__ == '__main__':
    main()