# Program to calculate a preadjustment for a traverse to simulate the precision of the TrackChain

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from const import *

def main():
    monte_carlo()

def monte_carlo():
    '''
    Analysis of precision using a Monte Carlo simulation
    '''

    r_dist = np.random.normal(dist * 1000, sigma_dist, (num_sensors, num_reps))
    r_ang = np.random.normal(0, sigma_tilt_rad, (num_sensors, num_reps))

    h0 = np.zeros(num_points)

    dh = r_dist * np.sin(r_ang)

    h_free = np.insert(np.cumsum(dh, axis=0), 0 ,np.zeros(num_reps), axis=0)

    h_fixed = h_free - np.multiply(np.repeat(h_free[-1,:][np.newaxis, :], num_points, axis=0), np.arange(0, num_points)[:,np.newaxis] / (num_points - 1))

    fig = plt.figure('Standardabweichung im Verlauf des Zuges')
    plt.plot(np.arange(101), np.std(h_free, axis=1), label='Einseitig angeschlossen', marker='o',markevery=5)
    plt.plot(np.arange(101), np.std(h_fixed, axis=1), label='Beidseitig angeschlossen', marker='o',markevery=5)
    plt.xlabel('Anzahl Punkte')
    plt.ylabel('Standardabweichung [mm]')
    plt.legend()
    plt.grid()
    plt.show()
    

def empirical_std_tilt():
    pass

if __name__ == '__main__':
    main()