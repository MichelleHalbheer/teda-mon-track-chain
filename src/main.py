# Program to calculate a preadjustment for a traverse to simulate the precision of the TrackChain

from tkinter.messagebox import NO
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import datetime

from const import *
import const

def main():

    # Calculate empirical standard deviation
    sensor_std = empirical_std_tilt(timedelta_days=3)
    
    # Define cases
    dist_std_list = [0, 50, 100, 200]
    num_sensor_list = [25, 50, 100]
    
    std_free_dict = {}
    std_fixed_dict = {}
    # Run all cases
    for num_sensors in num_sensor_list:
        for sigma_dist in dist_std_list:
            # For every case use predefined sigma for tilt and empirical sigma
            std_free, std_fixed = monte_carlo(sigma_dist=sigma_dist, num_sensors=num_sensors)
            std_free_emp, std_fixed_emp = monte_carlo(sigma_dist=sigma_dist, sigma_tilt=sensor_std, num_sensors=num_sensors)

            # Safe the values into the dictionary

            std_free_dict[(num_sensors, sigma_dist, 'default')] = std_free
            std_fixed_dict[(num_sensors, sigma_dist, 'default')] = std_fixed

            std_free_dict[(num_sensors, sigma_dist, 'empirical')] = std_free_emp
            std_fixed_dict[(num_sensors, sigma_dist, 'empirical')] = std_fixed_emp

            plt.figure()
            plt.suptitle('Standardabweichung im Verlauf der TrackChain')
            plt.title(f'Distanz: {dist_tot}, Anzahl Sensoren: {num_sensors}\nDistanz zw. Sensoren:{dist_tot / num_sensors}, $\sigma_d=${sigma_dist}')
            plt.plot(
                np.arange(num_sensors + 1),
                std_free,
                label='Einseitig angeschlossen',
                marker='o', 
                markevery=num_sensors//25
            )
            plt.plot(
                np.arange(num_sensors + 1),
                std_fixed,
                label='Beidseitig angeschlossen',
                marker='o',
                markevery=num_sensors//25
            )
            plt.xlabel('Anzahl Punkte')
            plt.ylabel('Standardabweichung [mm]')
            plt.legend()
            plt.savefig(os.path.join(dir_plots, 'plot_test.png'))
            plt.close()

def monte_carlo(sigma_dist=const.sigma_dist, sigma_tilt=const.sigma_tilt, num_sensors=const.num_sensors, nominal_dist=None):
    '''
    Analysis of precision using a Monte Carlo simulation

    Parameters
    ----------
    sigma_dist : Float, optional
        Standard deviation for the distances
    sigma_tilt : Float, optional
        Standard deviation for the tilt angles
    num_sensors : Integer, optional
        Number of sensors used
    nominal_dist : Float, optional
        Nominal distance between any two points
    '''
    
    if nominal_dist is None:
        nominal_dist = const.dist_tot / num_sensors

    num_points = num_sensors + 1

    # Create random observations for the distances and the angles
    r_dist = np.random.normal(nominal_dist * 1000, sigma_dist, (num_sensors, num_reps))
    r_ang = np.random.normal(0, sigma_tilt * rho_deg, (num_sensors, num_reps))

    # Height differences per point
    dh = r_dist * np.sin(r_ang)

    # Cumulative sum of height differences, last point unknown
    h_free = np.insert(np.cumsum(dh, axis=0), 0 ,np.zeros(num_reps), axis=0)

    # Correct for last point being known
    h_fixed = h_free - np.multiply(np.repeat(h_free[-1,:][np.newaxis, :], num_points, axis=0), np.arange(0, num_points)[:,np.newaxis] / (num_points - 1))

    std_free = np.std(h_free, axis=1)
    std_fixed = np.std(h_fixed, axis=1)

    return std_free, std_fixed


def empirical_std_tilt(
                        timedelta_days:int = None,
                        start_date:str = None,
                        start_date_format:str = '%Y-%m-%d %H:%M:%S'
                    ) -> float:
    '''
    Function to determine the empirical standard deviation of the tilt sensor data across a chosen timeframe

    Parameters
    ----------
    timedelta_days : Integer, optional
        The number of days to be considered.
        If None entire timeseries is used.
    start_date : str, optional
        Datetime string following the format sepecified in date_time_format.
        Needs to be set if timedelta_days is not None
    start_date_format : str, optional
        Datetime format string.

    Returns
    -------
    sensor_std : Float
        Empirical standard deviation of the tilt sensor across the chosen timedelta
    '''

    ### Import data
    sensor_raw = pd.read_csv(
        os.path.join(dir_data, 'Sensors_raw.csv'),
        sep=';',
        header=0,
        parse_dates=[['Date', 'Time']],
        infer_datetime_format=True,
        usecols=['Point', 'Date', 'Time', 'tilt_x0']
    )

    # Set the Dataframes index
    sensor_raw.set_index(['Point', 'Date_Time'], inplace=True)

    ### Filter chosen timedelta

    if timedelta_days is not None: # If None is specified use everything
        # If no start_date is specified use first measurement
        if start_date is None:
            start_date = sensor_raw.index.get_level_values(level=1).min()
        else:
            start_date = datetime.datetime.strptime(start_date, start_date_format)

        # Create the specified timedelta
        included_timedelta = datetime.timedelta(days=timedelta_days)

        # Filter the dataframe by Boolean mapping
        sensor_raw = sensor_raw[
            sensor_raw.index.map(
                lambda id: (id[1] >= start_date and id[1] <= start_date + included_timedelta)
            )
        ]

    ### Demean the data

    #Calculate the mean per sensor
    mean = sensor_raw.groupby(level=0).mean()

    # Demean every sensors measurements
    sensor_raw_demeaned = sensor_raw.subtract(mean, level=0)

    #Calcaulate the standard deviation
    sensor_std = sensor_raw_demeaned['tilt_x0'].std()
    
    return sensor_std

if __name__ == '__main__':
    main()