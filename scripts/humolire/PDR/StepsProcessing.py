# author: GHAOUI Anis
# date: 2021-01-18

from typing import Tuple, Union

import numpy as np
from scipy import signal

from humolire.PDR.visualize import plot_sequence


def filter_acceleration(acceleration: np.ndarray, frequency: float = 400, **kwargs) -> \
        (np.ndarray, np.ndarray):
    """
    This function filters then normalises and centers around 0 the input 3D acceleration

    Parameters
    ----------
    acceleration : N,3 shaped numpy array,
        Contains accelerometer data sequence in m/s^2 un-normalised with gravity offset
    frequency : float, default = 400 hz
        Frequency at which the data has been acquired

    kwargs : dict:
    - gravity : float. Earth's gravity value, default 9.7 m/s^2
    - butter_order : order of the low pass Butterworth filter. default: 2
    - cutoff_step_frequency : float. Represents the number of step one can walk in a second. default: 2 hz
    - filter_all_accelerations : is set to True, will filter along each axis

    Returns
    -------
    acceleration_norm : N,1 np.ndarray
        The filtered acceleration norm

    acceleration_sequence:  N,3 np.ndarray, optional
        The filtered acceleration axis. Set the argument filter_all_accelerations=True to have it.
        else returns the input.
    """

    gravity = kwargs.get("gravity", 9.7)
    delta_time = 1.0 / frequency  # seconds / sample
    low_pass_cutoff = kwargs.get("cutoff_step_frequency", 2.0)  # hz
    butter_order = kwargs.get("butter_order", 2)  # 2nd order butterworth

    # noinspection PyTupleAssignmentBalance
    lp_numer, lp_denom = signal.butter(butter_order, 2.0 * low_pass_cutoff * delta_time, "low")
    # normalise by gravity
    if len(acceleration.shape) > 1:
        acceleration = np.linalg.norm(acceleration, axis=1)
    acceleration /= gravity
    # center the acce norm around 0
    acceleration = signal.filtfilt(lp_numer, lp_denom, acceleration) - np.mean(acceleration)
    plot_sequence(acceleration)
    return acceleration


def detect_steps(acceleration: np.ndarray, acceleration_threshold: float, **kwargs) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    This function calls :func: `filter_acceleration` function by itself.

    Parameters
    ----------
    acceleration : N,3 shaped np.ndarray,
        Contains accelerometer data sequence in m/s^2 un-normalised with an gravity offset
    acceleration_threshold : float, default : 0.07
        The threshold at which the acceleration can be considered as part of a step.

    Returns
    -------
    acceleration_norm: np.ndarray
        The filtered acceleration norm
    zero_crossing: np.ndarray
        Indices where the normalised centered acceleration crosses 0.
    peaks: np.ndarray
        Indices where the local acceleration peaks are located
    valleys: np.ndarray
        Indices where the local acceleration peaks are located
    """

    acceleration = filter_acceleration(acceleration, **kwargs)
    zero_crossing = np.where((acceleration[:-1] * acceleration[1:]) < 0)[0]
    # delete the first index element if the wave is negative to ensure the acceleration goes positive then negative
    if acceleration[zero_crossing[0]] > 0 and acceleration[zero_crossing[0] + 1] < 0:
        zero_crossing = zero_crossing[1:]
    # t_plus and t_minus are respectively times where the signal transits from negative, positive to positive, negative
    peaks = []
    valleys = []
    for t_plus, t_minus in zip(zero_crossing[::2], zero_crossing[1::2]):
        max_index = np.argmax(acceleration[t_plus: t_minus])
        peaks.append(max_index + t_plus)

    for t_plus, t_minus in zip(zero_crossing[2::2], zero_crossing[1::2]):
        min_index = np.argmin(acceleration[t_minus: t_plus])
        valleys.append(min_index + t_minus)

    peaks = np.array(peaks, dtype=np.int)
    valleys = np.array(valleys, dtype=np.int)

    indices = np.where(acceleration[peaks] > acceleration_threshold)[0]
    peaks = peaks[indices]
    indices = np.where(acceleration[valleys] < -acceleration_threshold)[0]
    valleys = valleys[indices]

    if len(peaks) == 0:
        print("Warning: no acceleration peaks found during sequence")
    if len(valleys) == 0:
        print("Warning: no acceleration valleys found during sequence")

    return acceleration, zero_crossing, peaks, valleys


def compute_step_length(accelerometer_norm: Union[Tuple, np.ndarray], weinberg_gain: float) -> float:
    """
    # Article:  Xinyu Hou 2020 (eq 7, 8, 9, 10, 11)
    Weingberg method (eq 8):
    weinberg_gain is K in the formula

    Parameters
    ----------
    accelerometer_norm : array-like
        The function will find the map and min of this iterable. In m/s^2
    weinberg_gain :
        The gain is the empirical value that is  proportional to the pedestrian leg length

    Returns
    -------
    step_length : float
        The length of the walked step in meters
    """
    step_length = weinberg_gain * np.power(np.max(accelerometer_norm) - np.min(accelerometer_norm), 1.0 / 4)
    return step_length


def compute_new_location(previous_location, step_length: float, orientation, is_3D=False, **kwargs) -> np.ndarray:
    """ TODO: redoc
     Parameters
    ----------
    previous_location: nd.array, list, tuple
        previous 2D position [x,y] in meters
    step_length: float
        The length of the walked step in meters
    orientation: float
        The new heading of the step in radians

    Returns
    -------
        an [x,y] pair of the new position in meters
    """
    if is_3D:
        yaw = orientation[2]
        pitch = orientation[0]
        theta = np.pi / 2 - pitch
        x = np.cos(yaw) * np.sin(theta) * step_length + previous_location[0]
        y = np.sin(yaw) * np.sin(theta) * step_length + previous_location[1]
        z = np.cos(theta) * step_length + previous_location[2]
        return np.array([x, y, z])
    else:
        # print(orientation)
        x = np.cos(orientation) * step_length + previous_location[0]
        y = np.sin(orientation) * step_length + previous_location[1]
        return np.array([x, y])
