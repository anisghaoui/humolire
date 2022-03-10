from typing import Tuple, List, Union

import numpy as np
from scipy.spatial.distance import mahalanobis

from .Particle import Particle
from .StepsProcessing import detect_steps

_NANO_TO_SEC = 1e9
"""
This file contain helper and common functions that might be used all over the program
"""


def integrate_array(array: np.ndarray, frequency: float) -> np.ndarray:
    """

    Parameters
    ----------
    array : A N samples, M axis shaped np array containing measurements
    frequency : float, in hz. (inverse of sample rate)

    Returns
    -------
    A N,M shaped np array of integrated data along each axis M
    """
    dt = 1.0 / frequency
    if array.ndim == 2:
        intergrated = np.cumsum(array * dt, axis=0)
    elif array.ndim == 1:
        intergrated = np.cumsum(array * dt)
    else:
        raise ValueError("error array ndim isn't allowed")

    return intergrated


def normalize_time(time):
    """
        will show the real data rate of the acquired data in hz, supposed that time is in ns

        Returns:
        --------
            time vector in seconds
    """
    time = time / _NANO_TO_SEC
    diff_time = time[1:] - time[:-1]
    average_data_rate = np.mean(1 / diff_time)
    print(f"average_data rate is = {average_data_rate} hz")
    return time - time[0]


def sanitize_data(time: np.ndarray, acce: np.ndarray, gyro: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """

    Parameters
    ----------
    time : np.ndarray
    acce : np.ndarray
    gyro : np.ndarray

    Returns
    -------
        Tuple[np.ndarray, np.ndarray, np.ndarray]

    """
    print('data sanitasation: ensuring that all data arrays are the same size :')
    print(f"{time.shape=}")
    print(f"{acce.shape=}")
    print(f"{gyro.shape=}")
    if acce.shape[0] > gyro.shape[0]:
        acce = acce[:gyro.shape[0], :]
        time = time[:gyro.shape[0]]
        print("accelerometer too short")
    if acce.shape[0] < gyro.shape[0]:
        gyro = gyro[:acce.shape[0], :]
        time = time[:acce.shape[0]]
        print("accelerometer too short")

    print("sanitasation done: new arrays sizes")
    print(f"{acce.shape=}")
    print(f"{gyro.shape=}")
    print(f"{time.shape=}")

    return time, acce, gyro


def euclidean_error(reference_positions: List, particles_lists: List[List[Particle]]) -> np.ndarray:
    if len(reference_positions) != len(particles_lists):
        print("Warning: len(reference_positions) != len(estimated_positions)")

    errors_list = []
    for ref, particles in zip(reference_positions, particles_lists):
        X = [p.position.x for p in particles]
        Y = [p.position.y for p in particles]
        W = [p.weight for p in particles]
        try:
            mean_pos_X = np.average(X, weights=W)
            mean_pos_Y = np.average(Y, weights=W)
        except ZeroDivisionError:
            mean_pos_X = np.average(X)
            mean_pos_Y = np.average(Y)
        e = np.sqrt((ref.x - mean_pos_X) ** 2 + (ref.y - mean_pos_Y) ** 2)
        errors_list.append(e)
    return np.array(errors_list)


def compute_mahalanobis(position, X: [np.ndarray, List], Y: [np.ndarray, List], W: [np.ndarray, List] = None) -> float:
    X = np.array(X)
    Y = np.array(Y)
    W = np.array(W)

    m = np.stack([X, Y], axis=0)
    try:
        covar = np.cov(m, aweights=W)  # 2x2 matrix
    except ZeroDivisionError:
        covar = np.cov(m)  # 2x2 matrix

    try:
        covar_inv = np.linalg.inv(covar)  # inverse of covariance matrix
    except np.linalg.LinAlgError:
        covar_inv = np.eye(2)

    try:
        particles_mean = [np.average(X, weights=W), np.average(Y, weights=W)]
    except ZeroDivisionError:
        particles_mean = [np.average(X), np.average(Y)]

    position = [position.x, position.y]
    return mahalanobis(position, particles_mean, covar_inv)


def mahalanobis_error(reference_positions: List, particles_lists: List[List[Particle]]) -> np.ndarray:
    errors = []
    for particles, ref_pos in zip(particles_lists, reference_positions):
        X = [p.position.x for p in particles]
        Y = [p.position.y for p in particles]
        W = [p.weight for p in particles]

        errors.append(compute_mahalanobis(ref_pos, X, Y, W))
    return np.array(errors)


def compute_integrity(keypoints: [Union[List, np.ndarray]], particles_list: List[List[Particle]]):
    Err_x, Err_y = [], []
    Std_x, Std_y = [], []
    for particles, kp in zip(particles_list, keypoints):
        X = np.array([particle.position.x for particle in particles])
        Y = np.array([particle.position.y for particle in particles])
        W = np.array([particle.weight for particle in particles])

        if np.sum(W) == 0:
            W = None

        avr_x = np.average(X, weights=W)
        avr_y = np.average(Y, weights=W)

        err_x = avr_x - kp.x
        err_y = avr_y - kp.y

        var_x = np.average((X - avr_x) ** 2, weights=W)
        var_y = np.average((Y - avr_y) ** 2, weights=W)

        std_x = np.sqrt(var_x)
        std_y = np.sqrt(var_y)

        Std_x.append(std_x)
        Std_y.append(std_y)

        Err_x.append(err_x)
        Err_y.append(err_y)
    Err_x, Err_y = np.array(Err_x), np.array(Err_y)
    Std_x, Std_y = np.array(Std_x), np.array(Std_y)
    return Err_x, Err_y, Std_x, Std_y


def match_steps_timestamps(timestamps: Union[List, Tuple, np.ndarray],
                           acceleration: np.ndarray,
                           time: np.ndarray,
                           **kwargs) -> np.ndarray:
    """

    Parameters
    ----------
    timestamps : N Union[List, np.ndarray]
        an iterable containing timestamps when the user stepped on a keypoint.
    acceleration : N,3 shaped np.ndarray
        raw acceleration data
    time : N shaped np.ndarray
        contains the time recorded by the system. timestamps, acceleration and time have to be synchronised.
    kwargs : dict
        any kwargs given are propagated to called functions:  :func: `detect_steps`
    ..Warning:: You need to have the starting position in your position list
    Returns
    -------

    """
    _, zero_cross, peaks, valleys = detect_steps(acceleration, **kwargs)
    # steps_time = time[zero_cross[1::2]]  # the valley is closer to the landing than the peak
    steps_time = time[valleys]  # the valley is closer to the landing than the peak
    timestamps = np.array(timestamps)
    matched_steps = []
    for ts in timestamps:
        step_idx = (np.abs(steps_time - ts)).argmin()
        # we need the position after the step
        matched_steps.append(step_idx * kwargs.get('split_step', 1))

    # print(steps_time.shape)
    # print(len(matched_steps))
    # print(steps_time.max())
    # print(np.array(matched_steps).max())
    return np.array(matched_steps)
