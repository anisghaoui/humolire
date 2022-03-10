"""
prediction functions are all here.
"""
import ahrs
import numpy as np
from ahrs import QuaternionArray, Quaternion

from .StepsProcessing import detect_steps, compute_step_length, compute_new_location


def perform_pdr(acce, gyro, weinberg_gain, frequency, relative_orientation=True, is_3D=False, **kwargs):
    # detect step events
    acce_norm, zeroes, _min, _max = detect_steps(acce, **kwargs)
    # estimate Euler angles
    q0 = None
    if kwargs.get('q0'):
        q0 = Quaternion(rpy=kwargs.get('q0'))
    orientation = estimate_orientation(acce, gyro, frequency=frequency, q0=q0).to_angles()

    if is_3D:
        orientation = orientation[_max, :]
        if relative_orientation:
            orientation = np.diff(orientation, axis=0)
            orientation = np.vstack([[0] * orientation.shape[1], orientation])

    else:

        orientation = orientation[:, 2]  # z axis
        orientation = orientation[_max]
        if relative_orientation:
            orientation = np.diff(orientation)
            orientation = np.hstack([[0], orientation])

    # compute steps lengths
    steps_lengths = []
    for step in zip(acce_norm[_max], acce_norm[_min]):
        steps_lengths.append(compute_step_length(step, weinberg_gain=weinberg_gain))

    split = kwargs.get('step_split', 1)
    if split is not None and split > 1:
        if relative_orientation:
            steps_lengths = np.repeat(steps_lengths, split) / split
            orientation = np.repeat(orientation, split) / split
        else:
            print("Skipping split step integration in PDR with absolute orientation")
            # purposely made this useless ^
    return steps_lengths, orientation


def compute_positions(steps_length, steps_heading,
                      initial_position: list = None, initial_heading: list = None, **kwargs):
    # accumulate results
    if initial_position is None:
        initial_position = np.array([0, 0])
    if initial_heading is None:
        initial_heading = 0
    positions = [initial_position]
    pos = initial_position
    steps_heading += initial_heading

    for step_heading, step_length in zip(steps_heading, steps_length):
        pos = compute_new_location(pos, step_length, step_heading, **kwargs)
        positions.append(pos)

    positions = np.array(positions)
    total_distance = np.sum(steps_length)
    return positions, total_distance


def estimate_orientation(acceleration: np.ndarray, gyroscope: np.ndarray, **kwargs: dict) -> QuaternionArray:
    """
    You may want to see this before proceeding further : https://github.com/Mayitzin/ahrs/issues/31
    Parameters
    ----------
    acceleration : np.ndarray
    gyroscope : np.ndarray
    kwargs : dict

    Returns
    -------
    np.ndarray
        orientation_quat as a quaternion array
    """
    frequency = kwargs.get('frequency', 400.0)
    _filter = kwargs.get("attitude_filter", ahrs.filters.Madgwick)

    q0 = kwargs.get('q0', np.array([1.0, 0.0, 0.0, 0.0]))
    orientation_estimation = _filter(acc=acceleration, gyr=gyroscope, frequency=frequency, q0=q0)
    orientation_quat = orientation_estimation.Q
    orientation_quat = QuaternionArray(orientation_quat)
    return orientation_quat
