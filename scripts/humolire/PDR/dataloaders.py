import os.path as osp

import numpy as np

from .common import normalize_time, sanitize_data


def load_ronin_txts(data_path: str):
    """
    the app: https://github.com/Sachini/ronin
    data_path is path to folder
    returns a namespace.sensor and a dictionary["sensor"] = np.array
    the list of sensors is _SENSORS appended with "time"
    """
    if data_path[-1] != '/':
        data_path += "/"
    if osp.isdir(data_path) is False:
        raise IOError(f"missing folder {data_path}")

    acce = np.genfromtxt(osp.join(data_path) + "acce.txt", dtype="float")
    time = normalize_time(acce[:, 0])
    acce = acce[:, 1:]
    gyro = np.genfromtxt(osp.join(data_path) + "gyro.txt", dtype="float")[:, 1:]

    return sanitize_data(time, acce, gyro)


def load_inria_txts(data_path, load_references=True):
    """
    The app: https://github.com/tyrex-team/senslogs (also available in the google playstore)

    You need to create a text file named 'timestamps.txt' of which each line contains 'x y'.

    Parameters
    ----------
    data_path : str or Path-like

    Returns
    -------
    """
    if data_path[-1] != '/':
        data_path += "/"
    if osp.isdir(data_path) is False:
        raise IOError(f"missing folder {data_path}")

    acce = np.genfromtxt(osp.join(data_path) + "accelerometer-calibrated.txt",
                         dtype=float, delimiter=' ', skip_header=6)
    gyro = np.genfromtxt(osp.join(data_path) + "gyroscope-calibrated.txt",
                         dtype=float, delimiter=' ', skip_header=6)
    if load_references:
        recorded_timestamp = np.genfromtxt(osp.join(data_path) + "references.txt",
                                           dtype=float, delimiter=' ', skip_header=6)[:, 0]
    else:
        recorded_timestamp = None

    diff_in_time = abs(acce.shape[0] - gyro.shape[0])
    if acce.shape[0] > gyro.shape[0]:
        acce = acce[diff_in_time:, :]
        time = acce[:, 1]
    if acce.shape[0] < gyro.shape[0]:
        gyro = gyro[diff_in_time:, :]
        time = gyro[:, 1]

    acce = acce[:, 2:5]
    gyro = gyro[:, 2:5]

    sampling_frequency = 1.0 / np.mean(np.diff(time))
    print(f"Data sampling frequency is {sampling_frequency} Hz")
    return time, acce, gyro, recorded_timestamp
