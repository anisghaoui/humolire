from os import path as osp

import ahrs
import numpy as np
from matplotlib import pyplot as plt

from humolire.PDR.common import integrate_array
from humolire.PDR.visualize import plot_sequence


def test_repo_imu(path):
    # repoIMU dataset : https://github.com/agnieszkaszczesna/RepoIMU/blob/main/TStick/TStick_Test11_Trial1.csv
    data = np.genfromtxt(osp.join(path), dtype=float, delimiter=';', skip_header=2)

    frequency = 100.0
    time = data[:, 0]
    ref_quaternions = data[:, 1:5]
    acce = data[:, 5:8]
    gyro = data[:, 8:11]

    orientation = ahrs.filters.Madgwick(gyr=gyro, acc=acce, frequency=frequency)
    # ref_q0_conjugate = ref_quaternions[0] * np.array([1.0, -1.0, -1.0, -1.0])  # conjugate
    # errors = ahrs.utils.metrics.qad(ref_quaternions, ref_q0_conjugate)

    plot_sequence(ref_quaternions, time, y_axes_names=["w", "x", "y", "z"], title='ref quaternions')
    plt.show()

    plot_sequence(orientation.Q, time, y_axes_names=["w", "x", "y", "z"], title='madgwick quaternions')
    plt.show()

    integrated_gyro = integrate_array(gyro, frequency=frequency)
    plot_sequence(integrated_gyro, time, title=f'integrated gyroscope data: {path}', y_units="radians")
    plt.show()

    ref_angles = ahrs.QuaternionArray(ref_quaternions).to_angles()
    angles = ahrs.QuaternionArray(orientation.Q).to_angles()

    plot_sequence(angles, time, y_axes_names=["x", "y", "z"], title=f'Madgwick orientations {path}', y_units="radians")
    plt.show()

    plot_sequence(ref_angles, time, y_axes_names=["x", "y", "z"], title=f'ref orientations {path}', y_units="radians")
    plt.show()

    # plot error
    # plot_sequence(time, errors, title='error : vicon - Madgwick Quaternion Angle Difference', y_units="radians")
    # plt.show()


if __name__ == '__main__':
    test_repo_imu('test_sequences/TStick_Test03_Trial1.csv')
