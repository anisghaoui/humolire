import numpy as np
from ahrs.filters import Madgwick
from matplotlib import pyplot as plt

from humolire.PDR.PDR import estimate_orientation
from humolire.PDR.common import integrate_array
from humolire.PDR.dataloaders import load_inria_txts
from humolire.PDR.visualize import plot_sequence


def adjust_angle_array(angles):
    # https://github.com/Sachini/ronin/blob/805b7f0f28bb164ce89ada9ac05a9470dbe3d715/source/math_util.py#L7
    """
    Resolve ambiguities within a array of angles. It assumes neighboring angles should be close.
    Args:
        angles: an array of angles.

    Return:
        Adjusted angle array.
    """
    new_angle = np.copy(angles)
    angle_diff = angles[1:] - angles[:-1]

    diff_cand = angle_diff[:, None] - np.array([-np.pi * 4, -np.pi * 2, 0, np.pi * 2, np.pi * 4])
    min_id = np.argmin(np.abs(diff_cand), axis=1)

    diffs = np.choose(min_id, diff_cand.T)
    new_angle[1:] = np.cumsum(diffs) + new_angle[0]
    return new_angle


def test_orientation_estimation(dataset_path):
    print(f"{__name__} for {dataset_path}")
    frequency = 418
    time, acce, gyro, _ = load_inria_txts(dataset_path, load_references=False)

    q0 = np.array([1.0, 0, 0, 0])
    integrated_gyro = integrate_array(gyro, frequency=frequency)
    orientation_quat = estimate_orientation(acce, gyro, attitude_filter=Madgwick, frequency=frequency, q0=q0)

    fig, ax = plot_sequence(integrated_gyro, time, y_axes_names=["x", "y", "z"],
                            title=f"integrated_gyro : {dataset_path}", y_units="radians")

    orientation_angles = orientation_quat.to_angles()
    # just ensure that ther is no jump between Pi and 2 pi
    orientation_angles[:, 2] = adjust_angle_array(orientation_angles[:, 2])

    plot_sequence(orientation_angles, time, y_axes_names=["x", "y", "z"],
                  title=None, y_units="radians", x_axis_name="time",
                  fig=fig, ax=ax)
    fig.legend(["Gyro integration", "Magwick prediction"])
    # plot_sequence(orientation_quat, time, y_axes_names=["w", "x", "y", "z"],
    #               title=f"orientation quat :{dataset_path}", y_units="")


if __name__ == '__main__':
    test_orientation_estimation('../data/imus/1/8_shape')
    plt.show()
