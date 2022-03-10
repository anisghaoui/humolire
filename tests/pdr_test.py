import json
from argparse import Namespace

import matplotlib.pyplot as plt
import numpy as np

from humolire.PDR.PDR import perform_pdr, compute_positions
from humolire.PDR.StepsProcessing import detect_steps
from humolire.PDR.dataloaders import load_inria_txts
from humolire.PDR.visualize import plot_trajectory, plot_sequence

fig_trajectory, ax_trajectory = None, None


def test_trajectory(path):
    frequency = 400.0  # hz
    init_pos = (0, 0)
    init_head = 0
    # load and init
    time, acce, gyro, _ = load_inria_txts(path, load_references=False)
    steps_length, steps_heading = perform_pdr(acce, gyro, weinberg_gain=1, frequency=frequency,
                                              acceleration_threshold=0.05, relative_orientation=False)
    positions, total_distance = compute_positions(steps_length, steps_heading,
                                                  initial_position=init_pos,
                                                  initial_heading=init_head)
    print(f"Total traveled distance {total_distance} meters after {len(positions)} positions")
    global fig_trajectory, ax_trajectory
    fig_trajectory, ax_trajectory = plot_trajectory(positions[:, 0], positions[:, 1], title=" PDR ",
                                                    fig=fig_trajectory, ax=ax_trajectory)


def test_step_detect(dataset_path):
    print(f"test_step_detect for {dataset_path}")
    frequency = 400
    print(f"current {frequency = } hz\n\n")

    time, acce, gyro, _ = load_inria_txts(dataset_path, load_references=False)
    acce_norm, zero, _max, _min = detect_steps(acce, acceleration_threshold=0.05)

    fig, ax = plot_sequence(acce_norm, time, title="centered acceleration norm")
    plot_sequence(acce_norm[zero], time[zero], style="+r", fig=fig, ax=ax)
    plot_sequence(acce_norm[_min], time[_min].T, style="+b", fig=fig, ax=ax)
    plot_sequence(acce_norm[_max], time[_max].T, y_axes_names="acceleration", style="+g", fig=fig, ax=ax, y_units="g")

    step_time, step_acce_norm = time[zero[10] - 3:zero[12] + 3], acce_norm[zero[10] - 3:zero[12] + 3]
    fig, ax = plot_sequence(step_acce_norm, step_time, title="Acceleration in one step")


def test_3d(data_path):
    time, acce, gyro, _ = load_inria_txts(data_path)
    keypoints_d = json.load(open(data_path + "/keypoints.json"))
    keypoints = [Namespace(**keypoints_d[str(i)]) for i in range(keypoints_d["count"])]

    initial_position = [keypoints_d["starting_position"]['x'], keypoints_d["starting_position"]['y'], 0]
    initial_heading = np.radians(np.array([0, 0, 2]))

    pdr_positions, total_distance = compute_positions(
        *perform_pdr(acce, gyro, weinberg_gain=0.2425, frequency=400.0,
                     acceleration_threshold=0.09, is_3D=True, relative_orientation=False),
        initial_position=initial_position,
        initial_heading=initial_heading,
        is_3D=True)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(pdr_positions[:, 0], pdr_positions[:, 1], pdr_positions[:, 2])


if __name__ == "__main__":
    # test_trajectory('../data/imus/18_steps')
    # test_trajectory('../data/imus/36_steps')
    # test_step_detect('../data/imus/36_steps')

    # test_3d("../data/imus/1/straight")
    test_trajectory("test_sequences/10_m_15_steps/15_pas_0")
    test_trajectory("test_sequences/10_m_15_steps/15_pas_45")
    test_trajectory("test_sequences/10_m_15_steps/15_pas_90")

    test_step_detect("test_sequences/10_m_15_steps/15_pas_0")
    test_step_detect("test_sequences/10_m_15_steps/15_pas_45")
    test_step_detect("test_sequences/10_m_15_steps/15_pas_90")

    plt.legend(["0°", None, None, "45°", None, None, "90°"])
    plt.show()
