import json

import matplotlib.pyplot as plt
import numpy as np

from PDR.PDR import perform_pdr, compute_positions
from PDR.StepsProcessing import detect_steps, compute_step_length
from PDR.dataloaders import load_inria_txts
from PDR.visualize import plot_trajectory


def test_trajectory(path):
    frequency = 400.0  # hz
    init_pos = (0, 0)
    init_head = 0
    weinberg_gain = 1.07
    acceleration_threshold = 0.07
    # load and init
    time, acce, gyro, _ = load_inria_txts(path, load_references=False)
    steps_length, steps_heading = perform_pdr(acce, gyro, weinberg_gain=weinberg_gain,
                                              frequency=400, relative_orientation=False,
                                              acceleration_threshold=acceleration_threshold)
    positions, total_distance = compute_positions(steps_length, steps_heading,
                                                  initial_position=init_pos,
                                                  initial_heading=init_head)
    print(f"Total traveled distance {total_distance} meters after {len(positions)} steps")
    fig_trajectory, ax_trajectory = plot_trajectory(positions[:, 0], positions[:, 1], title=" PDR using madgwick")

    # test with Z-axis acceleration only:
    acce, zeroes, _min, _max = detect_steps(acce[:, 2], weinberg_gain=weinberg_gain,
                                            frequency=400, acceleration_threshold=acceleration_threshold)
    steps_lengths = []
    for step in zip(acce[_max], acce[_min]):
        steps_lengths.append(compute_step_length(step, weinberg_gain=weinberg_gain, ))
    positions, total_distance = compute_positions(steps_length, steps_heading,
                                                  initial_position=init_pos,
                                                  initial_heading=init_head)
    print(f"Total traveled distance {total_distance} meters after {len(positions)} steps")
    fig_trajectory, ax_trajectory = plot_trajectory(positions[:, 0], positions[:, 1], title=" PDR using madgwick")


def test_3d(data_path):
    time, acce, gyro, _ = load_inria_txts(data_path)
    keypoints_d = json.load(open(data_path + "/keypoints.json"))
    # keypoints = [Namespace(**keypoints_d[str(i)]) for i in range(keypoints_d["count"])]

    initial_position = [keypoints_d["starting_position"]['x'], keypoints_d["starting_position"]['y'], 0]
    initial_heading = np.radians(np.array([0, 0, 2]))

    steps_lengths, orientation = perform_pdr(acce, gyro, weinberg_gain=1.07,
                                             frequency=400.0, acceleration_threshold=0.07,
                                             is_3D=True, relative_orientation=False)

    pdr_positions, total_distance = compute_positions(steps_lengths, orientation,
                                                      initial_position=initial_position,
                                                      initial_heading=initial_heading,
                                                      is_3D=True)

    pdr_positions[:, 2][np.any(pdr_positions[:, 2] < 0.0001) and np.any(pdr_positions[:, 2] > -0.00001)] = 0
    #
    plt.figure()

    ax = plt.axes(projection='3d')
    ax.plot3D(pdr_positions[:, 0], pdr_positions[:, 1], pdr_positions[:, 2])
    ax.plot3D(pdr_positions[0, 0], pdr_positions[0, 1], pdr_positions[0, 2], '+r')
    ax.set(xlabel='x', ylabel='y', zlabel='z')

    fig, ax = plt.subplots()
    ax.plot(pdr_positions[:, 2], label="height")
    ax.set(xlabel="time", ylabel="Z (meters)")

    print("Z derivative")
    z_drift = np.diff(pdr_positions[:, 2])
    mean_drift = np.mean(z_drift)
    std_drift = np.std(z_drift)
    print(mean_drift)
    print(std_drift)

    plt.show()


if __name__ == "__main__":
    test_trajectory('../data/imus/1/straight')
    # test_trajectory('../data/imus/1/straight')

    # test_3d("../data/imus/1/straight")
    # test_3d("../data/imus/1/straight_reversed")
    # test_3d("../data/imus/1/straight_in_out_reversed")

    plt.show()
