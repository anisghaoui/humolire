"""
Ask the user to walk 10m while recording their IMUs. Do it multiple times to have a reliable ground truth.
The Weinberg gain can't change.
"""
import os.path as osp

import matplotlib.pyplot as plt
import numpy as np

from humolire.PDR.PDR import perform_pdr, compute_positions
from humolire.PDR.dataloaders import load_inria_txts


def test_weinberg(datapath, **kwargs):
    time, acce, gyro, recorded_timestamps = load_inria_txts(datapath, load_references=False)

    positions, total_distance = compute_positions(
        *perform_pdr(acce, gyro, **kwargs, frequency=400))
    print(f"for {osp.basename(datapath)}, total traveled distance {total_distance:.2f} "
          f"meters after {len(positions)} steps\n")


if __name__ == "__main__":
    # a 186
    test_weinberg('test_sequences/weinberg/1_10m', weinberg_gain=1.07, acceleration_threshold=0.07)
    test_weinberg('test_sequences/weinberg/1_10m_2', weinberg_gain=1.07, acceleration_threshold=0.07)

    # f 190
    test_weinberg('test_sequences/weinberg/2_10m', weinberg_gain=1.1, acceleration_threshold=0.05)
    test_weinberg('test_sequences/weinberg/2_10m_2', weinberg_gain=1.1, acceleration_threshold=0.05)

    # e 180
    test_weinberg('test_sequences/weinberg/3_10m', weinberg_gain=1.05, acceleration_threshold=0.07)
    test_weinberg('test_sequences/weinberg/3_10m_2', weinberg_gain=1.05, acceleration_threshold=0.07)

    # b 165
    test_weinberg('test_sequences/weinberg/4_10m', weinberg_gain=0.95, acceleration_threshold=0.09)
    test_weinberg('test_sequences/weinberg/4_10m_2', weinberg_gain=0.95, acceleration_threshold=0.09)

    weinbergs = [1.07, 1.1, 1.05, 0.95]
    heights = [186, 190, 180, 163]
    plt.scatter(heights, weinbergs, color='red', label='measures')
    coef = np.polyfit(heights, weinbergs, 1)
    poly1d_fn = np.poly1d(coef)
    plt.plot(heights, poly1d_fn(heights), "--")
    plt.grid()
    plt.xlabel('pedestrian height (cm)')
    plt.ylabel('weinberg gain')
    plt.show()
