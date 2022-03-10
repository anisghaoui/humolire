import matplotlib.pyplot as plt
import numpy as np

from humolire.PDR.dataloaders import load_ronin_txts
from humolire.PDR.visualize import plot_sequence


def test_bias(path):
    time, acce, gyro = load_ronin_txts(path)
    fig, ax = plot_sequence(time, gyro, y_axes_names=["x", "y", "z"], y_units="rad/s", x_axis_name="time", x_unit="s")

    # start the record and wait still for 10 s
    start_gyro = gyro[:500]  # 5 s * 400 hz = 200 samples
    # the average is the gyro bias
    start_avrg = np.mean(start_gyro, axis=0)
    print(f"average start gyro  = {start_avrg} rad/s")

    # subtract the start_avrg from the whole sequence:
    gyro = gyro - start_avrg

    # start the record and wait still for 10 s
    end_gyro = gyro[:-500]  # 5 s * 400 hz = 200 samples
    end_avrg = np.mean(end_gyro, axis=0)
    print(f"average end gyro  = {end_avrg} rad/s")
    # stop the recording

    plt.show()


if __name__ == "__main__":
    test_bias("test_sequences/gyro_bias")
    print("finished")
