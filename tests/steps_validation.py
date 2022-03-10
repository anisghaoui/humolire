import numpy as np
from matplotlib import pyplot as plt

from humolire.PDR.StepsProcessing import detect_steps, filter_acceleration, compute_step_length
from humolire.PDR.dataloaders import load_inria_txts
from humolire.PDR.visualize import plot_sequence


def norme_step(path):
    # the steps in this test have been timestamped by the user using the INRIA logging App
    # https://play.google.com/store/apps/details?id=fr.inria.tyrex.senslogs
    time, acce, gyro, refs_timing = load_inria_txts(path)
    acce_norm, zero, _max, _min = detect_steps(acce, acceleration_threshold=0.05)
    fig, ax = plot_sequence(acce_norm, time)
    plot_sequence(acce_norm[zero], time[zero], style="+r", fig=fig, ax=ax)
    plot_sequence(acce_norm[_min], time[_min].T, style="+b", fig=fig, ax=ax)
    plot_sequence(acce_norm[_max], time[_max].T, style="+g", fig=fig, ax=ax,
                  y_units="g", y_axes_names="acceleration")
    plot_sequence(np.zeros(refs_timing.shape), refs_timing, style="y*", fig=fig, ax=ax,
                  title="")
    ax.legend(["filtered acceleration", "zero crossing", "valleys", "peaks", "timestamps"])

    plt.show()


def z_axis_step(path):
    time, acce, gyro, refs_timing = load_inria_txts(path)
    print(acce.shape)
    acce = acce[:, 2]
    print(acce.shape)
    acce, zero, _max, _min = detect_steps(acce, acceleration_threshold=0.05)
    fig, ax = plot_sequence(acce, time)
    plot_sequence(acce[zero], time[zero], style="+r", fig=fig, ax=ax)
    plot_sequence(acce[_min], time[_min].T, style="+b", fig=fig, ax=ax)
    plot_sequence(acce[_max], time[_max].T, style="+g", fig=fig, ax=ax, y_units="g", y_axes_names="acceleration")
    plot_sequence(np.zeros(refs_timing.shape), refs_timing, style="y*", fig=fig, ax=ax, title="")
    ax.legend(["filtered acceleration", "zero crossing", "valleys", "peaks", "timestamps"])

    plt.show()


def single_step_display(path):
    time, acce, gyro, refs_timing = load_inria_txts(path, load_references=False)
    # acce = acce[1240:1675]
    # time = time[1240:1675]
    acce_norm, _ = filter_acceleration(acce)
    acce_norm, zero, _max, _min = detect_steps(acce, acceleration_threshold=0.7075)
    fig, ax = plot_sequence(acce_norm, time)
    plot_sequence(acce_norm[zero], time[zero], style="or", fig=fig, ax=ax, markersize=5)
    plot_sequence(acce_norm[_min], time[_min].T, style="ob", fig=fig, ax=ax)
    plot_sequence(acce_norm[_max], time[_max].T, style="og", fig=fig, ax=ax,
                  x_unit="s", x_axis_name="time", y_units="g", y_axes_names="acceleration")
    ax.legend(["filtered acceleration", "zero crossing", "valleys", "peaks"])

    # # min_t = time[_min][0]
    # # max_t = time[_max][0]
    # # A_min = acce_norm[_min]
    # # A_max = acce_norm[_max]
    # # ax.annotate("$t_-$", (min_t, 0), (min_t, +0.005))
    # # ax.annotate("$t_+$", (max_t, 0), (max_t, -0.005))
    # # ax.annotate("$A_{max}$", (max_t, A_max), (max_t, A_max + 0.005))
    # # ax.annotate("$A_{min}$", (min_t, A_min), (min_t, A_min - 0.005))
    # # ax.plot((max_t, max_t), [0, A_max], "--g")
    # # ax.plot((min_t, min_t), [0, A_min], "--b")

    plt.show()

    steps_lengths = []

    for step in zip(acce_norm[_max], acce_norm[_min]):
        steps_lengths.append(compute_step_length(step, weinberg_gain=1.07))
    print(steps_lengths)


if __name__ == "__main__":
    norme_step("test_sequences/7_steps")
    z_axis_step("test_sequences/7_steps")

    # norme_step("test_sequences/8_steps")
    # norme_step("test_sequences/little_step")

    # single_step_display("test_sequences/8_steps")
    # single_step_display("../data/imus/1/straight_reversed")

    # single_step_display("test_sequences/step_test/1")
    # single_step_display("test_sequences/step_test/2")
    # single_step_display("test_sequences/step_test/half")
    # single_step_display("test_sequences/step_test/full")
    # single_step_display("test_sequences/step_test/2_steps")
