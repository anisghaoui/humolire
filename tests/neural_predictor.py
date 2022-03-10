import matplotlib.patches as mpatch

from humolire.PDR.MapHandler import MapHandler
from humolire.PDR.PDR import perform_pdr
from humolire.PDR.ParticleFilter import ParticleFilter
from humolire.PDR.StepsProcessing import detect_steps
from humolire.PDR.dataloaders import load_ronin_txts
from humolire.PDR.visualize import *


def adjust_angle_array(angles):
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


def plot_heading(trajectory, heading, fig=None, ax=None):
    p_l = {'m': '--', 'c': 'b'}
    handles = [mpatch.Patch(color=p_l['c'], label='Predicted')]

    sh, l, h_w, window = 25, 0.9, 0.3, 75
    if fig is None:
        fig, ax = plt.figure(figsize=(12, 12))

    ax.plot(trajectory[:, 0], trajectory[:, 1], color='black')
    for i in range(window, heading.shape[0] - (window + sh), 200):
        p_i = heading[i + sh]
        ax.arrow(trajectory[i + sh, 0], trajectory[i + sh, 1],
                 np.cos(p_i) * l, np.sin(p_i) * l,
                 head_width=h_w, overhang=0.8,
                 linestyle=p_l['m'], color=p_l['c'])

    ax.axis('equal')
    ax.legend(handles=handles)
    fig.tight_layout()
    return fig, ax


def main(path, start_position=[7, 6.5], start_heading=0):
    n_particles = 100
    radius_range = (0, 0.15 / 3)
    heading_range = (0, np.pi / 30)
    subset_size = 10
    frequency = 200.0
    max_wall_range = 10

    # import RoNIN's lstm trajectory and heading predictions
    trajectory = np.load("../data/neural/longest_run/longest_run.npy") @ np.array(
        [(np.cos(np.pi), 0), (0, np.cos(np.pi))])
    trajectory += np.repeat(np.array([start_position]), trajectory.shape[0]).reshape(trajectory.shape)

    heading = np.load("../data/neural/longest_run/longest_run_lstm_heading.npy")
    # they have some weird angle shenanigans where it evolves negatively
    heading = adjust_angle_array(-np.arctan2(heading[:, 0], heading[:, 1]) + np.pi + np.pi / 2 + start_heading)

    map_file = '../data/maps/map_data.json'
    map_matcher = MapHandler(map_file)
    fig, ax = plot_walls(map_matcher.walls)
    plot_heading(trajectory, heading, fig, ax)
    fig.show()

    # slice the sequence:
    time, acce, gyro = load_ronin_txts(path)
    gyro = gyro[::2, :]
    acce = acce[::2, :]
    _, _, peaks, _ = detect_steps(acce)

    # peaks = 2 * peaks
    peaks = peaks[peaks < trajectory.shape[0]]

    steps_heading = np.diff(heading[peaks])

    trajectory = trajectory[peaks, :]
    trajectory = np.diff(trajectory, axis=0)
    steps_length = np.linalg.norm(trajectory, axis=1)
    steps_length_pdr, steps_heading_pdr = perform_pdr(acce, gyro, frequency=200)

    plt.figure()
    plt.plot(steps_length)
    plt.plot(steps_length_pdr)
    plt.figure()
    plt.plot(np.degrees(steps_heading))
    plt.plot(np.degrees(adjust_angle_array(steps_heading_pdr)))
    plt.show()
    pf = ParticleFilter(n_particles, map_matcher, {'x': start_position[0], 'y': start_position[1]},
                        initial_heading=start_heading, step_range=radius_range, heading_range=heading_range,
                        subset_size=10, frequency=200.0, max_wall_range=10)

    particles_life = []
    center_life = []
    # for the gif, the legend should be only on the last figure,
    fig_t, ax_t = plot_walls(map_matcher.walls)
    fig_t, ax_t = plot_particles(pf.particles, fig=fig_t, ax=ax_t, alpha=0.5)
    particles_life.append(fig_t)

    center_life.append(pf.current_position)
    for index, (step_length, step_heading) in enumerate(zip(steps_length, steps_heading)):
        pf.update(step_length, step_heading)
        if pf.particles_count == 0:
            print("ending the experiment")
            break
        center_life.append(pf.current_position)

        # for the gif, the legend should be only on the last figure,
        fig_t, ax_t = plot_walls(map_matcher.walls)
        fig_t, ax_t = plot_particles(pf.particles, fig=fig_t, ax=ax_t, alpha=0.5,
                                     add_legend=index == len(steps_length) - 1)
        particles_life.append(fig_t)
    fig, ax = plot_walls(map_matcher.walls)
    fig, ax = plot_particles_center(center_life, fig=fig, ax=ax,
                                    title=f"center points trajectory {n_particles=}\n  "
                                          f"gaussian: radius_std={radius_range[1]},"
                                          f" angle_std={heading_range[1]}\n{subset_size=}{max_wall_range=}")
    fig.show()


if __name__ == "__main__":
    path = "../data/imus/18_steps"
    main(path, start_position=[7, 5.5], start_heading=np.pi / 2)
