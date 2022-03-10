"""
Last edited Anis GHAOUI: 2022-03-10

The main: this script is called by generate_figure.py to perform 10 times the same run and average it
Inputs: reads the files "config.json" then overwrites it with the main call if you desire.
returns:
    average_error_euclid: average E of all particles at a landmark
    average_error_mahalano: average M of average E of all particles at a landmark
    completed_trajectory: estimated trajectory (blue dots)
    (err_x, err_y): signed error for x and y of all particles at a landmark (useful for computing the integrity)
    (std_x, std_y): std of error for x and y of all particles at a landmark
    keypoints: landmark crossed during the trajctory

"""
import datetime
import json
import os
import os.path as osp
import random
from argparse import Namespace
from copy import deepcopy

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from PDR.MapHandler import MapHandler
from PDR.PDR import perform_pdr, compute_positions
from PDR.ParticleFilter import ParticleFilter
from PDR.common import euclidean_error, match_steps_timestamps, mahalanobis_error, compute_integrity
from PDR.dataloaders import load_inria_txts
from PDR.visualize import plot_walls, plot_particles, plot_particles_center, plot_keypoints, render_gif, plot_ellipse, \
    plot_trajectory, plot_integrity

matplotlib.rcParams["figure.autolayout"] = True
matplotlib.rcParams["figure.max_open_warning"] = 500


def main(data_path, map_path, make_fig, make_gif, fig_info=None, **kwargs):
    from pprint import pprint
    print(data_path)
    pprint(kwargs)

    # Load data and keypoints
    time, acce, gyro, recorded_timestamps = load_inria_txts(data_path)
    keypoints_d = json.load(open(data_path + "/keypoints.json"))
    keypoints = [Namespace(**keypoints_d[str(i)]) for i in range(keypoints_d["count"])]

    initial_position = [keypoints_d["starting_position"]['x'], keypoints_d["starting_position"]['y']]
    initial_heading = np.radians(keypoints_d["starting_heading"])

    map_handler = MapHandler(map_path, **kwargs)

    pf = ParticleFilter(map_handler=map_handler, initial_position={'x': initial_position[0], 'y': initial_position[1]},
                        initial_heading=initial_heading, **kwargs)

    steps_length, steps_heading = perform_pdr(acce, gyro, **kwargs)

    kwargs["relative_orientation"] = False
    pdr_positions, total_distance = compute_positions(*perform_pdr(acce, gyro, **kwargs),
                                                      initial_position=initial_position,
                                                      initial_heading=initial_heading)

    print(f"Total traveled distance {total_distance:.2f} meters after {len(pdr_positions) - 1} steps")

    particles_life = [deepcopy(pf.particles)]  # used for error measurements and plotting
    center_life = [deepcopy(pf.current_position)]  # used for error measurements and plotting
    for index, (step_length, step_heading) in tqdm(enumerate(zip(steps_length, steps_heading)),
                                                   desc="steps", unit='step', total=len(steps_length)):
        pf.update(step_length, step_heading)

        if pf.particles_count == 0:
            print("All particles are dead. Ending the experiment")
            break

        center_life.append(pf.current_position)
        particles_life.append(deepcopy(pf.particles))

    # compute error by matching timestamps to the detected steps
    landmark_indices = match_steps_timestamps(recorded_timestamps, acce, time, **kwargs)
    # BE AWARE, the function returns step indices, while we have stored the initial center_life and particles
    # This means that the position is computed when the user pressed the button, not where they landed

    # if the particles died, there are less center pdr_positions then detected steps
    landmark_indices = landmark_indices[landmark_indices < len(center_life)]
    # pick the particles at the moment we get close to a key-point
    keypoints_particles = np.array(particles_life)[landmark_indices]

    # compute errors
    errors_euclid = euclidean_error(keypoints, keypoints_particles)
    errors_mahalano = mahalanobis_error(keypoints, keypoints_particles)

    # The trajectory is completed if the last position is less than 10 m away from the last key-point
    try:
        if errors_euclid[-1] < 10.0 and len(keypoints) == len(keypoints_particles):
            completed_trajectory = True
        else:
            completed_trajectory = False
    except IndexError:
        completed_trajectory = False

    try:
        average_error_euclid = np.mean(errors_euclid)
        average_error_mahalano = np.mean(errors_mahalano)
    except ZeroDivisionError:
        average_error_euclid = np.inf
        average_error_mahalano = np.inf

    print(f"average euclidean = {average_error_euclid} m, average mahalanobis = {average_error_mahalano}")

    # adding the initial position to the integrity computing
    landmark_indices = np.concatenate(([0], landmark_indices))

    keypoints = np.concatenate([[Namespace(**{'x': initial_position[0], 'y': initial_position[1]})], keypoints])
    particles_for_integrity = np.array(particles_life)[landmark_indices]
    err_x, err_y, std_x, std_y = compute_integrity(keypoints, particles_for_integrity)

    # I know that it is inefficient to save then load but I need it for analysis purpose
    save_path = osp.dirname(osp.dirname(osp.abspath(__file__))) + "/tmp"
    if not osp.isdir(save_path):
        os.makedirs(save_path)

    if make_fig:
        fig, _ = plot_integrity((err_x, err_y), (std_x, std_y), len(keypoints), std_factor=3.0)
        fig.show()
        display_results(center_life, keypoints, make_gif, map_handler, particles_life, pdr_positions,
                        step_indices=landmark_indices, fig_info=fig_info, fps=kwargs.get("fps"))

        fig, ax = plot_walls(map_handler.walls)
        fig, ax = plot_particles_center(center_life, fig=fig, ax=ax, add_legend=False)
        fig, ax = plot_keypoints(keypoints, fig=fig, ax=ax, format=".c")
        path_ronin_data = "../data/imus/1/Ronin/straight_ronin"
        # load ronin trajectories, you need to use ronin in test mode to obtain these and then bring them here manually
        ronin_traj = np.load(path_ronin_data + "/straight_ronin.npy")
        # reverse Y in the correct layout
        ronin_traj[:, 1] *= -1
        # rotate by 180° the trajectory to have the same reference as the PDR
        ronin_traj = np.matmul(ronin_traj, np.array([[-1, 0], [0, 1]]))
        ronin_traj[:, 0] += initial_position[0]
        ronin_traj[:, 1] += initial_position[1]
        ax.plot(ronin_traj[:, 0], ronin_traj[:, 1], color='magenta')
        plot_trajectory(pdr_positions[:, 0], pdr_positions[:, 1], fig=fig, ax=ax, color="orange")

        cyan_patch = mpatches.Patch(color='cyan', label='landmarks')
        blue_patch = mpatches.Patch(color='blue', label='Our system')
        magenta_patch = mpatches.Patch(color='magenta', label='Ronin')
        red_patch = mpatches.Patch(color='orange', label='deterministic PDR')
        ax.legend(handles=[cyan_patch, blue_patch, magenta_patch, red_patch])
        fig.show()

    del particles_life
    plt.close('all')

    return average_error_euclid, average_error_mahalano, completed_trajectory, (err_x, err_y), (std_x, std_y), keypoints


def display_results(center_life, keypoints, make_gif, map_handler, particles_life, pdr_positions, step_indices,
                    fig_info=None, fps=2):
    # arrange fig_info to be split into equal line
    if fig_info is None:
        fig_info = {}
    fig_info.pop("map_path")
    config = fig_info.copy()
    fig_info['data_path'] = osp.basename(fig_info['data_path'])

    fig_info = "".join([f"{v}-" + ("\n" if i % 4 == 0 else "") for i, (k, v) in enumerate(fig_info.items())])

    fig, ax = plot_walls(map_handler.walls)
    fig, ax = plot_particles_center(center_life, fig=fig, ax=ax)
    fig, ax = plot_keypoints(keypoints, fig=fig, ax=ax, label="landmarks")
    fig, ax = plot_keypoints(list(np.array(center_life)[step_indices]), format=".r", fig=fig, ax=ax,
                             label="estimated position to landmark")
    fig.tight_layout()
    ax.legend(loc='center')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.show()

    fig_info = fig_info.replace('\n', '').replace(' ', "")

    save_path = osp.dirname(osp.dirname(osp.abspath(__file__))) + "/pngs"
    if not osp.isdir(save_path):
        os.makedirs(save_path)
    fig.savefig(save_path + f"/{fig_info}.png")

    if make_gif:
        save_path = osp.dirname(osp.dirname(osp.abspath(__file__))) + "/gifs"

        if not osp.isdir(save_path):
            os.makedirs(save_path)

        # for the gif, the legend should be only on the last figure
        particles_for_gif = []
        for particles in particles_life:
            fig_t, ax_t = plot_walls(map_handler.walls)
            fig_t, ax_t = plot_particles(particles, fig=fig_t, ax=ax_t, alpha=0.1, add_legend=False)

            plot_trajectory(pdr_positions[:, 0], pdr_positions[:, 1], fig=fig_t, ax=ax_t)
            particles_for_gif.append(fig_t)
        render_gif(particles_for_gif, fps=fps, path=save_path, filename=f"{fig_info}.gif")

        fig, ax = plot_walls(map_handler.walls)
        fig, ax = plot_keypoints(keypoints, fig=fig, ax=ax, label="landmarks")

        particles_for_gif = []
        for landmark_particles in np.array(particles_life)[step_indices]:
            fig_t, ax_t = plot_walls(map_handler.walls)
            fig_t, ax_t = plot_particles(landmark_particles, fig=fig_t, ax=ax_t,
                                         alpha=0.1, add_legend=False, label="particles")
            fig_t, ax_t = plot_keypoints(keypoints, fig=fig_t, ax=ax_t, label="landmarks")
            fig_t, ax_t = plot_keypoints(list(np.array(center_life)[step_indices]), format=".b",
                                         fig=fig_t, ax=ax_t, label="Pf positions")
            fig_t, ax_t = plot_ellipse(landmark_particles, fig=fig_t, ax=ax_t)
            particles_for_gif.append(fig_t)

            fig, ax = plot_particles(landmark_particles, fig=fig, ax=ax,
                                     alpha=0.1, add_legend=False, label="particles")
            fig, ax = plot_ellipse(landmark_particles, fig=fig, ax=ax)
        render_gif(particles_for_gif, fps=fps / 10, path=save_path, filename=f"{fig_info}-landmarks.gif")
        fig.savefig(save_path + f"/{fig_info}.png")


if __name__ == '__main__':
    # for reproducibility, fix the seed
    seed = 1  # change this value
    np.random.seed(seed)
    random.seed(seed)

    # config load order from top to bottom:
    #   - config["key"] = value
    #   - read from config.json
    #   - use default values

    config = json.load(open("config.json"))
    config["data_path"] = '../data/imus/1/straight'
    # config["data_path"] = '../data/imus/1/straight_reversed'
    # config["data_path"] = '../data/imus/1/straight_in_out'
    # config["data_path"] = '../data/imus/1/straight_in_out_reversed'
    # config["data_path"] = '../data/imus/1/simple_in_out'
    # config["data_path"] = '../data/imus/1/simple_in_out_reversed'

    # config["data_path"] = '../data/imus/1/weinberg_sequences/1f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/1r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/2f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/2r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/4f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/4r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/5f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/5r'

    config["initial_radius_range"] = [0, 0.1]
    config["initial_heading_range"] = [0, 0.15]

    # filter parameters
    config["particles_count"] = 200
    config["weight_threshold"] = 0.00001  # remove a particle if its weight is below this

    # the noise that is induced for a single detected step
    config["step_range"] = [0, 0.15]  # 45 cm/3
    config["heading_range"] = [0, 0.0623598775598]  # pi/60

    # we tried to perform the integral into multiple parts by splitting a step. it didn't enhance the pf,
    # kept for compatibility or you want ot do someone with it idk...
    config["step_split"] = 1
    config["step_range"][1] /= config["step_split"]
    config["heading_range"][1] /= config["step_split"]

    # user parameter:
    config["weinberg_gain"] = 1.07  # for user 1 weinberg
    config["acceleration_threshold"] = 0.07  # for user1

    # plotting parameters:
    config["make_fig"] = True
    config["make_gif"] = False

    config["build_grid"] = True

    # the grid is cached to be reused without being recomputed, if you change any of the following, enable reloading:
    config["min_wall_range"] = 10
    config["max_wall_range"] = 12
    config["distance_to_proba"] = None  # define a Callable as distance -> likelihood function,
    # when None will use the default one PF
    config["k_adjacency"] = 4  # 8 or 4
    config["cell_size"] = 5  # 5cm
    config["reload_cache"] = True

    # call main
    main(**config, fig_info=config)
    #

    # # this makes the OS beep at the end
    # import os
    # duration = 0.2  # seconds
    # freq = 300  # Hz
    # os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
    print(datetime.datetime.now())

    try:  # me being fancy and sending myself an sms at the end
        import requests

        logins = json.load(open("logins.json"))
        requests.post(f'{logins["provider"]}'
                      f'user={logins["user"]}&pass={logins["pass"]}'
                      f'&msg=main%20computation%20finished!')
    except FileNotFoundError:
        pass
