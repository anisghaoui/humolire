# This file generates most figures presented in the paper.
# Please bare in mind that some GIMP/LaTex
# post-processing is still required
# Load an experiment then study the effect of the most important parameters
# if recompute is True the function will rerun the test overwriting previous results,
# if recompute is False it will only read the already existing *.npy object and display them

import datetime
import json
import multiprocessing
import os
import os.path as osp
import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from main import main
from humolire.PDR.visualize import plot_integrity

matplotlib.rcParams["figure.autolayout"] = True
matplotlib.rcParams["figure.figsize"] = (8, 6)

N_THREAD = 10  # tries to run python in parallel (lol as if)


def thread_luncher(id, context: dict):
    # this is the thread starter common to every parallel process
    config = context.copy()
    config['id'] = id
    euclid, maha, complet, errs, stds, keypoints = main(**config, fig_info=config)
    return euclid, maha, complet, errs, stds, keypoints


def particle_count_vs_errors(config_origin: dict, save_path: str = '../results/', recompute=True):
    print("particles")
    # freeze the config copy for each test:
    config = config_origin.copy()
    if not osp.isdir(save_path):
        os.makedirs(save_path)
    if save_path[-1] != '/':
        save_path += '/'
    save_path = save_path + osp.basename(config['data_path'])

    if recompute:
        # first test:
        config["test_0"] = osp.basename(config['data_path'])  # stains the pngs/gifs with a unique trace

        # E = Euclidean,  M = Mahalonobis
        # The effect of particles number on the randomness of the results, expect a E,M vs N particles graph
        # ensure that the seed always changes
        seed = None
        np.random.seed(seed)
        random.seed(seed)
        N_range = [10, 25, 50, 100, 150, 200, 500, 1000]
        for grid in [True, False]:
            config["build_grid"] = grid

            Euclidean = []
            Mahalanobis = []
            Completed = []
            for N in N_range:
                config["particles_count"] = N

                # parallel call of thread_luncher() for N_THREAD return an iterable object of len(N_THREAD)
                pool = multiprocessing.Pool(N_THREAD)
                E, M, C, _, _, _ = zip(*pool.starmap(thread_luncher, zip(range(0, N_THREAD), [config] * N_THREAD)))

                E_mean = np.mean(E)
                E_std = np.std(E)

                M_mean = np.mean(M)
                M_std = np.std(M)

                Euclidean.append((E_mean, E_std))
                Mahalanobis.append((M_mean, M_std))
                Completed.append(C)

            # saving them in case you don't want to redo the whole experiment
            np.save(f"{save_path}Euclidean_{grid}_particles.npy", np.array(Euclidean))
            np.save(f"{save_path}Mahalanobis_{grid}_particles.npy", np.array(Mahalanobis))
            np.save(f"{save_path}N_range_{grid}_particles.npy", N_range)
            np.save(f"{save_path}Completed_{grid}_particles.npy", np.array(Completed))

    Euclidean_g = np.load(f"{save_path}Euclidean_{True}_particles.npy")
    Mahalanobis_g = np.load(f"{save_path}Mahalanobis_{True}_particles.npy")

    Euclidean_no_g = np.load(f"{save_path}Euclidean_{False}_particles.npy")
    Mahalanobis_no_g = np.load(f"{save_path}Mahalanobis_{False}_particles.npy")

    N_range = np.load(f"{save_path}N_range_{False}_particles.npy")

    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.errorbar.html
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.errorbar(N_range, Euclidean_g[:, 0], yerr=Euclidean_g[:, 1], fmt="b+-",
                label="average Euclidean distance (meters), grid", alpha=0.5, capsize=5)
    ax.errorbar(N_range, Mahalanobis_g[:, 0], yerr=Mahalanobis_g[:, 1], fmt="c+-",
                label="average Mahalanobis distance, grid", alpha=0.5, capsize=5)
    ax.errorbar(N_range, Euclidean_no_g[:, 0], yerr=Euclidean_no_g[:, 1], fmt="r+-",
                label="average Euclidean distance (meters), no grid", alpha=0.5, capsize=5)
    ax.errorbar(N_range, Mahalanobis_no_g[:, 0], yerr=Mahalanobis_no_g[:, 1], fmt="y+-",
                label="average Mahalanobis distance, no grid", alpha=0.5, capsize=5)
    ax.set(xlabel="particles count", ylabel="measured error", xscale='log')
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    fig.savefig(f'{save_path}particles.png')

    # Completed_no_g = np.load(f"{save_path}Completed_{False}_particles.npy")
    # Completed_g = np.load(f"{save_path}Completed_{True}_particles.npy")

    fig.show()


def grid_adjacency_effect(config_origin: dict, save_path: str = '../results/', recompute=True):
    print("grid_effect")
    # TEST 2 effect of the grid function on the Errors
    # freeze the of config copy for each test:
    config = config_origin.copy()
    if not osp.isdir(save_path):
        os.makedirs(save_path)

    max_ranges = np.arange(2, 100 / config_origin["cell_size"])  # 1 meter / cell_size in cm)
    if recompute:
        config["test_1"] = osp.basename(config['data_path'])

        for grid in [True]:
            # for grid in [True, False]:
            config["build_grid"] = grid
            config["reload_cache"] = True

            if grid:
                # for adj in [4, 8]: # you can study an 8 or 4 adjacency grid
                for adj in [4]:
                    Euclidean = []
                    Mahalanobis = []
                    Completed = []

                    config["k_adjacency"] = adj
                    for max in max_ranges:

                        E = []
                        M = []
                        C = []

                        config["min_wall_range"] = 2
                        config["max_wall_range"] = max
                        for _ in range(10):
                            e, m, c, _, _, _ = main(**config, fig_info=config.copy())

                            C.append(c)
                            if c:
                                E.append(e)
                                M.append(m)

                        E_mean = np.mean(E)
                        E_std = np.std(E)

                        M_mean = np.mean(M)
                        M_std = np.std(M)

                        Euclidean.append((E_mean, E_std))
                        Mahalanobis.append((M_mean, M_std))
                        Completed.append(C)

                    # saving them in case you don't want to redo the whole experiment
                    np.save(f"{save_path}k_adj_{adj}_{grid}_E.npy", np.array(Euclidean))
                    np.save(f"{save_path}k_adj_{adj}_{grid}_M.npy", np.array(Mahalanobis))
                    np.save(f"{save_path}k_adj_{adj}_{grid}_C.npy", np.array(Completed))
            else:
                E = []
                M = []
                C = []
                for _ in range(10):
                    e, m, c, _, _, _ = main(**config, fig_info=config.copy())

                    C.append(c)
                    if c:
                        E.append(e)
                        M.append(m)

                E_mean = np.mean(E)
                E_std = np.std(E)

                M_mean = np.mean(M)
                M_std = np.std(M)

                # saving them in case you don't want to redo the whole experiment
                np.save(f"{save_path}k_adj_{grid}_E.npy", np.array((E_mean, E_std)))
                np.save(f"{save_path}k_adj_{grid}_M.npy", np.array((M_mean, M_std)))
                np.save(f"{save_path}k_adj_{grid}_C.npy", np.array(C))

    E_4_grid = np.load(f"{save_path}k_adj_4_{True}_E.npy")
    M_4_grid = np.load(f"{save_path}k_adj_4_{True}_M.npy")
    C_4_grid = np.load(f"{save_path}k_adj_4_{True}_C.npy")

    E_8_grid = np.load(f"{save_path}k_adj_8_{True}_E.npy")
    M_8_grid = np.load(f"{save_path}k_adj_8_{True}_M.npy")
    C_8_grid = np.load(f"{save_path}k_adj_8_{True}_C.npy")

    E_no_grid = list(np.load(f"{save_path}k_adj_{False}_E.npy"))
    M_no_grid = list(np.load(f"{save_path}k_adj_{False}_M.npy"))
    C_no_grid = np.load(f"{save_path}k_adj_{False}_C.npy")

    fig, ax = plt.subplots()
    ax.errorbar(max_ranges, E_4_grid[:, 0], yerr=E_4_grid[:, 1], fmt="b+-", label="Euclidean distance (meters), 4_adj")
    ax.errorbar(max_ranges, M_4_grid[:, 0], yerr=M_4_grid[:, 1], fmt="c+-", label="Mahalanobis distance, 4_adj")
    # Plotting success rate on the graph scaled by max error
    ax.plot(max_ranges, np.sum(C_4_grid, axis=1) / C_4_grid.shape[1], '--g', label="success rate 4_adj", alpha=0.5)

    # ax.errorbar(max_ranges, E_8_grid[:, 0], yerr=E_8_grid[:, 1], fmt="m+-", label="Euclidean distance (meters), 8_adj")
    # ax.errorbar(max_ranges, M_8_grid[:, 0], yerr=M_8_grid[:, 1], fmt="k+-", label="Mahalanobis distance, 8_adj")
    # ax.plot(max_ranges, np.sum(C_8_grid, axis=1) / C_8_grid.shape[1], '--m', label="success rate 8_adj", alpha=0.5)

    ax.errorbar(max_ranges, [E_no_grid[0]] * len(max_ranges), yerr=[E_no_grid[1]] * len(max_ranges), fmt="r+-",
                label="Euclidean distance (meters), no grid")
    ax.errorbar(max_ranges, [M_no_grid[0]] * len(max_ranges), yerr=[M_no_grid[1]] * len(max_ranges), fmt="y+-",
                label="Mahalanobis distance, no grid")

    ax.plot(max_ranges, np.mean(C_no_grid).repeat(len(max_ranges)), '--y', label="success rate no grid", alpha=0.5)

    ax.set(xlabel="distance (cells)", ylabel="measured error")
    ax.set_ylim(bottom=0)
    ax.grid()
    ax.legend()
    fig.savefig(f'{save_path}grid_effect.png')
    fig.show()


def optimal_weinberg_gain(config_origin: dict, save_path: str = '../results/', recompute=True):
    print("weinberg")
    config = config_origin.copy()
    if not osp.isdir(save_path):
        os.makedirs(save_path)
    if save_path[-1] != '/':
        save_path += '/'
    save_path = save_path + osp.basename(config['data_path'])

    # Gains for USER #1
    weinberg_gains = np.linspace(1.0, 1.16, 15, endpoint=True)
    if recompute:
        config["test_2"] = osp.basename(config['data_path'])  # stains the pngs/gifs with a unique trace
        seed = None
        np.random.seed(seed)
        random.seed(seed)

        for grid in [True, False]:
            config["build_grid"] = grid
            config["reload_cache"] = True

            Euclidean = []
            Mahalanobis = []
            Completed = []
            for weinberg_gain in weinberg_gains:

                config['weinberg_gain'] = weinberg_gain
                E = []
                M = []
                C = []
                # this hasn't been parallelised as the others, if you want to do so, just copy the same structure as above
                for id in range(10):
                    config['id'] = id
                    e, m, c, _, _, _ = main(**config, fig_info=config.copy())

                    C.append(c)
                    if c:
                        E.append(e)
                        M.append(m)

                E_mean = np.mean(E)
                E_std = np.std(E)

                M_mean = np.mean(M)
                M_std = np.std(M)

                Euclidean.append((E_mean, E_std))
                Mahalanobis.append((M_mean, M_std))
                Completed.append(C)

            np.save(f"{save_path}Euclidean_weinberg_{grid}.npy", np.array(Euclidean))
            np.save(f"{save_path}Mahalanobis_weinberg_{grid}.npy", np.array(Mahalanobis))
            np.save(f"{save_path}Completed_weinberg_{grid}.npy", np.array(Completed))
            np.save(f"{save_path}weinberg_gains_weinberg_{grid}.npy", np.array(weinberg_gains))

    Euclidean_grid = np.load(f"{save_path}Euclidean_weinberg_{True}.npy")
    Mahalanobis_grid = np.load(f"{save_path}Mahalanobis_weinberg_{True}.npy")
    Completed_g = np.load(f"{save_path}Completed_weinberg_{True}.npy")

    Euclidean_no_grid = np.load(f"{save_path}Euclidean_weinberg_{False}.npy")
    Mahalanobis_no_grid = np.load(f"{save_path}Mahalanobis_weinberg_{False}.npy")
    Completed_no_g = np.load(f"{save_path}Completed_weinberg_{False}.npy")

    weinberg_gains = np.load(f"{save_path}weinberg_gains_weinberg_{False}.npy")

    fig, ax = plt.subplots()

    ax.errorbar(weinberg_gains, Euclidean_grid[:, 0], yerr=Euclidean_grid[:, 1], fmt="b+-",
                label="average Euclidean distance (meters), grid", capsize=5)
    ax.errorbar(weinberg_gains, Mahalanobis_grid[:, 0], yerr=Mahalanobis_grid[:, 1], fmt="c+-",
                label="average Mahalanobis distance, grid", capsize=5)
    # Plotting success rate on the graph scaled by max error
    ax.plot(weinberg_gains,
            np.sum(Completed_g, axis=1) / Completed_g.shape[1],
            '--g', label="success rate grid (success/total)", alpha=0.5)

    ax.errorbar(weinberg_gains, Euclidean_no_grid[:, 0], yerr=Euclidean_no_grid[:, 1], fmt="r+-",
                label="average Euclidean distance (meters), no grid", capsize=5)
    ax.errorbar(weinberg_gains, Mahalanobis_no_grid[:, 0], yerr=Mahalanobis_no_grid[:, 1], fmt="y+-",
                label="average Mahalanobis distance, no grid", capsize=5)
    ax.plot(weinberg_gains,
            np.sum(Completed_no_g, axis=1) / Completed_no_g.shape[1],
            '--m', label="success rate no grid (success/total)", alpha=0.5)

    ax.legend()
    ax.set(xlabel="weinberg gain", ylabel="measured error")
    ax.set_ylim(bottom=0)
    ax.grid()
    fig.savefig(f"{save_path}weinberg.png")
    fig.show()


def angle_noise_effect(config_origin: dict, save_path: str = '../results/', recompute=True):
    print("angle_noise")
    config = config_origin.copy()

    if not osp.isdir(save_path):
        os.makedirs(save_path)

    if save_path[-1] != '/':
        save_path += '/'
    save_path = save_path + osp.basename(config['data_path'])

    angles_noises = np.linspace(0, np.pi / 30, 10, endpoint=True)  # mean = 0 ,  std of a gaussian

    if recompute:
        config["test_3"] = osp.basename(config['data_path'])
        seed = None
        np.random.seed(seed)
        random.seed(seed)

        for grid in [True, False]:
            config["build_grid"] = grid
            config["reload_cache"] = True

            Euclidean = []
            Mahalanobis = []
            Completed = []
            for angle_range in angles_noises:

                config['heading_range'] = [0, angle_range]
                E = []
                M = []
                C = []
                for id in range(10):
                    config['id'] = id
                    e, m, c, _, _, _ = main(**config, fig_info=config.copy())

                    C.append(c)
                    if c:
                        E.append(e)
                        M.append(m)

                E_mean = np.mean(E)
                E_std = np.std(E)

                M_mean = np.mean(M)
                M_std = np.std(M)

                Euclidean.append((E_mean, E_std))
                Mahalanobis.append((M_mean, M_std))
                Completed.append(C)

            np.save(f"{save_path}Euclidean_angle_{grid}.npy", np.array(Euclidean))
            np.save(f"{save_path}Mahalanobis_angle_{grid}.npy",
                    np.array(Mahalanobis))
            np.save(f"{save_path}Completed_angle_{grid}.npy", np.array(Completed))

    Euclidean_grid = np.load(f"{save_path}Euclidean_angle_{True}.npy")
    Mahalanobis_grid = np.load(f"{save_path}Mahalanobis_angle_{True}.npy")

    Euclidean_no_grid = np.load(f"{save_path}Euclidean_angle_{False}.npy")
    Mahalanobis_no_grid = np.load(f"{save_path}Mahalanobis_angle_{False}.npy")

    fig, ax = plt.subplots()
    ax.errorbar(angles_noises, Euclidean_grid[:, 0], yerr=Euclidean_grid[:, 1], fmt="b+-",
                label="average Euclidean distance (meters), grid")
    ax.errorbar(angles_noises, Mahalanobis_grid[:, 0], yerr=Mahalanobis_grid[:, 1], fmt="c+-",
                label="average Mahalanobis distance, grid")
    ax.errorbar(angles_noises, Euclidean_no_grid[:, 0], yerr=Euclidean_no_grid[:, 1], fmt="r+-",
                label="average Euclidean distance (meters), no grid")
    ax.errorbar(angles_noises, Mahalanobis_no_grid[:, 0], yerr=Mahalanobis_no_grid[:, 1], fmt="y+-",
                label="average Mahalanobis distance, no grid")

    ax.legend()
    ax.set(xlabel="angles_noises (std) in radians", ylabel="measured error")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid()
    fig.savefig(f"{save_path}angle_noise.png")
    fig.show()


def step_noise_effect(config_origin: dict, save_path: str = '../results/', recompute=True):
    print("step_noise")
    config = config_origin.copy()

    if not osp.isdir(save_path):
        os.makedirs(save_path)

    if save_path[-1] != '/':
        save_path += '/'
    save_path = save_path + osp.basename(config['data_path'])

    step_ranges = np.linspace(0, 0.30, 5, endpoint=True)  # in meters
    # step_ranges = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]) / 3  # in meters

    if recompute:
        config["test_4"] = osp.basename(config['data_path'])
        seed = None
        np.random.seed(seed)
        random.seed(seed)

        for grid in [True, False]:
            config["build_grid"] = grid
            config["reload_cache"] = True

            Euclidean = []
            Mahalanobis = []
            Completed = []
            for step_range in step_ranges:
                config['step_range'] = [0, step_range]
                E = []
                M = []
                C = []
                for id in range(10):
                    config['id'] = id
                    e, m, c, _, _, _ = main(**config, fig_info=config.copy())

                    C.append(c)
                    if c:
                        E.append(e)
                        M.append(m)

                E_mean = np.mean(E)
                E_std = np.std(E)

                M_mean = np.mean(M)
                M_std = np.std(M)

                Euclidean.append((E_mean, E_std))
                Mahalanobis.append((M_mean, M_std))
                Completed.append(C)

            np.save(f"{save_path}Euclidean_step_{grid}.npy", np.array(Euclidean))
            np.save(f"{save_path}Mahalanobis_step_{grid}.npy", np.array(Mahalanobis))
            np.save(f"{save_path}Completed_step_{grid}.npy", np.array(Completed))

    Euclidean_grid = np.load(f"{save_path}Euclidean_step_{True}.npy")[:5]
    Mahalanobis_grid = np.load(f"{save_path}Mahalanobis_step_{True}.npy")[:5]
    Completed_grid = np.load(f"{save_path}Completed_step_{True}.npy")[:5]

    Euclidean_no_grid = np.load(f"{save_path}Euclidean_step_{False}.npy")[:5]
    Mahalanobis_no_grid = np.load(f"{save_path}Mahalanobis_step_{False}.npy")[:5]
    Completed_no_grid = np.load(f"{save_path}Completed_step_{False}.npy")[:5]

    fig, ax = plt.subplots()
    ax.errorbar(step_ranges, Euclidean_grid[:, 0], yerr=Euclidean_grid[:, 1], fmt="b+-",
                label="average Euclidean distance (meters), grid", capsize=5, alpha=0.5)
    ax.errorbar(step_ranges, Mahalanobis_grid[:, 0], yerr=Mahalanobis_grid[:, 1], fmt="c+-",
                label="average Mahalanobis distance, grid", capsize=5, alpha=0.5)

    ax.plot(step_ranges, np.sum(Completed_grid, axis=1) / Completed_grid.shape[1],
            '--g', label="success rate grid (success/total)", alpha=0.5)

    ax.errorbar(step_ranges, Euclidean_no_grid[:, 0], yerr=Euclidean_no_grid[:, 1], fmt="r+-",
                label="average Euclidean distance (meters), no grid", capsize=5, alpha=0.5)
    ax.errorbar(step_ranges, Mahalanobis_no_grid[:, 0], yerr=Mahalanobis_no_grid[:, 1], fmt="y+-",
                label="average Mahalanobis distance, no grid", capsize=5, alpha=0.5)
    ax.plot(step_ranges, np.sum(Completed_no_grid, axis=1) / Completed_no_grid.shape[1],
            '--m', label="success rate no grid (success/total)", alpha=0.5)

    ax.legend()
    ax.set(xlabel="$\sigma_{step\ noise}$ in meters", ylabel="measured error")
    ax.set_ylim(bottom=0)
    ax.set_xlim(left=0)
    ax.grid()
    fig.savefig(f"{save_path}step_noise.png")
    fig.show()


def compare_integrity(config: dict, save_path: str = '../results/', recompute=True):
    config["make_fig"] = False
    config["make_gif"] = False

    config["min_wall_range"] = 11
    config["max_wall_range"] = 13
    config["cell_size"] = 5

    tmp_path = osp.dirname(osp.dirname(osp.abspath(__file__))) + f"/tmp/{osp.basename(config['data_path'])}"
    save_path = save_path + osp.basename(config['data_path'])

    if recompute:
        config["test_5"] = osp.basename(config['data_path'])
        config['build_grid'] = False

        errs_no_grid = []
        stds_no_grid = []
        for id in range(10):
            config['id'] = id
            _, _, _, e, s, _ = main(**config, fig_info=config.copy())
            errs_no_grid.append(e)
            stds_no_grid.append(s)
        # save the errs and stds
        errs_no_grid = np.array(errs_no_grid)
        stds_no_grid = np.array(stds_no_grid)
        print(errs_no_grid.shape)
        errs_no_grid = np.mean(errs_no_grid, axis=0)
        stds_no_grid = np.mean(stds_no_grid, axis=0)
        np.save(save_path + "_average_integrity_errs_False.npy", errs_no_grid)
        np.save(save_path + "_average_integrity_stds_False.npy", stds_no_grid)

        config['build_grid'] = True
        errs_grid = []
        stds_grid = []
        for id in range(10):
            config['id'] = id
            _, _, _, e, s, keypoints = main(**config, fig_info=config.copy())
            errs_grid.append(e)
            stds_grid.append(s)
        # save the errs and stds
        errs_grid = np.array(errs_grid)
        stds_grid = np.array(stds_grid)
        errs_grid = np.mean(errs_grid, axis=0)
        stds_grid = np.mean(stds_grid, axis=0)
        np.save(save_path + "_average_integrity_errs_True.npy", errs_grid)
        np.save(save_path + "_average_integrity_stds_True.npy", stds_grid)

    errors_grid = np.load(save_path + "_average_integrity_errs_True.npy", allow_pickle=True)
    stds_grid = np.load(save_path + "_average_integrity_stds_True.npy", allow_pickle=True)

    errors_no_grid = np.load(save_path + "_average_integrity_stds_False.npy", allow_pickle=True)
    stds_no_grid = np.load(save_path + "_average_integrity_stds_False.npy", allow_pickle=True)

    keypoints = np.load(tmp_path + "_keypoints.npy", allow_pickle=True)

    fig, ax = plot_integrity(errors_grid, stds_grid, len(keypoints), color="rmrm",
                             label=f'grid {config["min_wall_range"]} {config["max_wall_range"]}')
    fig, ax = plot_integrity(errors_no_grid, stds_no_grid, len(keypoints),
                             fig=fig, ax=ax, color="bcbc", label="no grid")

    # returns 12 lines from Artist
    lines = ax[0].get_lines() + ax[1].get_lines()
    # 3rd element is duplicated with 4th
    lines = [value for key, value in enumerate(lines, 1) if key % 3 != 0]
    fig.legend(handles=lines[:4], loc="center right")
    fig.savefig(f'{save_path}_integrity_grid_{config["min_wall_range"]}_{config["max_wall_range"]}.png')
    fig.show()


def optimal_grid_intergrity(config: dict, save_path: str = '../results/', recompute=True):
    config["make_fig"] = False
    config["make_gif"] = False
    save_path = save_path + osp.basename(config['data_path'])

    config["cell_size"] = 5

    if recompute:
        config["test_6"] = osp.basename(config['data_path'])
        # compute the no grid version to have a comparison
        config['build_grid'] = False

        pool = multiprocessing.Pool(N_THREAD)
        E, M, C, errs_no_grid, stds_no_grid, _ = zip(
            *pool.starmap(thread_luncher, zip(range(0, N_THREAD), [config] * N_THREAD)))

        errs_no_grid = np.array(errs_no_grid)[np.array(C), :]
        stds_no_grid = np.array(stds_no_grid)[np.array(C), :]
        errs_no_grid = np.mean(errs_no_grid, axis=0)
        stds_no_grid = np.mean(stds_no_grid, axis=0)
        np.save(save_path + "_average_integrity_errs_False.npy", errs_no_grid)
        np.save(save_path + "_average_integrity_stds_False.npy", stds_no_grid)

        # then, compute each grid param
        config['build_grid'] = True

        #  define the score map:
        min_wall_range_MAX = 25
        max_wall_range_MAX = 25
        assert max_wall_range_MAX > 0 and min_wall_range_MAX > 0  # you can turn off these
        assert max_wall_range_MAX >= min_wall_range_MAX
        assert type(max_wall_range_MAX) == int and type(min_wall_range_MAX) == int
        assert max_wall_range_MAX * config["cell_size"] <= 150, \
            "max_wall_range_MAX must be <= 150cm/cell_size, because corridors are restrained"

        score_map = np.zeros((min_wall_range_MAX, max_wall_range_MAX), dtype="float")

        for _min in range(min_wall_range_MAX):
            for _max in range(max_wall_range_MAX):
                if _min > _max:
                    continue
                config["min_wall_range"] = _min
                config["max_wall_range"] = _max

                pool = multiprocessing.Pool(N_THREAD)
                E, M, C, errs_grid, stds_grid, _ = zip(
                    *pool.starmap(thread_luncher, zip(range(0, N_THREAD), [config] * N_THREAD)))

                errs_grid = np.array(errs_grid)[np.array(C), :]
                stds_grid = np.array(stds_grid)[np.array(C), :]

                # this condition just says that the grid & no grid have the same keypoint crossing
                if errs_grid.shape[-1] == errs_no_grid.shape[-1]:
                    errs_grid = np.mean(errs_grid, axis=0)
                    stds_grid = np.mean(stds_grid, axis=0)

                    # The low and upper bounds need to be of opposite signs => their products must be negative
                    mask_grid = ((errs_no_grid + stds_no_grid) * (errs_no_grid - stds_no_grid)) < 0
                    mask_no_grid = ((errs_grid + stds_grid) * (errs_grid - stds_grid)) < 0
                    # the masks will be element-wise multiplied by the integral in order to reduce
                    # the score of the points that don't contain 0 between their bounds
                    stds_no_grid = np.where(mask_no_grid, stds_no_grid, 0)
                    stds_grid = np.where(mask_grid, stds_grid, 0)
                    # the difference between the integral of std_no_grid and std_grid
                    score_x = np.cumsum(stds_no_grid[0, :] - stds_grid[0, :])[-1]
                    score_y = np.cumsum(stds_no_grid[1, :] - stds_grid[1, :])[-1]
                    score = (score_x + score_y) / 2
                else:
                    score = 0
                score_map[_min, _max] = score

        np.save(save_path + "score_map.npy", score_map)
    # display a heatmap
    score_map = np.load(save_path + "score_map.npy")
    fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
    h, w = score_map.shape
    # ax.imshow(score_map, cmap='RdBu', origin="lower", interpolation=None)
    c = ax.pcolormesh(np.arange(0, w), np.arange(0, h), score_map, cmap='RdBu', label="this")
    # viridis and gnuplot are other cool coluor pallets I guess
    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.get_yaxis().labelpad = 30
    cbar.set_label('score $ = \int \sigma_{no\ grid} - \int \sigma_{grid}$', rotation=-90, size=15)
    ax.set(xlabel="max wall range (cells)", ylabel="min wall range (cells)")
    fig.show()


if __name__ == "__main__":
    # config load priority order from top to bottom:
    #   - config["key"] = value
    #   - read from config.json
    #   - use default values

    config = json.load(open("config.json"))
    config["data_path"] = '../data/imus/1/straight'
    # config["data_path"] = '../data/imus/1/straight_reversed'
    # config["data_path"] = '../data/imus/1/straight_in

    # 5 test
    # config["data_path"] = '../data/imus/1/weinberg_sequences/1f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/1r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/2f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/2r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/3f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/3r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/4f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/4r'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/5f'
    # config["data_path"] = '../data/imus/1/weinberg_sequences/5r'

    # filter parameters
    config["particles_count"] = 200
    config["weight_threshold"] = 0.00001

    # user parameter:
    config["weinberg_gain"] = 1.07
    config["acceleration_threshold"] = 0.06

    # plotting parameters:
    config["make_fig"] = False
    config["make_gif"] = False

    # the grid is cached to be reused without being recomputed, if you change any of the following, enable reloading:
    config["build_grid"] = False

    config["min_wall_range"] = 3
    config["max_wall_range"] = 6
    config["cell_size"] = 5
    config["distance_to_proba"] = None  # define a Callable as distance -> likelihood function, if None,
    # the filter will create a "sigmoid"
    config["k_adjacency"] = 4
    config["reload_cache"] = True  # THE MOST IMPORTANT, reload the cache due to modifications

    timer_start = datetime.datetime.now()
    # particle_count_vs_errors(config, recompute=True)
    # optimal_weinberg_gain(config, recompute=False)
    # grid_adjacency_effect(config, recompute=True)
    # angle_noise_effect(config, recompute=True)
    # step_noise_effect(config, recompute=False)
    compare_integrity(config, recompute=False)
    # optimal_grid_intergrity(config, recompute=True)
    # compare_to_ronin(config, recompute=True)

    print(datetime.datetime.now())
    try:  # me being fancy and sending myself an sms at the end
        import requests
        from urllib.parse import quote

        logins = json.load(open("logins.json"))

        message = quote(f'&msg=gene_fig computation finished!'
                        f' after:{datetime.datetime.now() - timer_start} seconds')
        requests.post(f'{logins["provider"]}' + f'user={logins["user"]}&pass={logins["pass"]}' + message)
    except FileNotFoundError:
        pass
