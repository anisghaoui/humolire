import math
import random
from copy import deepcopy
from math import cos, sin
from random import uniform

import matplotlib.pyplot as plt

from humolire.PDR.ParticleFilter import ParticleFilter
from humolire.PDR.visualize import plot_particles


# DISABLED
# def uniform_test():
#     N = 10000
#     radius_range =(-0.3,0.3)
#     heading_range = (-math.pi / 10, +math.pi / 10)
#     pf = ParticleFilter(N, step_range=radius_range, heading_range=heading_range, distribution_law_fn=random.uniform,
#                         build_grid=False)
#     fig, ax = plot_particles(pf.particles, title=f"{N} generated particles", add_legend=False)
#
#     pf.update(step_length= 1,step_heading = 1.5) # 1m 90 °
#     after_particles = pf.particles
#     plot_particles(after_particles,fig=fig, ax=ax, format='.b',title = "Update process", add_legend=False)
#     fig.legend(["initial particles","particles after update"])

def gaussian_test():
    N = 2000  # in meters
    step_range = (0, 0.1)
    heading_range = (0, +math.pi / 30)
    initial_radius_range = [0, 0.05]
    initial_heading_range = [0, math.pi / 10]

    pf = ParticleFilter(N, step_range=step_range, heading_range=heading_range, distribution_law_fn=random.gauss,
                        initial_radius_range=initial_radius_range, initial_heading_range=initial_heading_range,
                        build_grid=False)
    before_particles = deepcopy(pf.particles)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.grid()
    fig, ax = plot_particles(pf.particles, title=f"{N} generated particles",
                             add_legend=False, color="red", size=20, fig=fig, ax=ax)

    pf.update(step_length=0.75, step_heading=1.5)  # 1m and 90°

    after_particles = pf.particles
    # plot_particles(after_particles, fig=fig, ax=ax, title="Update process", add_legend=False, color="blue", size=20)
    # fig.legend(["initial particles", "particles after update"])
    # fig.legend(["initial particles", "particles after update"])

    arrow_scale = 0.02
    for b_p in before_particles:
        x = b_p.position.x
        y = b_p.position.y
        dx = arrow_scale * cos(b_p.heading)
        dy = arrow_scale * sin(b_p.heading)
        plt.arrow(x, y, dx, dy, head_width=0.001,
                  shape="full", color="orange", alpha=0.7, length_includes_head=True)


def update_test():
    N = 5
    radius_range = (-0.3, 0.3)
    heading_range = (-math.pi / 10, +math.pi / 10)
    pf = ParticleFilter(N, step_range=radius_range, heading_range=heading_range,
                        initial_radius_range=[0, 0.1], initial_heading_range=[0, math.pi / 10])
    before_particles = deepcopy(pf.particles)

    fig, ax = plot_particles(before_particles, add_legend=False, color="red", size=20)

    pf.update(step_length=0.75, step_heading=1.5)  # 1m 90 °
    after_particles = deepcopy(pf.particles)
    plot_particles(after_particles, fig=fig, ax=ax, title="Update process", add_legend=False, color="blue", size=20)
    fig.legend(["before update", "after update"])

    for b_p, a_p in zip(before_particles, after_particles):
        x = b_p.position.x
        y = b_p.position.y
        dx = a_p.position.x - x
        dy = a_p.position.y - y
        plt.arrow(x, y, dx, dy, head_width=0.02, shape="full", color="green", length_includes_head=True)


def random_update_test():
    N = 5
    radius_range = (-0.15, 0.15)
    heading_range = (-math.pi / 10, +math.pi / 10)
    pf = ParticleFilter(N, step_range=radius_range, heading_range=heading_range,
                        initial_radius_range=[0, 0.1], initial_heading_range=[0, math.pi / 10])
    before_particles = deepcopy(pf.particles)
    deterministic_particles = deepcopy(pf.particles)
    noised_particles = deepcopy(pf.particles)

    step_heading = 1.5  # 90°
    step_length = 1  # 1m
    for dert_particle, noised_particle in zip(deterministic_particles, noised_particles):
        dert_particle.heading += step_heading
        dx = cos(dert_particle.heading) * step_length
        dy = sin(dert_particle.heading) * step_length
        dert_particle.position.x += dx
        dert_particle.position.y += dy

        noised_particle.heading = dert_particle.heading + uniform(*heading_range)
        radius = uniform(*radius_range)  # just to easier visualisation, I am usign uniform
        dx = cos(noised_particle.heading) * (step_length + radius)
        dy = sin(noised_particle.heading) * (step_length + radius)
        noised_particle.position.x += dx
        noised_particle.position.y += dy

    fig, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.grid()
    fig, ax = plot_particles(before_particles, title="Update process detailed",
                             add_legend=False, color="red", size=10, fig=fig, ax=ax)
    fig, ax = plot_particles(deterministic_particles, fig=fig, ax=ax, add_legend=False, color="blue", size=10)
    plot_particles(noised_particles, fig=fig, ax=ax, add_legend=False, color="green", size=20)
    fig.legend(["Before update", "Deterministic particles update", "Random noise added"])

    for b_p, d_p, a_p in zip(before_particles, deterministic_particles, noised_particles):
        x = b_p.position.x
        y = b_p.position.y
        dx = d_p.position.x - x
        dy = d_p.position.y - y
        plt.arrow(x, y, dx, dy,
                  head_width=0.02, shape="full", color="brown", length_includes_head=True)

        x = b_p.position.x
        y = b_p.position.y
        dx = a_p.position.x - x
        dy = a_p.position.y - y
        plt.arrow(x, y, dx, dy,
                  head_width=0.02, shape="full", color="cyan", length_includes_head=True)


if __name__ == "__main__":
    random.seed(1)
    gaussian_test()

    update_test()
    random_update_test()
    plt.show()
