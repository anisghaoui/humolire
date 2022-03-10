import math

from humolire.PDR.MapHandler import MapHandler
from humolire.PDR.PDR import perform_pdr
from humolire.PDR.ParticleFilter import ParticleFilter
from humolire.PDR.dataloaders import load_ronin_txts
from humolire.PDR.visualize import plot_walls, plot_particles


def no_starting_point_test(path):
    frequency = 400.0  # hz
    map_file = '../data/maps/map_data.json'
    # load and init
    time, acce, gyro = load_ronin_txts(path)
    map_matcher = MapHandler(map_file)
    n_particles = 100

    print(f"\n\n********{__name__} with {path} :"
          f"\n {frequency = }"
          f"\n {n_particles = }"
          f"\n {map_file =} "
          f"\n\n")

    pf = ParticleFilter(n_particles, step_range=(-0.1, +0.1), heading_range=(-math.pi / 10, +math.pi / 10),
                        map_handler=map_matcher)

    steps_length, steps_heading = perform_pdr(acce, gyro, frequency)

    fig, ax = plot_walls(map_matcher.walls)
    plot_particles(pf.particles, fig=fig, ax=ax, alpha=0.5)
    fig.show()

    # store particles for the gif
    particles_life = []
    for idx, (step_length, step_heading) in enumerate(zip(steps_length, steps_heading)):
        pf.update(step_length, step_heading)

        fig_t, ax_t = plot_walls(map_matcher.walls)
        plot_particles(pf.particles, fig=fig_t, ax=ax_t, alpha=0.5)
        particles_life.append(fig_t)

    # render_gif(particles_life, fps=2)


if __name__ == "__main__":
    no_starting_point_test("data/imus/1/long_no_pi")
