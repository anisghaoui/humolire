import matplotlib
import matplotlib.pyplot as plt
import numpy
import numpy as np

from humolire.PDR.MapHandler import MapHandler, MapGrid
from humolire.PDR.visualize import plot_grid, plot_sequence, plot_walls

matplotlib.rcParams["figure.autolayout"] = True


def test_map():  # wrote a small jsno to see
    mm = MapHandler("test_sequences/small_test.json")
    fig, ax = plot_walls(mm.walls)
    ax.set_xlim(mm.x_range)
    ax.set_ylim(mm.y_range)
    plt.show()


def test_grid():  # just to visualise the grid itself
    mm = MapHandler("../data/maps/map_data.json")
    fig, ax = plot_walls(mm.walls)
    ax.set(xlabel="x in meters", ylabel="y in meters")

    ax.spines['left'].set_position(('outward', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks(np.linspace(*mm.y_range, 5))
    ax.set_xlim(mm.x_range)
    ax.set_ylim(mm.y_range)

    cell_size = 5.0
    m_grid = MapGrid(map_height=mm.height, map_width=mm.width, cell_size=cell_size, walls=mm.walls, min_wall_range=5,
                     max_wall_range=20, reload_cache=True)
    fig, ax = plot_grid(m_grid.grid)
    ax.set(xlabel=f"x in cells (cell = {cell_size}x{cell_size}cm²)", ylabel="y in cells")


def test_likeihood_func():
    mm = MapHandler("../data/maps/map_data.json", reload_cache=True)  # won(t use it
    m_grid = MapGrid(map_height=mm.height, map_width=mm.width, cell_size=5.0, walls=mm.walls)

    # plotting the distance to likelihood function
    likelihood_f = m_grid.distance_to_likelihood
    cell_count = numpy.arange(0, 11, 1)
    fig, ax = None, None
    labels = []
    for _min, _max in [(0, 2), (4, 4), (6, 9), (2, 8)]:
        probabilities = np.array([likelihood_f(c,
                                               min_wall_range=_min,
                                               max_wall_range=_max)
                                  for c in cell_count])
        fig, ax = plot_sequence(probabilities, cell_count, linewidth=4, alpha=0.6,
                                y_axes_names="likelihood", title="", x_axis_name="distance",
                                x_unit="cells", fig=fig, ax=ax)
        labels.append(f"with min={_min}, max={_max}")
    fig.legend(labels, loc="center right")


if __name__ == "__main__":
    # test_map()
    test_grid()
    # test_likeihood_func()
    plt.show()
