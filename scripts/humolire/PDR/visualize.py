import os.path as osp
from typing import List

import imageio
import matplotlib.pyplot as plt
import numpy as np
from adjustText import adjust_text
from tqdm import tqdm

from .MapHandler import Wall
from .Particle import Particle


def plot_sequence(sequence, time=None,
                  y_axes_names=None, x_axis_name=None, style=None,
                  title="", fig=None, ax=None,
                  y_units="", x_unit="s",
                  **kwargs):
    """
    TODO: document
    """
    if time is None:
        time = np.arange(0, sequence.shape[0])

    if y_axes_names is not None:
        if len(sequence.shape) > 1:
            y_axes_names = y_axes_names
        else:
            y_axes_names = y_axes_names
    else:
        y_axes_names = ""
        if len(sequence.shape) > 1:
            y_axes_names = [""] * sequence.shape[1]

    if x_axis_name is None:
        x_axis_name = ""

    if style is None:
        style = "-"

    if isinstance(y_units, str):
        if len(sequence.shape) > 1:
            y_units = [f"({y_units})"] * sequence.shape[1]
        elif y_units != '':
            y_units = f"({y_units})"
    x_unit = f"({x_unit})"

    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
    if title == "" or title is None:
        fig.suptitle("")
    else:
        fig.suptitle(title)

    # check if shape is n,4 => quaternion
    if len(sequence.shape) > 1:
        sequence = sequence.T
        time = time.T

        n_rows = sequence.shape[0]
        for ((index, value), axis_name, unit) in zip(enumerate(sequence), y_axes_names, y_units):
            ax = plt.subplot(n_rows, 1, index + 1)
            ax.grid(b=True, which='both')
            ax.plot(time, value, style, **kwargs)
            ax.set_ylabel(axis_name + f" {unit}")
            ax.set_xlabel(f"{x_axis_name} {x_unit}")

    # if it is a 1D
    else:
        ax.grid(b=True, which='major')
        ax.plot(time, sequence, style, **kwargs)
        ax.set_xlabel(f"{x_axis_name} {x_unit}")
        ax.set_ylabel(f"{y_axes_names} {y_units}")

    return fig, ax


def plot_trajectory(x, y, title="", fig=None, ax=None, **kwargs):
    assert len(x) == len(y)
    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        assert x.ndim == y.ndim
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200, **kwargs)
    ax.plot(x, y, label="trajectory", **kwargs)
    ax.plot(x[0], y[0], 'oy', label="start point", **kwargs)
    ax.plot(x[-1], y[-1], 'xg', label="end point", **kwargs)
    ax.set(xlabel='X (meters)', ylabel='Y (meters)', title=title)
    ax.axis('equal')
    return fig, ax


def plot_particles(particles: List[Particle], title="", fig=None, ax=None,
                   alpha=0.95, add_legend=True, size=4, **kwargs):
    if len(particles) <= 0:
        print("Warning: len(particles) <= 0")
        return None, None
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
        ax.grid()
    X = [particle.position.x for particle in particles]
    Y = [particle.position.y for particle in particles]
    W = [particle.weight for particle in particles]
    if kwargs.get("color"):
        W = None
    ax.axis('equal')
    ax.scatter(X, Y, c=W, cmap='gnuplot', s=size, alpha=alpha, **kwargs)
    if add_legend:
        ax.legend(["trajectory", "start point", "end point"])
    ax.set(xlabel='X (meters)', ylabel='Y(meters)', title=title)
    return fig, ax


def plot_particles_center(center_positions: List, format: str = ".-b", title="",
                          add_legend=True, fig=None, ax=None,
                          **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200, **kwargs)
        ax.grid()
        ax.axis('equal')
    if len(center_positions) <= 0:
        print("Warning: len(center_positions) <= 0")
        return fig, ax

    X = [center.x for center in center_positions]
    Y = [center.y for center in center_positions]

    ax.plot(X, Y, format, alpha=0.5, **kwargs)
    ax.plot(center_positions[0].x, center_positions[0].y, 'oy')  # plot the starting center
    ax.plot(center_positions[-1].x, center_positions[-1].y, 'xg')  # plot the ending center
    ax.set(xlabel='meters', ylabel='meters', title=title)
    if add_legend:
        ax.legend(["trajectory", "start point", "end point"])
    return fig, ax


def plot_keypoints(keypoints: List, format='8c', alpha=0.5, fig=None, ax=None, figsize=(10, 8), dpi=200, title="",
                   label=None):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.axis('equal')
    ax.set(xlabel=' X (meters)', ylabel='Y (meters)', title=title)
    ax.legend()
    ax.spines['left'].set_position(('data', 0))
    ax.spines['bottom'].set_position(('data', 0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    plt.yticks(np.linspace(0, 36.6, 5))

    X = [point.x for point in keypoints]
    Y = [point.y for point in keypoints]
    ax.plot(X, Y, format, alpha=alpha, label=label)
    if format == '8c':
        texts = [ax.annotate(f'{idx}', (x, y), xytext=(x - 1, y - 1.5),
                             ha='left', va='center', color="purple")
                 for idx, (x, y) in enumerate(zip(X, Y))]
        adjust_text(texts, X, Y, ax=ax,  # arrowprops=dict(arrowstyle='->', color='red', lw=0.5),
                    force_points=10, only_move={'points': '', 'text': 'xy', 'objects': 'xy'})

    return fig, ax


def plot_walls(walls: List[Wall], title="", fig=None, ax=None, color='black', figsize=(10, 8), dpi=200, **kwargs):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    for wall in walls:
        ax.plot([wall.node0.x, wall.node1.x], [wall.node0.y, wall.node1.y], color, **kwargs)
    ax.axis('equal')
    ax.set(xlabel='meters', ylabel='meters', title=title)
    return fig, ax


def plot_grid(grid, fig=None, ax=None, figsize=(10, 8), dpi=200, vmin=0, vmax=1):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # The display should ensure to visualise the correct grid, if origin is not lower, the display is a matrix
    ax.imshow(grid, cmap=plt.cm.gray, origin="lower", vmin=vmin, vmax=vmax, interpolation=None)
    return fig, ax


def plot_ellipse(particles: List[Particle], fig=None, ax=None, figsize=(10, 8), dpi=200, **kwargs):
    # https://matplotlib.org/stable/gallery/statistics/confidence_ellipse.html#sphx-glr-gallery-statistics-confidence-ellipse-py

    X = np.array([particle.position.x for particle in particles])
    Y = np.array([particle.position.y for particle in particles])
    W = np.array([particle.weight for particle in particles])
    if ax is None or fig is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    confidence_ellipse(X, Y, ax, weights=W, edgecolor='red', **kwargs)
    return fig, ax


def confidence_ellipse(x, y, ax, weights=None, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.

    Parameters
    ----------
    x, y : array-like, shape (n, )
        Input data.
    weights : array-like, shape (n, )
        weight of the data
    ax : matplotlib.axes.Axes
        The axes object to draw the ellipse into.

    n_std : float
        The number of standard deviations to determine the ellipse's radii.

    **kwargs
        Forwarded to `~matplotlib.patches.Ellipse`

    Returns
    -------
    matplotlib.patches.Ellipse
    """
    from matplotlib.patches import Ellipse
    import matplotlib.transforms as transforms

    if x.size != y.size != weights.size:
        raise ValueError("x and y must be the same size")

    if np.sum(weights) == 0:
        weights = None
    cov = np.cov(x, y, aweights=weights)

    pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])

    # Using a special case to obtain the eigenvalues of this two-dimensional dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from the square root of the variance
    # and multiplying with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.average(x, weights=weights)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.average(y, weights=weights)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)


def plot_integrity(errors, stds, keypoints_count: int, std_factor=3.0,
                   fig=None, ax=None, color=None, alpha=0.5, label=None):
    assert len(errors) == 2 and len(stds) == 2, "computed errors and their std must be 2 arrays  each"
    assert 0 < std_factor <= 3, "you can only plot some sigma factor, max is 3 sigma"
    err_x, err_y = errors
    std_x, std_y = stds[0] * std_factor, stds[1] * std_factor

    if fig is None or ax is None:
        fig, (ax_x, ax_y) = plt.subplots(2, 1, figsize=(10, 8), dpi=200, sharex=True)
    else:
        (ax_x, ax_y) = ax

    if color is None or len(color) != 4:
        color = "rmrm"

    if type(label) is str:
        label = [label] * 4
    elif label is None:
        label = [" "] * 4

    keypoints_t = np.arange(0, keypoints_count)
    ax_x.plot(keypoints_t, err_x, '.-', color=color[0],
              label="error " + label[0], alpha=alpha)
    ax_x.plot(keypoints_t, err_x + std_x, '.-',
              keypoints_t, err_x - std_x, '.-',
              color=color[1], label=f"+/-{std_factor} $\sigma$ " + label[1], alpha=alpha)

    ax_y.plot(keypoints_t, err_y, '.-', color=color[2],
              label="error " + label[2], alpha=alpha)
    ax_y.plot(keypoints_t, err_y + std_y, '.-',
              keypoints_t, err_y - std_y, '.-', color=color[3],
              label=f"+/-{std_factor} $\sigma$" + label[3], alpha=alpha)

    ax_y.set(xlabel="keypoint index (time)", ylabel='Y (meters)')
    ax_x.set(ylabel='X (meters)')
    ax_y.grid(True)
    ax_x.grid(True)
    ax_x.set_xlim(left=0)

    if label == "    ":
        fig.legend()

    return fig, (ax_x, ax_y)


def plot_particles_evolution(surviving_particles_counts, optimal_particles_count, fig=None, ax=None, figsize=(10, 8)):
    if fig is None or ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    ax.plot(surviving_particles_counts, 'b+-', label="particles count evolution")
    ax.plot(optimal_particles_count, 'b+-', label="optimal particles count")


def render_gif(figures_sequence, path="./", filename="animated.gif", fps=2.5):
    #  https://ndres.me/post/matplotlib-animated-gifs-easily/

    images_sequence = []
    for fig in tqdm(figures_sequence, desc="creating the gif", unit="figure", total=len(figures_sequence)):
        fig.canvas.draw()  # draw the canvas, cache the renderer

        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        images_sequence.append(image)

    if path[-1] != '/':
        path += '/'
    save_path = osp.join(path + filename)
    print(f"saving the gif under:{osp.abspath(save_path)} at {fps} fps")
    imageio.mimsave(save_path, images_sequence, fps=fps)
