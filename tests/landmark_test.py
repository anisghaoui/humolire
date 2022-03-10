import json
from argparse import Namespace

from humolire.PDR.MapHandler import MapHandler
from humolire.PDR.visualize import plot_walls, plot_keypoints


def main():
    mm = MapHandler("../data/maps/map_data.json")
    fig, ax = plot_walls(mm.walls)
    ax.set_xlim(mm.x_range)
    ax.set_ylim(mm.y_range)
    keypoints_d = json.load(open("test_sequences/keypoints.json"))
    keypoints = [Namespace(**keypoints_d[str(i)]) for i in range(keypoints_d["count"])]
    fig, ax = plot_keypoints(keypoints, fig=fig, ax=ax, label="landmarks")

    fig.show()


if __name__ == "__main__":
    main()
