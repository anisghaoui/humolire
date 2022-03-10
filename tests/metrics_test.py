from argparse import Namespace

import numpy as np

from humolire.PDR.common import mahalanobis_error


def main():
    N = 10
    X = np.arange(0, N)
    Y = np.arange(0, N) ** 2

    pos = Namespace(**{'x': 4.5, 'y': 27.0})
    print(mahalanobis_error(pos, X, Y))


if __name__ == "__main__":
    main()
