import random
from typing import Any, List

import numpy as np


# Comparison of Resampling Schemes for Particle Filtering
# Randal Douc
# 2006
def multinomial(weights: np.ndarray, U: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    weights : np.ndarray[float],
        weights of particles
    U : np.ndarray[float],
        ordered random values between 0 and 1 (both included)

    Returns
    -------
    new_particles_indices :  np.ndarray[int]
        indices of the new particles. subscript this to obtain your new particles
    """
    weights = weights / np.sum(weights)
    cum_sum = np.hstack([[0], np.cumsum(weights, axis=-1)])
    # find if cum_i <= u_i <= cum_i+1
    new_weights = np.where(np.logical_and(cum_sum[:-1] < U, U <= cum_sum[1:]), weights, 0)

    # from https://stackoverflow.com/a/30489294/13642668
    prev = np.arange(len(new_weights))
    prev[new_weights == 0] = 0
    new_particles_indices = np.maximum.accumulate(prev)

    first_non_zero = np.nonzero(new_weights)[0][0]
    new_particles_indices[:first_non_zero] = first_non_zero

    return new_particles_indices


def stratified(weights: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    weights : np.ndarray[float],
        weights of particles

    Returns
    -------
    new_particles_indices : np.ndarray[int]
        indices of the new particles. subscript this to obtain your new particles

    """
    N = len(weights)
    U = np.arange(N) / N + np.random.uniform(0, 1 / N, size=N)
    return multinomial(weights, U)


def systematic(weights: np.ndarray) -> np.ndarray:
    """

    Parameters
    ----------
    weights : np.ndarray[float],
        weights of particles

    Returns
    -------
    new_particles_indices : np.ndarray[int]
        indices of the new particles. subscript this to obtain your new particles

    """
    U = (np.arange(len(weights)) + 1 - np.random.uniform(0, 1)) / len(weights)
    return multinomial(weights, U)


def residual(weights: np.ndarray, rng=None) -> np.ndarray:
    """

    Parameters
    ----------
    weights : np.ndarray[float],
        weights of particles
    rng : RNG,ndarray
        a random number generator object that contains a `uniform()` method to be called.

    Returns
    -------
    new_particles_indices : np.ndarray[int]
        indices of the new particles. subscript this to obtain your new particles.
        Preferably: rng = np.random.default_rng(seed)

    """

    # assert False , "There is still a bug here 2021-04-20"
    if rng is None:
        rng = np.random.default_rng()
    N = len(weights)
    N_w = (N * weights).astype("int")

    # generated U
    N_prime = int((N * weights - N_w).sum().round())
    U = np.array(sorted(rng.uniform(0, 1, size=N_prime)))

    residues_index = ~N_w.astype("bool")
    to_be_resampled = rng.choice(weights[residues_index], N_prime, replace=False)  # pick N_prime
    # perform multinomial on residues and fuse them with the other particles
    valid_indices = np.repeat(weights.nonzero()[0], N_w)
    new_particles_index = np.hstack([valid_indices, multinomial(to_be_resampled, U)])

    return new_particles_index


def metropolis(weights, B=10):
    # TODO: what value should B have?
    """
        https://doi.org/10.1063/1.1699114

    Parameters
    ----------
    weights :
    B :

    Returns
    -------

    """
    result = []
    for i in range(len(weights)):
        k = i
        for n in range(B):
            u = random.uniform(0, 1)
            j = int(np.floor(random.uniform(0, len(weights))))
            if u <= weights[j] / weights[k]:
                k = j
        result.append(k)
    return result


def rejection(weights):
    """
        https://doi.org/10.1080/10618600.2015.1062015


    Parameters
    ----------
    weights :

    Returns
    -------

    """
    max_weight = max(weights)
    result = []
    for i in range(len(weights)):
        j = i
        u = random.uniform(0, 1)
        while u > weights[j] / max_weight:
            j = int(np.floor(random.uniform(0, len(weights))))
            u = random.uniform(0, 1)
        result.append(j)
    return result


def kong_criterion(weights: [np.ndarray, List]) -> float:
    """
        https://doi.org/10.1080/01621459.1994.10476469

    Parameters
    ----------
    weights : np.ndarray

    Returns
    -------
        effective particles count : float
    """
    N_eff = 1 / np.sum(np.square(weights))
    return N_eff


def pham_criterion(weights: [np.ndarray, List]) -> float:
    """
    https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.640.2136&rep=rep1&type=pdf
    Parameters
    ----------
    weights : np.ndarray

    Returns
    -------
        entropy : float
    """
    entropy = np.log(len(weights)) + np.sum(weights * np.log(weights))
    return entropy
