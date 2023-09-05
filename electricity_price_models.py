"""
author: Florian Krach
"""

import numpy as np


def sample_exponential_price_jumps_paths(
        lam, T, initial_price=1., nb_samples=1000):
    """
    Sample electricity price paths that have at each time step a jump (of
    absolute value) sampled from an exponential distribution with rate lam.
    """
    paths = np.zeros((nb_samples, T))
    paths[:, 0] = initial_price
    paths[:, 1:] = np.random.exponential(1/lam, size=(nb_samples, T-1))
    paths = np.cumsum(paths, axis=1)
    return paths


def sample_exponential_percentage_price_jumps_paths(
        mean_percentage, T, initial_price=1., nb_samples=1000,
        factors=None,
):
    """
    Sample electricity price paths that have at each time step a percentage jump
    sampled from an exponential distribution with rate lam.

    mean_percentage: float in (0,1), the mean percentage jump per time step
    """
    paths = np.zeros((nb_samples, T))
    paths[:, 0] = initial_price
    percentage_increases = np.random.exponential(
        scale=mean_percentage, size=(nb_samples, T-1))+1.
    cum_perc_inc = np.cumprod(percentage_increases, axis=1)
    paths[:, 1:] = cum_perc_inc*initial_price
    if factors is not None:
        paths = paths * factors.reshape(1, -1).repeat(nb_samples, axis=0)
    return paths





if __name__ == '__main__':
    pass
