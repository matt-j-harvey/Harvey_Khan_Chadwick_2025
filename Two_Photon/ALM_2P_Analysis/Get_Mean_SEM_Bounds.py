import numpy as np
from scipy import stats


def get_sem_and_bounds(data):
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    upper_bound = np.add(mean, sem)
    lower_bound = np.subtract(mean, sem)
    return mean, upper_bound, lower_bound
