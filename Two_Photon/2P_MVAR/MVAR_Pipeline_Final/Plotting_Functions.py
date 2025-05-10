import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import MVAR_Utils_2P


def plot_line_graph(group_1_list, group_2_list, x_values, paired=True):

    group_1_list = np.array(group_1_list)
    group_2_list = np.array(group_2_list)

    group_1_mean, group_1_upper, group_1_lower = MVAR_Utils_2P.get_sem_and_bounds(group_1_list)
    group_2_mean, group_2_upper, group_2_lower = MVAR_Utils_2P.get_sem_and_bounds(group_2_list)

    # Load Frame Rate
    """
    period = float(1) / frame_rate
    x_values = list(range(start_window_frames, stop_window_frames))
    x_values = np.multiply(x_values, period)
    x_values = x_values[1:]
    """

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(x_values, group_1_mean, c='b')
    axis_1.plot(x_values, group_2_mean, c='g')
    axis_1.fill_between(x=x_values, y1=group_1_upper, y2=group_1_lower, alpha=0.4, color="blue")
    axis_1.fill_between(x=x_values, y1=group_2_upper, y2=group_2_lower, alpha=0.4, color="green")

    axis_1.axvline(0, c='k', linestyle='dashed')

    # Test sig
    """
    if paired == True:
        t_stat, p_values = stats.ttest_rel(group_1_list, group_2_list)
        rejected = np.where(p_values < 0.05, 1, 0)
        rejected = np.multiply(rejected, np.max([group_1_upper, group_2_upper]))
        axis_1.scatter(x_values, rejected)
        print("P", p_values)
    """
    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylabel("Lick CD (A.U.)")
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.show()
