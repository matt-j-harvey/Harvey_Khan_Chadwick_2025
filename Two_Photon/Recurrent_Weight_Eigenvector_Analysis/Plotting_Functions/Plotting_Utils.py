import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import Plotting_Functions.Data_Loading_Functions as Data_Loading_Functions




def get_mean_and_bounds(data):
    print("data", np.shape(data))
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)
    return data_mean, lower_bound, upper_bound



def plot_line_graph(wt_data, nx_data, plot_title, x_value_time=True, ylim=None, set_x_values=None):

    # Get Bounds
    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_data)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_data)

    # Test Significance
    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    max_value = np.max([wt_upper_bound, nx_upper_bound]) * 1.1
    sig_values = np.multiply(binary_sig, max_value)
    print("p values", p_values)


    # Get X Values
    x_values = list(range(len(wt_mean)))
    if x_value_time == True:
        x_values = np.multiply(x_values, (1000 / 6.37))

    elif set_x_values is not None:
        x_values = set_x_values

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    # Plot Data
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    # Add Signficance Markers
    axis_1.scatter(x_values, sig_values, alpha=binary_sig, c='k')

    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    if ylim != None:
        axis_1.set_ylim(ylim)

    axis_1.set_title(plot_title)
    plt.show()



def test_significance(wt_data, nx_data, wt_upper_bound, nx_upper_bound):
    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    max_value = np.max([wt_upper_bound, nx_upper_bound]) * 1.1
    sig_values = np.multiply(binary_sig, max_value)
    return sig_values, binary_sig



def plot_dual_line_graph(wt_data_1, nx_data_1, wt_data_2, nx_data_2, plot_titles, x_value_time=True, ylim=None):

    # Get Bounds
    wt_mean_1, wt_lower_bound_1, wt_upper_bound_1 = get_mean_and_bounds(wt_data_1)
    nx_mean_1, nx_lower_bound_1, nx_upper_bound_1 = get_mean_and_bounds(nx_data_1)
    wt_mean_2, wt_lower_bound_2, wt_upper_bound_2 = get_mean_and_bounds(wt_data_2)
    nx_mean_2, nx_lower_bound_2, nx_upper_bound_2 = get_mean_and_bounds(nx_data_2)

    # Test Significance
    sig_values_1, binary_sig_1 = test_significance(wt_data_1, nx_data_1, wt_upper_bound_1, nx_upper_bound_1)
    sig_values_2, binary_sig_2 = test_significance(wt_data_2, nx_data_2, wt_upper_bound_2, nx_upper_bound_2)

    # Get X Values
    x_values = list(range(len(wt_mean_1)))
    if x_value_time == True:
        x_values = np.multiply(x_values, (1000 / 6.37))

    # Create Figure
    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1, 2, 1)
    axis_2 = figure_1.add_subplot(1, 2, 2)

    # Plot Data
    axis_1.plot(x_values, wt_mean_1, c='b')
    axis_1.plot(x_values, nx_mean_1, c='m')
    axis_1.fill_between(x_values, wt_lower_bound_1, wt_upper_bound_1, alpha=0.2, color='b')
    axis_1.fill_between(x_values, nx_lower_bound_1, nx_upper_bound_1, alpha=0.2, color='m')

    axis_2.plot(x_values, wt_mean_2, c='b')
    axis_2.plot(x_values, nx_mean_2, c='m')
    axis_2.fill_between(x_values, wt_lower_bound_2, wt_upper_bound_2, alpha=0.2, color='b')
    axis_2.fill_between(x_values, nx_lower_bound_2, nx_upper_bound_2, alpha=0.2, color='m')

    # Add Significance Markers
    axis_1.scatter(x_values, sig_values_1, alpha=binary_sig_1, c='k')
    axis_2.scatter(x_values, sig_values_2, alpha=binary_sig_2, c='k')

    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_2.spines[['right', 'top']].set_visible(False)

    # Set Bounds
    if ylim == None:
        max_value = np.max([wt_upper_bound_1, wt_upper_bound_2, nx_upper_bound_1, nx_upper_bound_2])
        min_value = np.min([wt_lower_bound_1, wt_lower_bound_2, nx_lower_bound_1, nx_lower_bound_2])
        max_value = max_value + (0.2 * np.abs(max_value))
        min_value = min_value - (0.2 * np.abs(min_value))
        axis_1.set_ylim([min_value, max_value])
        axis_2.set_ylim([min_value, max_value])

    else:
        axis_1.set_ylim(ylim)
        axis_2.set_ylim(ylim)

    axis_1.set_title(plot_titles[0])
    axis_2.set_title(plot_titles[1])

    plt.show()







def plot_scatter_graph(wt_data, nx_data, plot_title, ylim=None):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    wt_xvalues = np.zeros(len(wt_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_data))
    nx_xvalues = np.ones(len(nx_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_data))

    axis_1.scatter(wt_xvalues, wt_data, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_data, c='m', alpha=0.4)

    t_stat, p_value = stats.ttest_ind(wt_data, nx_data)
    print("t_stat", t_stat)
    print("p_value", p_value)

    axis_1.set_xlim([-0.5, 1.5])
    axis_1.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    axis_1.spines[['right', 'top']].set_visible(False)

    if ylim != None:
        axis_1.set_ylim(ylim)

    axis_1.set_title(plot_title + "\n p = " + str(np.around(p_value, 3)))

    plt.show()




def plot_dual_scatter_graph(wt_data_1, nx_data_1, wt_data_2, nx_data_2, plot_titles, ylim=None):

    # Create Figure
    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1, 2, 1)
    axis_2 = figure_1.add_subplot(1, 2, 2)

    # Get X Values
    wt_xvalues = np.zeros(len(wt_data_1)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_data_1))
    nx_xvalues = np.ones(len(nx_data_1)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_data_1))

    # Test Significance
    t_stat_1, p_value_1 = stats.ttest_ind(wt_data_1, nx_data_1)
    t_stat_2, p_value_2 = stats.ttest_ind(wt_data_2, nx_data_2)

    # Plot Data
    print("wt_xvalues", np.shape(wt_xvalues), "wt_data_1", np.shape(wt_data_1))
    axis_1.scatter(wt_xvalues, wt_data_1, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_data_1, c='m', alpha=0.4)
    axis_2.scatter(wt_xvalues, wt_data_2, c='b', alpha=0.4)
    axis_2.scatter(nx_xvalues, nx_data_2, c='m', alpha=0.4)

    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_2.spines[['right', 'top']].set_visible(False)

    # Set Bounds
    max_value = np.max(np.concatenate([wt_data_1, wt_data_2, nx_data_1, nx_data_2]))
    min_value = np.min(np.concatenate([wt_data_1, wt_data_2, nx_data_1, nx_data_2]))
    max_value = max_value + (0.2 * np.abs(max_value))
    min_value = min_value - (0.2 * np.abs(min_value))
    axis_1.set_ylim([min_value, max_value])
    axis_2.set_ylim([min_value, max_value])

    axis_1.set_xlim([-0.5, 1.5])
    axis_2.set_xlim([-0.5, 1.5])

    if ylim != None:
        axis_1.set_ylim(ylim)
        axis_2.set_ylim(ylim)

    axis_1.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    axis_2.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])

    axis_1.set_title(plot_titles[0] + "\n p = " + str(np.around(p_value_1, 3)))
    axis_2.set_title(plot_titles[1] + "\n p = " + str(np.around(p_value_2, 3)))

    plt.show()


