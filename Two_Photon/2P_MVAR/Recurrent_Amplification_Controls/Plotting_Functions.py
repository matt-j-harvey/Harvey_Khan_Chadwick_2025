import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def sort_raster(raster, sorting_window_start, sorting_window_stop):

    # Get Mean Response in Sorting Window
    response = raster[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)

    # Get Sorted Indicies
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    # Sort Rasters
    sorted_raster = raster[:, sorted_indicies]

    return sorted_raster


def view_psth(mean_activity):

    # Plot Raster
    magnitude = np.percentile(np.abs(mean_activity), q=99.5)

    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(np.transpose(mean_activity), vmin=-magnitude, vmax=magnitude, cmap='bwr')
    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylabel("Neurons")
    forceAspect(axis_1)

    plt.show()


def get_means_and_bounds(data_list):

    print("data_list", np.shape(data_list))
    data_list = np.array(data_list)
    print("data_list", np.shape(data_list))

    data_mean = np.mean(data_list, axis=0)

    data_sem = stats.sem(data_list, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound





def test_signficiance(vis_context_vis_1_projection_list, vis_context_vis_2_projection_list, odour_context_vis_1_projection_list, odour_context_vis_2_projection_list):

    vis_diff_list = []
    odour_diff_list = []

    n_mice = len(vis_context_vis_1_projection_list)

    for mouse in range(n_mice):
        vis_diff = np.subtract(vis_context_vis_1_projection_list[mouse], vis_context_vis_2_projection_list[mouse])
        odr_diff = np.subtract(odour_context_vis_1_projection_list[mouse], odour_context_vis_2_projection_list[mouse])

        vis_diff_list.append(vis_diff)
        odour_diff_list.append(odr_diff)

    t_stats, p_values = stats.ttest_rel(vis_diff_list, odour_diff_list)
    print("t_stats", t_stats)
    print("p_values", p_values)


def plot_single_session(diagonal_projections, full_projections):

    figure_1 = plt.figure()
    diagonal_axis = figure_1.add_subplot(1, 2, 1)
    full_axis = figure_1.add_subplot(1, 2, 2)

    diagonal_axis.plot(diagonal_projections[0], c='b')
    diagonal_axis.plot(diagonal_projections[1], c='r')
    diagonal_axis.plot(diagonal_projections[2], c='g')
    diagonal_axis.plot(diagonal_projections[3], c='m')

    full_axis.plot(full_projections[0], c='b')
    full_axis.plot(full_projections[1], c='r')
    full_axis.plot(full_projections[2], c='g')
    full_axis.plot(full_projections[3], c='m')

    plt.show()


def plot_scatter_graph(diagonal_projection_group_list, full_projection_group_list):

    # Get Mean Time Window
    mean_full = np.mean(full_projection_group_list, axis=2)
    mean_diagonal = np.mean(diagonal_projection_group_list, axis=2)

    # Compare Vis 1
    vis_1_full = mean_full[:, 0]
    vis_1_diag = mean_diagonal[:, 0]


    t_stat, p_value = stats.ttest_rel(vis_1_full, vis_1_diag)
    print("t_stat", t_stat, "p_value", p_value)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.scatter(x=np.zeros(len(vis_1_diag)), y=vis_1_diag)
    axis_1.scatter(x=np.ones(len(vis_1_full)), y=vis_1_full)
    plt.show()

    print("full_projection_group_list", np.shape(full_projection_group_list))


def plot_scatter_graph_diff(diagonal_projection_group_list, full_projection_group_list):
    # Get Mean Time Window
    mean_full = np.mean(full_projection_group_list, axis=2)
    mean_diagonal = np.mean(diagonal_projection_group_list, axis=2)

    # Compare Vis 1
    vis_1_full = mean_full[:, 0]
    vis_2_full = mean_full[:, 1]
    odr_1_full = mean_full[:, 2]
    odr_2_full = mean_full[:, 3]

    vis_1_diag = mean_diagonal[:, 0]
    vis_2_diag = mean_diagonal[:, 1]
    odr_1_diag = mean_diagonal[:, 2]
    odr_2_diag = mean_diagonal[:, 3]

    # Get Modulation
    vis_diff_full = np.subtract(vis_1_full, vis_2_full)
    odr_diff_full = np.subtract(odr_1_full, odr_2_full)
    full_modulation = np.subtract(vis_diff_full, odr_diff_full)

    vis_diff_diag = np.subtract(vis_1_diag, vis_2_diag)
    odr_diff_diag = np.subtract(odr_1_diag, odr_2_diag)
    diag_modulation = np.subtract(vis_diff_diag, odr_diff_diag)

    t_stat, p_value = stats.ttest_rel(full_modulation, diag_modulation)
    print("t_stat", t_stat, "p_value", p_value)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.scatter(x=np.zeros(len(diag_modulation)), y=diag_modulation)
    axis_1.scatter(x=np.ones(len(full_modulation)), y=full_modulation)
    plt.show()

    print("full_projection_group_list", np.shape(full_projection_group_list))


def plot_stimuli_amplification(diagonal_projection_group_list, full_projection_group_list, ylim=[-0.7, 2.3]):

    # Get Means and SEMs
    full_vis_context_vis_1_mean, full_vis_context_vis_1_lower_bound, full_vis_context_vis_1_upper_bound = get_means_and_bounds(full_projection_group_list[:, 0])
    full_vis_context_vis_2_mean, full_vis_context_vis_2_lower_bound, full_vis_context_vis_2_upper_bound = get_means_and_bounds(full_projection_group_list[:, 1])
    full_odour_context_vis_1_mean, full_odour_context_vis_1_lower_bound, full_odour_context_vis_1_upper_bound = get_means_and_bounds(full_projection_group_list[:, 2])
    full_odour_context_vis_2_mean, full_odour_context_vis_2_lower_bound, full_odour_context_vis_2_upper_bound = get_means_and_bounds(full_projection_group_list[:, 3])

    diagonal_vis_context_vis_1_mean, diagonal_vis_context_vis_1_lower_bound, diagonal_vis_context_vis_1_upper_bound = get_means_and_bounds(diagonal_projection_group_list[:, 0])
    diagonal_vis_context_vis_2_mean, diagonal_vis_context_vis_2_lower_bound, diagonal_vis_context_vis_2_upper_bound = get_means_and_bounds(diagonal_projection_group_list[:, 1])
    diagonal_odour_context_vis_1_mean, diagonal_odour_context_vis_1_lower_bound, diagonal_odour_context_vis_1_upper_bound = get_means_and_bounds(diagonal_projection_group_list[:, 2])
    diagonal_odour_context_vis_2_mean, diagonal_odour_context_vis_2_lower_bound, diagonal_odour_context_vis_2_upper_bound = get_means_and_bounds(diagonal_projection_group_list[:, 3])

    # Create Figure
    figure_1 = plt.figure(figsize=(20, 5))
    diagonal_axis = figure_1.add_subplot(1, 2, 1)
    full_axis = figure_1.add_subplot(1, 2, 2)

    # get X Values
    x_values = list(range(len(diagonal_vis_context_vis_1_mean)))
    x_values = np.multiply(x_values, 1000.0 / 6.4)

    # Plot Diagonal Lines
    diagonal_axis.plot(x_values, diagonal_vis_context_vis_1_mean, c='b')
    diagonal_axis.fill_between(x_values, diagonal_vis_context_vis_1_lower_bound, diagonal_vis_context_vis_1_upper_bound, color='b', alpha=0.5)

    diagonal_axis.plot(x_values, diagonal_vis_context_vis_2_mean, c='r')
    diagonal_axis.fill_between(x_values, diagonal_vis_context_vis_2_lower_bound, diagonal_vis_context_vis_2_upper_bound, color='r', alpha=0.5)

    diagonal_axis.plot(x_values, diagonal_odour_context_vis_1_mean, c='g')
    diagonal_axis.fill_between(x_values, diagonal_odour_context_vis_1_lower_bound, diagonal_odour_context_vis_1_upper_bound, color='g', alpha=0.5)

    diagonal_axis.plot(x_values, diagonal_odour_context_vis_2_mean, c='m')
    diagonal_axis.fill_between(x_values, diagonal_odour_context_vis_2_lower_bound, diagonal_odour_context_vis_2_upper_bound, color='m', alpha=0.5)

    # Plot Full
    full_axis.plot(x_values, full_vis_context_vis_1_mean, c='b')
    full_axis.fill_between(x_values, full_vis_context_vis_1_lower_bound, full_vis_context_vis_1_upper_bound, color='b', alpha=0.5)

    full_axis.plot(x_values, full_vis_context_vis_2_mean, c='r')
    full_axis.fill_between(x_values, full_vis_context_vis_2_lower_bound, full_vis_context_vis_2_upper_bound, color='r', alpha=0.5)

    full_axis.plot(x_values, full_odour_context_vis_1_mean, c='g')
    full_axis.fill_between(x_values, full_odour_context_vis_1_lower_bound, full_odour_context_vis_1_upper_bound, color='g', alpha=0.5)

    full_axis.plot(x_values, full_odour_context_vis_2_mean, c='m')
    full_axis.fill_between(x_values, full_odour_context_vis_2_lower_bound, full_odour_context_vis_2_upper_bound, color='m', alpha=0.5)

    # Set Y Lims
    diagonal_axis.set_ylim(ylim)
    full_axis.set_ylim(ylim)

    # Remove Spines
    diagonal_axis.spines[['right', 'top']].set_visible(False)
    full_axis.spines[['right', 'top']].set_visible(False)

    plt.show()
