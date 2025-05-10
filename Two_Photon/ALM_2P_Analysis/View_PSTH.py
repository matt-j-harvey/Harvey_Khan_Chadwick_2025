import os
import numpy as np
import matplotlib.pyplot as plt

from scipy import stats


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def test_psth_significance(tensor_1, tensor_2):

    n_trials, n_timepoints, n_neurons = np.shape(tensor_1)

    significance_map = np.zeros((n_timepoints, n_neurons))

    for timepoint_index in range(n_timepoints):
        for neuron_index in range(n_neurons):
            condition_1_values = tensor_1[:, timepoint_index, neuron_index]
            condition_2_values = tensor_2[:, timepoint_index, neuron_index]
            t_stat, p_value = stats.ttest_ind(condition_1_values, condition_2_values)
            p_value = np.nan_to_num(p_value,nan=1)

            if p_value < 0.05:
                significance_map[timepoint_index, neuron_index] = 1

    return significance_map



def test_signficance_one_sided(tensor):

    n_trials, n_timepoints, n_neurons = np.shape(tensor)

    significance_map = np.zeros((n_timepoints, n_neurons))

    for timepoint_index in range(n_timepoints):
        for neuron_index in range(n_neurons):
            neuron_values = tensor[:, timepoint_index, neuron_index]
            d_stat, p_value = stats.ttest_1samp(a=neuron_values, popmean=0)

            p_value = np.nan_to_num(p_value,nan=1)

            if p_value < 0.05:
                significance_map[timepoint_index, neuron_index] = 1

    return significance_map





def test_signficance_one_sided_window(tensor, window_start, window_stop):

    n_neurons = np.shape(tensor)[2]
    significance_vector = np.zeros(n_neurons)

    for neuron_index in range(n_neurons):
        neuron_values = tensor[:, window_start:window_stop, neuron_index]
        neuron_values = np.mean(neuron_values, axis=1)

        t_stat, p_value = stats.ttest_1samp(a=neuron_values, popmean=0)
        p_value = np.nan_to_num(p_value, nan=1)

        if p_value < 0.05:
            significance_vector[neuron_index] = 1

    mean_activity = np.mean(tensor, axis=0)
    sig_cells = np.multiply(mean_activity, significance_vector)
    print("Sig cells", np.shape(sig_cells))
    return sig_cells




def compute_single_raster(tensor):

    # Get Mean Activity
    mean_activity = np.nanmean(tensor, axis=0)

    # Test Significance
    significance_map = test_signficance_one_sided(tensor)

    # Get Activity Thresholded By Significance
    sig_activity = np.multiply(mean_activity, significance_map)

    return mean_activity, sig_activity


def sort_raster(master_raster, other_raster_list, sorting_window_start, sorting_window_stop):

    # Get Mean Response in Sorting Window
    response = sig_activity[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)

    # Get Sorted Indicies
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    # Sort Rasters
    sorted_master_raster = master_raster[:, sorted_indicies]
    sorted_raster_list = []
    for raster in other_raster_list:
        sorted_raster = raster[:, sorted_indicies]
        sorted_raster_list.append(sorted_raster)

    return sorted_master_raster, sorted_raster_list




def view_single_psth(tensor, start_window, stop_window, frame_rate, save_directory, condition_name, sorting_window_start, sorting_window_stop):

    # Get PSTHs
    mean_activity, sig_activity = compute_single_raster(tensor)

    # Sort Rasters
    sig_activity, [mean_activity] = sort_raster(sig_activity, [mean_activity])

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_rate)

    # Plot Raster
    n_neurons = np.shape(tensor)[2]
    magnitude = np.percentile(np.abs(mean_activity), q=99.5)

    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1,2,1)
    axis_2 = figure_1.add_subplot(1,2,2)

    axis_1.imshow(np.transpose(mean_activity), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent = [x_values[0], x_values[-1], 0, n_neurons])
    axis_2.imshow(np.transpose(sig_activity), vmin=-magnitude, vmax=magnitude, cmap='bwr',  extent = [x_values[0], x_values[-1], 0, n_neurons])

    axis_1.axvline(0, linestyle='dashed', c='k')
    axis_2.axvline(0, linestyle='dashed', c='k')

    axis_1.set_xlabel("Time (S)")
    axis_2.set_xlabel("Time (S)")

    axis_1.set_ylabel("Neurons")
    axis_2.set_ylabel("Neurons")

    forceAspect(axis_1)
    forceAspect(axis_2)

    figure_1.suptitle(condition_name)
    plt.show()
    #plt.savefig(os.path.join(save_directory, comparison_name))
    #plt.close()





def view_two_psth(tensor_1, tensor_2, start_window, stop_window, frame_rate, save_directory, comparison_name, sorting_window_start, sorting_window_stop, plot_titles=None):

    # Get Means
    mean_1 = np.nanmean(tensor_1, axis=0)
    mean_2 = np.nanmean(tensor_2, axis=0)
    mean_1 = np.nan_to_num(mean_1)
    mean_2 = np.nan_to_num(mean_2)

    # Get Significance
    significance_map = test_psth_significance(tensor_1, tensor_2)

    # Get Diff
    diff = np.subtract(mean_1, mean_2)
    sig_diff = np.multiply(diff, significance_map)

    # Sort PSTHs
    response = sig_diff[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    psth_1 = mean_1[:, sorted_indicies]
    psth_2 = mean_2[:, sorted_indicies]
    diff = diff[:, sorted_indicies]
    sig_diff = sig_diff[:, sorted_indicies]

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_rate)
    n_neurons = np.shape(psth_1)[1]

    #magnitude = np.percentile(np.abs(psth_1), q=99.5)
    magnitude = 1

    figure_1 = plt.figure(figsize=(15,5))
    axis_1 = figure_1.add_subplot(1,4,1)
    axis_2 = figure_1.add_subplot(1,4,2)
    axis_3 = figure_1.add_subplot(1,4,3)
    axis_4 = figure_1.add_subplot(1,4,4)

    axis_1.imshow(np.transpose(psth_1), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent = [x_values[0], x_values[-1], 0, n_neurons])
    axis_2.imshow(np.transpose(psth_2), vmin=-magnitude, vmax=magnitude, cmap='bwr',  extent = [x_values[0], x_values[-1], 0, n_neurons])
    axis_3.imshow(np.transpose(diff), vmin=-magnitude, vmax=magnitude, cmap='bwr',  extent = [x_values[0], x_values[-1], 0, n_neurons])
    axis_4.imshow(np.transpose(sig_diff), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])

    axis_1.axvline(0, linestyle='dashed', c='k')
    axis_2.axvline(0, linestyle='dashed', c='k')
    axis_3.axvline(0, linestyle='dashed', c='k')
    axis_4.axvline(0, linestyle='dashed', c='k')

    axis_1.set_xlabel("Time (S)")
    axis_2.set_xlabel("Time (S)")
    axis_3.set_xlabel("Time (S)")
    axis_4.set_xlabel("Time (S)")

    axis_1.set_ylabel("Neurons")
    axis_2.set_ylabel("Neurons")
    axis_3.set_ylabel("Neurons")
    axis_4.set_ylabel("Neurons")

    forceAspect(axis_1)
    forceAspect(axis_2)
    forceAspect(axis_3)
    forceAspect(axis_4)

    # Set Ttitles
    if plot_titles != None:
        axis_1.set_title(plot_titles[0])
        axis_2.set_title(plot_titles[1])

    figure_1.suptitle(comparison_name)
    plt.savefig(os.path.join(save_directory, comparison_name))
    plt.show()
    plt.close()


def view_two_mean_psth(mean_1, mean_2, start_window, stop_window, frame_rate, sorting_window_start, sorting_window_stop, plot_titles=None):

    # Get Diff
    diff = np.subtract(mean_1, mean_2)

    # Sort PSTHs
    response = diff[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    psth_1 = mean_1[:, sorted_indicies]
    psth_2 = mean_2[:, sorted_indicies]
    diff = diff[:, sorted_indicies]

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_rate)
    n_neurons = np.shape(psth_1)[1]

    # magnitude = np.percentile(np.abs(psth_1), q=99.5)
    magnitude = 1

    figure_1 = plt.figure(figsize=(15, 5))
    axis_1 = figure_1.add_subplot(1, 3, 1)
    axis_2 = figure_1.add_subplot(1, 3, 2)
    axis_3 = figure_1.add_subplot(1, 3, 3)

    axis_1.imshow(np.transpose(psth_1), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])
    axis_2.imshow(np.transpose(psth_2), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])
    axis_3.imshow(np.transpose(diff), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])

    axis_1.axvline(0, linestyle='dashed', c='k')
    axis_2.axvline(0, linestyle='dashed', c='k')
    axis_3.axvline(0, linestyle='dashed', c='k')

    axis_1.set_xlabel("Time (S)")
    axis_2.set_xlabel("Time (S)")
    axis_3.set_xlabel("Time (S)")

    axis_1.set_ylabel("Neurons")
    axis_2.set_ylabel("Neurons")
    axis_3.set_ylabel("Neurons")

    forceAspect(axis_1)
    forceAspect(axis_2)
    forceAspect(axis_3)

    # Set Ttitles
    if plot_titles != None:
        axis_1.set_title(plot_titles[0])
        axis_2.set_title(plot_titles[1])

    plt.show()


def view_single_psth_sig_pre_computed(mean_activity,
                                      sig_activity,
                                      window_sig_cells,
                                      start_window,
                                      stop_window,
                                      frame_rate,
                                      save_directory,
                                      condition_name,
                                      sorting_window_start,
                                      sorting_window_stop,
                                      plot_titles=None,
                                      magnitude=None):

    # Sort Raster
    response = mean_activity[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)
    mean_activity = mean_activity[:, sorted_indicies]
    sig_activity = sig_activity[:, sorted_indicies]
    window_sig_cells = window_sig_cells[:, sorted_indicies]

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_rate)

    # Plot Raster
    n_neurons = np.shape(mean_activity)[1]

    if magnitude == None:
        magnitude = np.percentile(np.abs(mean_activity), q=99.5)

    figure_1 = plt.figure(figsize=(12, 2.5))
    axis_1 = figure_1.add_subplot(1, 3, 1)
    axis_2 = figure_1.add_subplot(1, 3, 2)
    axis_3 = figure_1.add_subplot(1, 3, 3)

    axis_1_handle = axis_1.imshow(np.transpose(mean_activity),
                  vmin=-magnitude,
                  vmax=magnitude,
                  cmap='bwr',
                  extent=[x_values[0], x_values[-1], 0, n_neurons])

    axis_2_handle = axis_2.imshow(np.transpose(sig_activity),
                  vmin=-magnitude,
                  vmax=magnitude,
                  cmap='bwr',
                  extent=[x_values[0], x_values[-1], 0, n_neurons])

    axis_3_handle = axis_3.imshow(np.transpose(window_sig_cells),
                  vmin=-magnitude,
                  vmax=magnitude,
                  cmap='bwr',
                  extent=[x_values[0], x_values[-1], 0, n_neurons])


    axis_1.axvline(0, linestyle='dashed', c='k')
    axis_2.axvline(0, linestyle='dashed', c='k')
    axis_3.axvline(0, linestyle='dashed', c='k')

    axis_1.set_xlabel("Time (S)")
    axis_2.set_xlabel("Time (S)")
    axis_3.set_xlabel("Time (S)")

    axis_1.set_ylabel("Neurons")
    #axis_2.set_ylabel("Neurons")
    #axis_3.set_ylabel("Neurons")

    forceAspect(axis_1)
    forceAspect(axis_2)
    forceAspect(axis_3)

    # Add Colourbars
    figure_1.colorbar(axis_1_handle, orientation='vertical')
    figure_1.colorbar(axis_2_handle, orientation='vertical')
    figure_1.colorbar(axis_3_handle, orientation='vertical')

    # Set Ttitles
    if plot_titles != None:
        axis_1.set_title(plot_titles[0])
        axis_2.set_title(plot_titles[1])
        axis_3.set_title(plot_titles[2])

    figure_1.suptitle(condition_name)


    plt.show()
    #plt.savefig(os.path.join(save_directory, condition_name))
    #plt.close()



