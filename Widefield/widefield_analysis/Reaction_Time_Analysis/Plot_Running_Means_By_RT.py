import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import canny
from scipy import stats




def get_mean_and_bounds(data):
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    upper_bound = np.add(data_mean, data_sem)
    lower_bound = np.subtract(data_mean, data_sem)
    return data_mean, upper_bound, lower_bound



def get_bin_data(trial_type, bin_index, bin_start_list, bin_stop_list, output_directory):

    # Load Bin Data
    file_name = trial_type + "_Running_" + str(bin_start_list[bin_index]) + "_to_" + str(bin_stop_list[bin_index]) + ".npy"
    bin_data = np.load(os.path.join(output_directory, "RT_Bin_Mean_Running", file_name))

    return bin_data



def plot_running_trace(output_directory, n_bins, bin_start_list, bin_stop_list, start_window, stop_window):

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    colourmap = plt.get_cmap('plasma')

    for bin_index in range(n_bins):

        # Load Data
        hit_data = get_bin_data("Hit", bin_index, bin_start_list, bin_stop_list, output_directory)
        cr_data = get_bin_data("Cr", bin_index, bin_start_list, bin_stop_list, output_directory)
        #fa_data = get_bin_data("Fa", bin_index, bin_start_list, bin_stop_list, output_directory)

        # Get Mean and SD
        hit_mean, hit_upper_bound, hit_lower_bound = get_mean_and_bounds(hit_data)
        #fa_mean, fa_upper_bound, fa_lower_bound = get_mean_and_bounds(fa_data)
        cr_mean, cr_upper_bound, cr_lower_bound = get_mean_and_bounds(cr_data)

        # Plot Data
        colour = colourmap(float(bin_index) / n_bins)

        # Get X Values
        x_values = list(range(start_window, stop_window))

        axis_1.plot(x_values, hit_mean, c=colour, alpha=1)
        axis_1.fill_between(x_values, hit_lower_bound, hit_upper_bound, color=colour, alpha=0.1)

        #axis_1.plot(x_values, fa_mean, c=colour, alpha=1, linestyle='dashed')
        #axis_1.fill_between(x_values, fa_lower_bound, fa_upper_bound, color=colour, alpha=0.1)

    axis_1.plot(x_values, cr_mean, c='g', alpha=1, linestyle='dashed')
    axis_1.fill_between(x_values, cr_lower_bound, cr_upper_bound, color='g', alpha=0.1)

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_title("Running Speed")
    axis_1.spines['top'].set_visible(False)
    axis_1.spines['right'].set_visible(False)

    # Save Figures
    save_directory = os.path.join(output_directory, "Output_Plots")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    plt.savefig(os.path.join(save_directory, "Running_Speed.svg"))
    plt.close()



