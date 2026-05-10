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


def plot_roi_trace(output_directory, n_bins, bin_start_list, bin_stop_list, atlas, atlas_dict, selected_roi, bin_stop_frame_list, start_window):

    # Get ROI Pixels
    roi_label = atlas_dict[selected_roi]
    image_height, image_width = np.shape(atlas)
    atlas = np.reshape(atlas, image_height * image_width)
    pixel_map = np.where(atlas == roi_label)[0]

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    colourmap = plt.get_cmap('plasma')


    for bin_index in range(n_bins):

        # Load Bin Data
        file_name = str(bin_start_list[bin_index]) + "_to_" + str(bin_stop_list[bin_index]) + ".npy"
        bin_data = np.load(os.path.join(output_directory, "RT_Bin_Means", file_name))

        # Get ROI Mean
        n_mice, image_height, image_width, n_timepoints = np.shape(bin_data)
        bin_data = np.reshape(bin_data, (n_mice, image_height * image_width, n_timepoints))
        roi_data = bin_data[:, pixel_map]
        roi_data = np.mean(roi_data, axis=1)

        # Cut Off Data at Lick
        roi_data = roi_data[:, 0:bin_stop_frame_list[bin_index]]

        # Get Mean and SD
        data_mean, upper_bound, lower_bound = get_mean_and_bounds(roi_data)

        # Plot Data
        colour = colourmap(float(bin_index) / n_bins)

        # Get X Values
        x_values = list(range(bin_stop_frame_list[bin_index]))
        x_values = np.add(x_values, start_window)
        x_values = np.multiply(x_values, 37)

        axis_1.plot(x_values, data_mean, c=colour, alpha=1)
        axis_1.scatter([x_values[-1]],[data_mean[-1]], c=colour)
        axis_1.fill_between(x_values, lower_bound, upper_bound, color=colour, alpha=0.1)

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_ylim([-0.005, 0.08])
    axis_1.set_title(selected_roi)
    axis_1.spines['top'].set_visible(False)
    axis_1.spines['right'].set_visible(False)

    # Save Figures
    save_directory = os.path.join(output_directory, "Output_Plots")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    plt.savefig(os.path.join(save_directory, selected_roi + ".svg"))
    plt.close()

