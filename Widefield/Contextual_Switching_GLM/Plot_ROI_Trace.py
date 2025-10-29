import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

from Widefield_Utils import widefield_utils



def get_mean_sd(data):
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound

def get_roi_pixels(atlas, roi_list):

    # Mask Atlas
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    atlas = np.reshape(atlas, image_height * image_width)
    atlas = atlas[indicies]

    selected_indicies = []
    for roi in roi_list:
        roi_mask = np.where(atlas==roi, 1, 0)
        roi_indicies = np.argwhere(roi_mask)

        for index in roi_indicies:
            selected_indicies.append(index[0])

    selected_indicies = np.array(selected_indicies)
    return selected_indicies





def plot_roi_trace(coef_directory, condition_1, condition_2, start_window, stop_window, frame_period, roi_list, sig_height):

    # Load Atlas
    atlas = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/M2_Three_Segments_Masked_All_Sessions.npy")
    atlas = np.abs(atlas)
    plt.imshow(atlas)
    plt.show()

    # Get ROI Indicies
    roi_indicies = get_roi_pixels(atlas, roi_list)
    print("roi_indicies", roi_indicies)


    # Load Matrix
    group_1_coef_matrix = np.load(os.path.join(coef_directory, condition_1 + "_group_coefs.npy"))
    group_2_coef_matrix = np.load(os.path.join(coef_directory, condition_2 + "_group_coefs.npy"))

    # Get ROI
    group_1_roi_matrix = np.mean(group_1_coef_matrix[:, :, roi_indicies], axis=2)
    group_2_roi_matrix = np.mean(group_2_coef_matrix[:, :, roi_indicies], axis=2)

    group_1_roi_matrix = group_1_roi_matrix[:, 27:]
    group_2_roi_matrix = group_2_roi_matrix[:, 27:]

    # Get Means and STD
    group_1_mean, group_1_lower, group_1_upper = get_mean_sd(group_1_roi_matrix)
    group_2_mean, group_2_lower, group_2_upper = get_mean_sd(group_2_roi_matrix)

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_period)
    x_values = x_values[27:]

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)


    axis_1.plot(x_values, group_1_mean, c='b')
    axis_1.fill_between(x_values, group_1_lower, group_1_upper, color='b', alpha=0.5)

    axis_1.plot(x_values, group_2_mean, c='m')
    axis_1.fill_between(x_values, group_2_lower, group_2_upper, color='m', alpha=0.5)

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.spines[['right', 'top']].set_visible(False)

    # Plot Signficance
    t_stats, p_values = stats.ttest_rel(group_1_roi_matrix, group_2_roi_matrix, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_markers = np.ones(len(x_values))
    sig_markers = np.multiply(sig_markers, sig_height)
    sig_markers = np.multiply(sig_markers, binary_sig)

    axis_1.scatter(x_values, sig_markers, alpha=binary_sig, c='Grey', marker='s')

    plt.show()


coef_directory = r"/media/matthew/29D46574463D2856/Paper_Results/Contextual_Swtiching_GLM/Group_Regressors"
#plot_roi_trace(coef_directory, "vis_context_vis_2", "odr_context_vis_2", roi_list=[9])


frame_period = 36
start_window_ms = -1500
stop_window_ms = 2000
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


plot_roi_trace(coef_directory, "vis_context_vis_2", "odr_context_vis_2", start_window, stop_window, frame_period, roi_list=[14, 15, 16], sig_height=0.5)
plot_roi_trace(coef_directory, "vis_context_vis_2", "odr_context_vis_2", start_window, stop_window, frame_period, roi_list=[9], sig_height=1.2)

