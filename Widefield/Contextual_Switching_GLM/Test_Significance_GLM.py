import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
from statsmodels.stats.multitest import fdrcorrection

from Widefield_Utils import widefield_utils


def test_signficance(nested_session_list, glm_output_directory, condition_1, condition_2, start_window, stop_window):


    group_condition_1_list = []
    group_condition_2_list = []

    for mouse in tqdm(nested_session_list):

        mouse_condition_1_list = []
        mouse_condition_2_list = []

        # Get Sessions For Each Mouse
        for session in mouse:

            condition_1_coefs = np.load(os.path.join(glm_output_directory, session, "Model_Output", condition_1 + "_coefs.npy"))
            condition_2_coefs = np.load(os.path.join(glm_output_directory, session, "Model_Output", condition_2 + "_coefs.npy"))

            mouse_condition_1_list.append(condition_1_coefs)
            mouse_condition_2_list.append(condition_2_coefs)

        # Get Mean Of Each Mouse
        mouse_mean_condition_1 = np.mean(mouse_condition_1_list, axis=0)
        mouse_mean_condition_2 = np.mean(mouse_condition_2_list, axis=0)

        group_condition_1_list.append(mouse_mean_condition_1)
        group_condition_2_list.append(mouse_mean_condition_2)

    t_stats, p_values = stats.ttest_rel(group_condition_1_list, group_condition_2_list, axis=0)


    colourmap = widefield_utils.get_musall_cmap()
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    n_timepoints = len(t_stats)

    """
    for timepoint_index in range(n_timepoints):
        timepoint_data = t_stats[timepoint_index]
        timepoint_p_values = p_values[timepoint_index]

        rejected, corrected_p = fdrcorrection(timepoint_p_values, alpha=0.05, is_sorted=False)
        timepoint_data = np.where(rejected == 1, timepoint_data, 0)

        image = widefield_utils.create_image_from_data(timepoint_data, indicies, image_height, image_width)

        plt.imshow(image, cmap=colourmap, vmin=-2.5, vmax=2.5)
        plt.title(str(x_values[timepoint_index]))
        plt.show()
    """


    # Test Within Window
    window_start = np.abs(start_window)
    window_stop = window_start + 42
    print("window_start", window_start)
    print("window_stop", window_stop)


    group_condition_1_list = np.array(group_condition_1_list)
    group_condition_2_list = np.array(group_condition_2_list)
    print("group_condition_1_list", np.shape(group_condition_1_list))
    print("group_condition_2_list", np.shape(group_condition_2_list))

    window_mean_condition_1 = np.mean(group_condition_1_list[:, window_start:window_stop], axis=1)
    window_mean_condition_2 = np.mean(group_condition_2_list[:, window_start:window_stop], axis=1)
    print("window_mean_condition_1", np.shape(window_mean_condition_1))
    print("window_mean_condition_2", np.shape(window_mean_condition_2))

    grand_mean_1 = np.mean(window_mean_condition_1, axis=0)
    grand_mean_2 = np.mean(window_mean_condition_2, axis=0)
    mean_diff = np.subtract(grand_mean_1, grand_mean_2)

    t_stats, p_values = stats.ttest_rel(window_mean_condition_1, window_mean_condition_2, axis=0)
    rejected, corrected_p = fdrcorrection(p_values, alpha=0.05, is_sorted=False)
    thresholded_diff = np.where(rejected==True, mean_diff, 0)
    #thresholded_diff = mean_diff

    image = widefield_utils.create_image_from_data(thresholded_diff, indicies, image_height, image_width)

    plt.imshow(image, cmap=colourmap, vmin=-0.2, vmax=0.2)
    plt.title("Window Average")
    plt.show()

    print("t stats", np.shape(t_stats))
