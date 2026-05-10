import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import numpy
import Opto_GLM_Utils
import Plotting_Functions


def baseline_correct_regressors(regressor):

    baseline_corrected_regressor = []
    for mouse in regressor:
        mouse_baseline = mouse[0:14]
        mouse_baseline = np.mean(mouse_baseline, axis=0)
        mouse = np.subtract(mouse, mouse_baseline)
        baseline_corrected_regressor.append(mouse)
    baseline_corrected_regressor = np.array(baseline_corrected_regressor)
    return baseline_corrected_regressor


def z_score_regressor(data_root, session, regressor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()
    pixel_means = Opto_GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Opto_GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    z_scored_regressor = []
    n_timepoints = np.shape(regressor)[2]
    for timepoint_index in range(n_timepoints):
        timepoint_data = regressor[:, :, timepoint_index]
        timepoint_data = np.subtract(timepoint_data, pixel_means)
        timepoint_data = np.divide(timepoint_data, pixel_stds)
        z_scored_regressor.append(timepoint_data)

    z_scored_regressor = np.array(z_scored_regressor)
    z_scored_regressor = np.nan_to_num(z_scored_regressor)

    return z_scored_regressor



def reconstruct_regressor_into_pixel_space(data_root, session, regressor):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(data_root, session, "Local_NMF", "Spatial_Components.npy"))
    print("spatial_components", np.shape(spatial_components))

    # Reconstrct
    regressor = np.dot(spatial_components, regressor)
    print("regressor", np.shape(regressor))

    # Z Score
    regressor = z_score_regressor(data_root, session, regressor)
    print("regressor", np.shape(regressor))

    return regressor



def get_group_average_results(data_root, session_list, glm_output_directory, start_window, stop_window, regressor_list, experiment_name):

    # Create Save Directory
    save_directory = os.path.join(glm_output_directory, "Group_Results", experiment_name, "Group_Coefs")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Get Stim Coef Length
    stim_coef_size = stop_window - start_window
    n_regressors = len(regressor_list)
    print("stim_coef_size", stim_coef_size)

    # Get Coefs For Each Regressor
    for regressor_index in range(n_regressors):
        regressor_start = regressor_index * stim_coef_size
        regressor_stop = regressor_start + stim_coef_size

        # Get Group Averaged Results
        group_result_list = []
        for session in session_list:

            # Load Coefs
            coefs = np.load(os.path.join(glm_output_directory, session, "Model_Output", "Model_Coefs.npy"))
            print("Coefs", np.shape(coefs))

            # Get Coefs For This Regressor
            regressor_coefs = coefs[:, regressor_start:regressor_stop]
            print("regressor coefs", np.shape(regressor_coefs))

            # Reconstruct Into Pixel Space
            regressor_coefs = reconstruct_regressor_into_pixel_space(data_root, session, regressor_coefs)

            # Get Mouse Mean
            group_result_list.append(regressor_coefs)

        # Save These Coefs
        group_result_list = np.array(group_result_list)
        print("group_result_list", np.shape(group_result_list))

        np.save(os.path.join(save_directory, regressor_list[regressor_index] + "_group_coefs.npy"), group_result_list)






def view_group_average_results(data_root, session_list, output_directory, start_window, stop_window, regressor_list, experiment_name):

    # Get Group Average Results
    get_group_average_results(data_root, session_list, output_directory, start_window, stop_window, regressor_list, experiment_name)

    # Plot Each Regressor
    for regressor in regressor_list:

        # Load Coef
        group_coefs = np.load(os.path.join(output_directory, "Group_Results", experiment_name, "Group_Coefs", regressor + "_group_coefs.npy"))
        print("group_coefs", np.shape(group_coefs))

        # Select Save Directory
        save_directory = os.path.join(output_directory, "Group_Results", experiment_name, "Coef_Maps", regressor)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Plot Mean Regressor
        Plotting_Functions.plot_mean_regressor(group_coefs, start_window, stop_window, save_directory)



def compare_regressors(output_root, start_window, stop_window, experiment_name, comparison_window_start, comparison_window_stop):

    # Load Group Coefs
    control_visual_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Visual_Context_Control" + "_group_coefs.npy"))
    control_visual_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Visual_Context_Light" + "_group_coefs.npy"))
    control_odour_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Odour_Context_Control" + "_group_coefs.npy"))
    control_odour_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Odour_Context_Light" + "_group_coefs.npy"))

    opsin_visual_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Visual_Context_Control" + "_group_coefs.npy"))
    opsin_visual_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Visual_Context_Light" + "_group_coefs.npy"))
    opsin_odour_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Odour_Context_Control" + "_group_coefs.npy"))
    opsin_odour_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Odour_Context_Light" + "_group_coefs.npy"))

    # Baseline Correct Regressors
    control_visual_context_no_light = baseline_correct_regressors(control_visual_context_no_light)
    control_visual_context_red_light = baseline_correct_regressors(control_visual_context_red_light)
    control_odour_context_no_light = baseline_correct_regressors(control_odour_context_no_light)
    control_odour_context_red_light = baseline_correct_regressors(control_odour_context_red_light)

    opsin_visual_context_no_light = baseline_correct_regressors(opsin_visual_context_no_light)
    opsin_visual_context_red_light = baseline_correct_regressors(opsin_visual_context_red_light)
    opsin_odour_context_no_light = baseline_correct_regressors(opsin_odour_context_no_light)
    opsin_odour_context_red_light = baseline_correct_regressors(opsin_odour_context_red_light)


    # Get Light Effects
    control_visual_context_light_effect = np.subtract(control_visual_context_red_light, control_visual_context_no_light)
    control_odour_context_light_effect = np.subtract(control_odour_context_red_light, control_odour_context_no_light)
    opsin_visual_context_light_effect = np.subtract(opsin_visual_context_red_light, opsin_visual_context_no_light)
    opsin_odour_context_light_effect = np.subtract(opsin_odour_context_red_light, opsin_odour_context_no_light)

    # Get Contextual Modulation Of Light Effect
    control_contextual_modulation = np.subtract(control_visual_context_light_effect, control_odour_context_light_effect)
    opsin_contextual_modulation = np.subtract(opsin_visual_context_light_effect, opsin_odour_context_light_effect)

    save_directory = os.path.join(output_root, "Group_Results", experiment_name, "Interaction_Comparison")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Compare In Window
    print("control_contextual_modulation", np.shape(control_contextual_modulation))
    print("opsin_contextual_modulation", np.shape(opsin_contextual_modulation))
    control_contextual_modulation_window = control_contextual_modulation[:, comparison_window_start:comparison_window_stop]
    opsin_contextual_modulation_window = opsin_contextual_modulation[:, comparison_window_start:comparison_window_stop]
    print("control_contextual_modulation_window", np.shape(control_contextual_modulation_window))
    print("opsin_contextual_modulation_window", np.shape(opsin_contextual_modulation_window))

    control_contextual_modulation_window = np.mean(control_contextual_modulation_window, axis=1)
    opsin_contextual_modulation_window = np.mean(opsin_contextual_modulation_window, axis=1)
    print("control_contextual_modulation_window", np.shape(control_contextual_modulation_window))
    print("opsin_contextual_modulation_window", np.shape(opsin_contextual_modulation_window))

    plt.imshow(np.mean(control_contextual_modulation_window, axis=0))
    plt.show()

    plt.imshow(np.mean(opsin_contextual_modulation_window, axis=0))
    plt.show()

    diff = np.subtract(np.mean(control_contextual_modulation_window, axis=0), np.mean(opsin_contextual_modulation_window, axis=0))
    plt.imshow(diff)
    plt.show()

    t_stats, p_value = stats.ttest_ind(control_contextual_modulation_window, opsin_contextual_modulation_window)

    plt.imshow(p_value)
    plt.show()

    #Plotting_Functions.compare_regressors(control_contextual_modulation, opsin_contextual_modulation, save_directory, start_window, stop_window)