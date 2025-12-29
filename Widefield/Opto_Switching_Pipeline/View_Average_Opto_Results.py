import os
import numpy as np

import numpy
import Opto_GLM_Utils
import Plotting_Functions


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
    """
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
    """

def compare_regressors(output_root, start_window, stop_window, regressor_list, experiment_name):

    # Compare Red Light Interactions
    control_interaction = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Red_Light_x_Attention" + "_group_coefs.npy"))
    opsin_interaction = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Red_Light_x_Attention" + "_group_coefs.npy"))
    print("control_interaction", np.shape(control_interaction))
    print("opsin_interaction", np.shape(opsin_interaction))

    save_directory = os.path.join(output_root, "Group_Results", experiment_name, "Interaction_Comparison")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    Plotting_Functions.compare_regressors(control_interaction, opsin_interaction, save_directory, start_window, stop_window)