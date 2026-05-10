import os
import numpy as np
import matplotlib.pyplot as plt

import Opto_GLM_Utils

def z_score_regressor(data_root, session, regressor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()
    pixel_means = Opto_GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Opto_GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    regressor = np.subtract(regressor, pixel_means)
    regressor = np.divide(regressor, pixel_stds)

    return regressor


def reconstruct_regressor_into_pixel_space(data_root, session, regressor):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(data_root, session, "Local_NMF", "Spatial_Components.npy"))

    # Reconstruct
    regressor = np.dot(spatial_components, regressor)
    regressor = np.moveaxis(regressor, -1, 0)

    # Z Score
    regressor = z_score_regressor(data_root, session, regressor)

    return regressor


def get_group_average_results(data_root,
                              session_list,
                              glm_output_directory,
                              start_window,
                              stop_window,
                              comparison_window_start,
                              comparison_window_stop,
                              regressor_list,
                              experiment_name):

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

            # Get Mean In Comparison Window
            regressor_coefs = regressor_coefs[comparison_window_start:comparison_window_stop]
            regressor_coefs = np.mean(regressor_coefs, axis=0)

            # Get Mouse Mean
            group_result_list.append(regressor_coefs)

        # Save These Coefs
        group_result_list = np.array(group_result_list)
        print("group_result_list", np.shape(group_result_list))

        np.save(os.path.join(save_directory, regressor_list[regressor_index] + "_group_coefs.npy"), group_result_list)



