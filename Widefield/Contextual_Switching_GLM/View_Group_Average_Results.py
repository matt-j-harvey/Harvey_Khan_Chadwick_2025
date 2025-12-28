import os
import numpy as np

import GLM_Utils
import Plotting_Functions

def z_score_regressor(data_root, session, regressor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()
    pixel_means = GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

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



def get_group_average_results(data_root, session_list, glm_output_directory, start_window, stop_window, regressor_list):

    # Create Save Directory
    save_directory = os.path.join(glm_output_directory, "Group_Results", "Group_Coefs")
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
        for mouse in session_list:
            mouse_result_list = []

            for session in mouse:

                # Load Coefs
                coefs = np.load(os.path.join(glm_output_directory, session, "Model_Output", "Model_Coefs.npy"))
                print("Coefs", np.shape(coefs))

                # Get Coefs For This Regressor
                regressor_coefs = coefs[:, regressor_start:regressor_stop]
                print("regressor coefs", np.shape(regressor_coefs))

                # Reconstruct Into Pixel Space
                regressor_coefs = reconstruct_regressor_into_pixel_space(data_root, session, regressor_coefs)

                # Add To List
                mouse_result_list.append(regressor_coefs)

            # Get Mouse Mean
            mouse_mean = np.mean(mouse_result_list, axis=0)
            group_result_list.append(mouse_mean)

        # Save These Coefs
        group_result_list = np.array(group_result_list)
        print("group_result_list", np.shape(group_result_list))

        np.save(os.path.join(save_directory, regressor_list[regressor_index] + "_group_coefs.npy"), group_result_list)







def view_group_average_results(data_root, session_list, glm_output_directory, start_window, stop_window, regressor_list):

    # Get Group Average Results
    #get_group_average_results(data_root, session_list, glm_output_directory, start_window, stop_window, regressor_list)

    """
    # Iterate Through Each Regressor
    for regressor in regressor_list:

        # Load Group Regressor Data
        regressor_data = np.load(os.path.join(glm_output_directory, "Group_Results", "Group_Coefs", regressor + "_group_coefs.npy"))
        print("regressor_data", np.shape(regressor_data))

        # Create Visualisation Save Directory
        save_directory = os.path.join(glm_output_directory,  "Group_Results", "Regressor_Maps", regressor)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Visualise Regressor
        mean_regressor = np.mean(regressor_data, axis=0)
        Plotting_Functions.visualise_mean_regressor(mean_regressor, save_directory,  start_window, stop_window)
    """


    # Compare Regressors
    vis_1_rel = np.load(os.path.join(glm_output_directory, "Group_Results", "Group_Coefs", "Vis_1_Rel_group_coefs.npy"))
    vis_1_irrel = np.load(os.path.join(glm_output_directory, "Group_Results", "Group_Coefs", "Vis_1_Irrel_group_coefs.npy"))
    vis_2_rel = np.load(os.path.join(glm_output_directory, "Group_Results", "Group_Coefs", "Vis_2_Rel_group_coefs.npy"))
    vis_2_irrel = np.load(os.path.join(glm_output_directory, "Group_Results", "Group_Coefs", "Vis_2_Irrel_group_coefs.npy"))

    # Create Visualisation Save Directory
    vis_1_save_directory = os.path.join(glm_output_directory, "Group_Results", "Regressor_Maps", "Vis 1 Rel v Irrel")
    vis_2_save_directory = os.path.join(glm_output_directory, "Group_Results", "Regressor_Maps", "Vis 2 Rel v Irrel")

    if not os.path.exists(vis_1_save_directory):
        os.makedirs(vis_1_save_directory)

    if not os.path.exists(vis_2_save_directory):
        os.makedirs(vis_2_save_directory)

    Plotting_Functions.compare_regressors(vis_1_rel, vis_1_irrel, vis_1_save_directory, start_window, stop_window)
    Plotting_Functions.compare_regressors(vis_2_rel, vis_2_irrel, vis_2_save_directory, start_window, stop_window)



    # Compare Across Time Window
    window_start = np.abs(start_window)
    window_stop = window_start + 41
    #Plotting_Functions.compare_regressor_window(vis_2_rel, vis_2_irrel, save_directory, window_start, window_stop)


    # Plot ROI Diffs
    roi_save_directory = os.path.join(glm_output_directory, "Group_Results", "ROI Traces")
    if not os.path.exists(roi_save_directory):
        os.makedirs(roi_save_directory)

    Plotting_Functions.plot_roi_diff(vis_2_rel, vis_2_irrel, roi_save_directory, start_window, stop_window, [15, 14, 16], "vis 2 M2")
    Plotting_Functions.plot_roi_diff(vis_2_rel, vis_2_irrel, roi_save_directory, start_window, stop_window, [9], "vis 2 V1")

    Plotting_Functions.plot_roi_diff(vis_1_rel, vis_1_irrel, roi_save_directory, start_window, stop_window, [15, 14, 16], "vis 1 M2")
    Plotting_Functions.plot_roi_diff(vis_1_rel, vis_1_irrel, roi_save_directory, start_window, stop_window, [9], "vis 1 V1")

