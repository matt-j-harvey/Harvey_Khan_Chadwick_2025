import os
import numpy as np
import matplotlib.pyplot as plt

import Session_List
import GLM_Utils
import Plotting_Functions

def get_cr_activity(data_root, session_list, start_window, stop_window):

    group_list = []

    for mouse in session_list:
        mouse_list = []

        for session in mouse:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Get CR Onsets
            onsets_list = GLM_Utils.get_cr_onsets(behaviour_matrix)

            # Load Activity Matrix
            activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
            activity_matrix = np.transpose(activity_matrix)

            # Get Activity Tensors
            activity_tensor = GLM_Utils.get_data_tensor(activity_matrix, onsets_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14)

            # Get Mean Response
            mean_response = np.mean(activity_tensor, axis=0)

            # Reconstruct Back Into Pixel Space
            mean_response = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, mean_response)

            mouse_list.append(mean_response)

        mouse_list = np.array(mouse_list)
        print("mouse_list", np.shape(mouse_list))

        mouse_mean = np.mean(mouse_list, axis=0)
        group_list.append(mouse_mean)

    group_list = np.array(group_list)
    print("group_list", np.shape(group_list))
    return group_list


def get_hit_activity_rt_window(data_root, session_list, rt_window_start, rt_window_stop):

    group_list = []

    for mouse in session_list:
        mouse_list = []

        for session in mouse:

            # Get CR Onsets
            onsets_list = GLM_Utils.get_hit_onsets_rt_window(data_root, session, rt_window_start, rt_window_stop)

            if len(onsets_list) > 2:

                # Load Activity Matrix
                activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
                activity_matrix = np.transpose(activity_matrix)

                # Get Activity Tensors
                activity_tensor = GLM_Utils.get_data_tensor(activity_matrix, onsets_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14)

                # Get Mean Response
                mean_response = np.mean(activity_tensor, axis=0)

                # Reconstruct Back Into Pixel Space
                mean_response = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, mean_response)
                mouse_list.append(mean_response)

        mouse_list = np.array(mouse_list)
        print("mouse_list", np.shape(mouse_list))

        mouse_mean = np.mean(mouse_list, axis=0)
        group_list.append(mouse_mean)

    group_list = np.array(group_list)
    return group_list


def view_mean_post_learning_activity(data_root, session_list, output_root, start_window, stop_window):

    current_x_values = list(range(start_window, stop_window))
    current_x_values = np.multiply(current_x_values, 37)

    # Get Mean CR Activity
    cr_activity = get_cr_activity(data_root, session_list, start_window, stop_window)
    mean_cr_activity = np.mean(cr_activity, axis=0)

    # Get Mean Hit Activity
    hit_activity = get_hit_activity_rt_window(data_root, session_list, 1400, 1600)
    mean_hit_activity = np.mean(hit_activity, axis=0)


    # Quantify ROI Activity
    roi_save_directory = os.path.join(output_root,"Mean_Post_Learning_Activity", "ROI Plots")
    if not os.path.exists(roi_save_directory):
        os.makedirs(roi_save_directory)

    #Plotting_Functions.compare_roi_within_groups(hit_activity, cr_activity, roi_save_directory, start_window, stop_window, [9], "V1", ylim=[-0.7, 3])
    #Plotting_Functions.compare_roi_within_groups(hit_activity, cr_activity, roi_save_directory, start_window, stop_window, [1], "M1", ylim=[-0.7, 3])
    #Plotting_Functions.compare_roi_within_groups(hit_activity, cr_activity, roi_save_directory, start_window, stop_window, [14, 15, 16], "M2", ylim=[-0.7, 3])





    # Plot Mean CR Activity
    save_directory = os.path.join(output_root, "Mean_Post_Learning_Activity", "CR")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    #mean_activity, new_x_values = GLM_Utils.linearly_interpolate(mean_cr_activity, 37, 10, current_x_values, 5)
    #Plotting_Functions.visualise_mean_regressor(mean_activity, save_directory, new_x_values, magnitude=[0, 2], cmap=plt.get_cmap('jet'))

    # Plot Mean CR Activity
    save_directory = os.path.join(output_root, "Mean_Post_Learning_Activity", "Hit")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    #mean_activity, new_x_values = GLM_Utils.linearly_interpolate(mean_hit_activity, 37, 10, current_x_values, 5)
    #Plotting_Functions.visualise_mean_regressor(mean_activity, save_directory, new_x_values, magnitude=[0, 2], cmap=plt.get_cmap('jet'))


    # Get Difference
    difference = np.subtract(mean_hit_activity, mean_cr_activity)
    difference, new_x_values = GLM_Utils.linearly_interpolate(difference, 37, 10, current_x_values, 5)

    # Plot Mean CR Activity
    save_directory = os.path.join(output_root, "Mean_Post_Learning_Activity", "Diff")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    Plotting_Functions.visualise_mean_regressor(difference, save_directory, new_x_values, magnitude=[-2, 2], cmap=GLM_Utils.get_musall_cmap())



frame_period = 37
start_window_ms = -2800
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

session_list = Session_List.control_all_post_learning
output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"
data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"

view_mean_post_learning_activity(data_root, session_list, output_root, start_window, stop_window)