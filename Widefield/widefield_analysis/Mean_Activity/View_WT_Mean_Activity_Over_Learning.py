import os
import numpy as np
import matplotlib.pyplot as plt

import Session_List
import Mean_Activity_Utils
import Plotting_Functions


def get_session_mean_activity(data_root, session, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Onsets
    vis_1_onsets = Mean_Activity_Utils.get_vis_1_onsets(behaviour_matrix)
    vis_2_onsets = Mean_Activity_Utils.get_vis_2_onsets(behaviour_matrix)

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
    activity_matrix = np.transpose(activity_matrix)
    print("activity_matrix", np.shape(activity_matrix))

    # Get Data Tensors
    vis_1_tensor = Mean_Activity_Utils.get_data_tensor(activity_matrix, vis_1_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)
    vis_2_tensor = Mean_Activity_Utils.get_data_tensor(activity_matrix, vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)

    # Get Trial Mean
    vis_1_mean = np.mean(vis_1_tensor, axis=0)
    vis_2_mean = np.mean(vis_2_tensor, axis=0)

    # Reconstruct Into Pixel Space
    vis_1_mean = Mean_Activity_Utils.reconstruct_regressor_into_pixel_space(data_root, session, vis_1_mean)
    vis_2_mean = Mean_Activity_Utils.reconstruct_regressor_into_pixel_space(data_root, session, vis_2_mean)

    return vis_1_mean, vis_2_mean



def get_group_mean_activity(data_root, session_list, output_root, start_window, stop_window):

    group_vis_1_list = []
    group_vis_2_list = []

    for mouse in session_list:
        mouse_vis_1_list = []
        mouse_vis_2_list = []

        for session in mouse:
            vis_1_coefs, vis_2_coefs = get_session_mean_activity(data_root, session, start_window, stop_window)
            mouse_vis_1_list.append(vis_1_coefs)
            mouse_vis_2_list.append(vis_2_coefs)

        mouse_vis_1_list = np.array(mouse_vis_1_list)
        mouse_vis_2_list = np.array(mouse_vis_2_list)

        mouse_vis_1_mean = np.mean(mouse_vis_1_list, axis=0)
        mouse_vis_2_mean = np.mean(mouse_vis_2_list, axis=0)

        group_vis_1_list.append(mouse_vis_1_mean)
        group_vis_2_list.append(mouse_vis_2_mean)

    group_vis_1_list = np.array(group_vis_1_list)
    group_vis_2_list = np.array(group_vis_2_list)

    save_directory = os.path.join(output_root, "Mean_Activity")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Group_Vis_1_Activity.npy"), group_vis_1_list)
    np.save(os.path.join(save_directory, "Group_Vis_2_Activity.npy"), group_vis_2_list)



def visualise_activity_comparison(group_1_activity, group_2_activity, output_root, start_window, stop_window):

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 37)

    # Get Means and Diffs
    group_1_mean = np.mean(group_1_activity, axis=0)
    group_2_mean = np.mean(group_2_activity, axis=0)
    difference = np.subtract(group_1_mean, group_2_mean)

    # Create Save Directories
    save_directory_list = [os.path.join(output_root, "Vis_1_Activity_Map"),
                           os.path.join(output_root, "Vis_2_Activity_Map"),
                           os.path.join(output_root, "Diff_Activity_Map"),]

    for save_directory in save_directory_list:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    # Plot These
    Plotting_Functions.visualise_mean_regressor(group_1_mean, save_directory_list[0], x_values, magnitude=[0, 2], cmap=plt.get_cmap("inferno"))
    Plotting_Functions.visualise_mean_regressor(group_2_mean, save_directory_list[1], x_values, magnitude=[0, 2], cmap=plt.get_cmap("inferno"))
    Plotting_Functions.visualise_mean_regressor(difference, save_directory_list[2],   x_values, magnitude=[-0.75, 0.75], cmap=Mean_Activity_Utils.get_musall_cmap())


def learning_mean_activity_pipeline(data_root,
                                    pre_learning_sessions,
                                    intermediate_learning_sessions,
                                    post_learning_session,
                                    output_root,
                                    start_window,
                                    stop_window):

    # Get Group Mean Activity
    get_group_mean_activity(data_root, pre_learning_sessions, os.path.join(output_root, "WT_Pre_Learning"), start_window, stop_window)
    get_group_mean_activity(data_root, intermediate_learning_sessions, os.path.join(output_root, "WT_Intermediate_Learning"), start_window, stop_window)
    get_group_mean_activity(data_root, post_learning_session, os.path.join(output_root, "WT_Post_Learning"), start_window, stop_window)


    # Load Activity
    pre_learning_vis_1 = np.load(os.path.join(r"C:\Learning_Mean_Activity\WT_Pre_Learning\Mean_Activity", "Group_Vis_1_Activity.npy"))
    pre_learning_vis_2 = np.load(os.path.join(r"C:\Learning_Mean_Activity\WT_Pre_Learning\Mean_Activity", "Group_Vis_2_Activity.npy"))

    int_learning_vis_1 = np.load(os.path.join(r"C:\Learning_Mean_Activity\WT_Intermediate_Learning\Mean_Activity", "Group_Vis_1_Activity.npy"))
    int_learning_vis_2 = np.load(os.path.join(r"C:\Learning_Mean_Activity\WT_Intermediate_Learning\Mean_Activity", "Group_Vis_2_Activity.npy"))

    post_learning_vis_1 = np.load(os.path.join(r"C:\Learning_Mean_Activity\WT_Post_Learning\Mean_Activity", "Group_Vis_1_Activity.npy"))
    post_learning_vis_2 = np.load(os.path.join(r"C:\Learning_Mean_Activity\WT_Post_Learning\Mean_Activity", "Group_Vis_2_Activity.npy"))


    # View Comparisons
    visualise_activity_comparison(pre_learning_vis_1, pre_learning_vis_2, r"C:\Learning_Mean_Activity\WT_Pre_Learning", start_window, stop_window)
    visualise_activity_comparison(int_learning_vis_1, int_learning_vis_2, r"C:\Learning_Mean_Activity\WT_Intermediate_Learning", start_window, stop_window)
    visualise_activity_comparison(post_learning_vis_1, post_learning_vis_2, r"C:\Learning_Mean_Activity\WT_Post_Learning", start_window, stop_window)




# Select Analysis Details
frame_period = 37
start_window_ms = -1000
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
output_root = r"C:\Learning_Mean_Activity"

control_pre_learning_list = Session_List.control_pre_learning
control_intermediate_learning_list = Session_List.control_intermediate_learning
control_post_learning_list = Session_List.control_all_post_learning

learning_mean_activity_pipeline(data_root,
                                control_pre_learning_list,
                                control_intermediate_learning_list,
                                control_post_learning_list,
                                output_root,
                                start_window,
                                stop_window)