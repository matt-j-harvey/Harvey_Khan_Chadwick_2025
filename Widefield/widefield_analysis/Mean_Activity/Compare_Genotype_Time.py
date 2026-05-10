import os
import numpy as np
from scipy import stats
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
    #Plotting_Functions.visualise_mean_regressor(group_1_mean, save_directory_list[0], x_values, magnitude=[0, 2], cmap=plt.get_cmap("inferno"))
    #Plotting_Functions.visualise_mean_regressor(group_2_mean, save_directory_list[1], x_values, magnitude=[0, 2], cmap=plt.get_cmap("inferno"))
    #Plotting_Functions.visualise_mean_regressor(difference, save_directory_list[2],   x_values, magnitude=[-1, 1], cmap=Mean_Activity_Utils.get_musall_cmap())

    # View Mean Window
    mean_save_directory = os.path.join(output_root, "Mean_Windows")
    if not os.path.exists(mean_save_directory):
        os.makedirs(mean_save_directory)

    window_size = 28
    group_1_mean_window = np.mean(group_1_mean[np.abs(start_window):np.abs(start_window) + window_size], axis=0)
    group_2_mean_window = np.mean(group_2_mean[np.abs(start_window):np.abs(start_window) + window_size], axis=0)
    diff_mean_window = np.mean(difference[np.abs(start_window):np.abs(start_window) + window_size], axis=0)

    Plotting_Functions.visualise_single_timepoint(group_1_mean_window, mean_save_directory, "Vis_1_Window" ,magnitude=[0, 1.5], cmap=plt.get_cmap("inferno"))
    Plotting_Functions.visualise_single_timepoint(group_2_mean_window, mean_save_directory, "Vis_2_Window" ,magnitude=[0, 1.5], cmap=plt.get_cmap("inferno"))
    Plotting_Functions.visualise_single_timepoint(diff_mean_window, mean_save_directory, "Diff_Window", magnitude=[-0.2, 0.2], cmap=Mean_Activity_Utils.get_musall_cmap())



def sanity_check_mice(control_vis_1, control_vis_2):
    n_mice = len(control_vis_1)

    for mouse_index in range(n_mice):
        plt.plot(control_vis_1[mouse_index], c='g')
        plt.plot(control_vis_2[mouse_index], c='r')

        plt.show()



def plot_roi_activity(control_vis_1, control_vis_2, neurexin_vis_1, neurexin_vis_2, save_directory, start_window, stop_window, roi_list, graph_name):

    # Load Atlas
    atlas = Mean_Activity_Utils.load_atlas()
    atlas = np.abs(atlas)
    plt.imshow(atlas)
    plt.show()

    # Get Mean in ROI5
    control_vis_1 = Plotting_Functions.get_roi_mean(control_vis_1, atlas, roi_list)
    control_vis_2 = Plotting_Functions.get_roi_mean(control_vis_2, atlas, roi_list)
    neurexin_vis_1 = Plotting_Functions.get_roi_mean(neurexin_vis_1, atlas, roi_list)
    neurexin_vis_2 = Plotting_Functions.get_roi_mean(neurexin_vis_2, atlas, roi_list)
    print("control_vis_1", np.shape(control_vis_1))
    print("control vis 2", np.shape(control_vis_2))

    #sanity_check_mice(control_vis_1, control_vis_2)

    # Get Mean and SD
    control_vis_1_mean, control_vis_1_lower_bound, control_vis_1_upper_bound = Mean_Activity_Utils.get_mean_sd(control_vis_1)
    control_vis_2_mean, control_vis_2_lower_bound, control_vis_2_upper_bound = Mean_Activity_Utils.get_mean_sd(control_vis_2)
    neurexin_vis_1_mean, neurexin_vis_1_lower_bound, neurexin_vis_1_upper_bound = Mean_Activity_Utils.get_mean_sd(neurexin_vis_1)
    neurexin_vis_2_mean, neurexin_vis_2_lower_bound, neurexin_vis_2_upper_bound = Mean_Activity_Utils.get_mean_sd(neurexin_vis_2)

    max_value = np.max(np.concatenate([control_vis_1_upper_bound, control_vis_2_upper_bound, neurexin_vis_1_upper_bound, neurexin_vis_2_upper_bound]))
    max_value = max_value * 1.1

    # Get Signficance
    control_t_stats, control_p_values = stats.ttest_ind(control_vis_1, control_vis_2, axis=0)
    print("control_p_values", control_p_values)

    control_binary_signficance = np.where(control_p_values < 0.05, 1, 0)
    control_signficance_markers = np.multiply(control_binary_signficance, max_value * 1.1)

    neurexin_t_stats, neurexin_p_values = stats.ttest_ind(neurexin_vis_1, neurexin_vis_2, axis=0)
    neurexin_binary_significance = np.where(neurexin_p_values < 0.05, 1, 0)
    neurexin_signficance_markers = np.multiply(neurexin_binary_significance, max_value * 1.2)




    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 37)

    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    neurexin_pink = (0.76, 0.17, 0.99)
    axis_1.plot(x_values, control_vis_1_mean, c='royalblue')
    axis_1.plot(x_values, control_vis_2_mean, c='royalblue', linestyle="dashed")
    axis_1.plot(x_values, neurexin_vis_1_mean, c=neurexin_pink)
    axis_1.plot(x_values, neurexin_vis_2_mean, c=neurexin_pink, linestyle="dashed")

    axis_1.fill_between(x=x_values, y1=control_vis_1_lower_bound, y2=control_vis_1_upper_bound, alpha=0.3, color="royalblue")
    axis_1.fill_between(x=x_values, y1=control_vis_2_lower_bound, y2=control_vis_2_upper_bound, alpha=0.3, color="royalblue")
    axis_1.fill_between(x=x_values, y1=neurexin_vis_1_lower_bound, y2=neurexin_vis_1_upper_bound, alpha=0.3, color=neurexin_pink)
    axis_1.fill_between(x=x_values, y1=neurexin_vis_2_lower_bound, y2=neurexin_vis_2_upper_bound, alpha=0.3, color=neurexin_pink)

    #axis_1.scatter(x_values, control_signficance_markers, alpha=control_binary_signficance, color='royalblue', marker='s')
    #axis_1.scatter(x_values, neurexin_signficance_markers, alpha=neurexin_binary_significance, color=neurexin_pink, marker='s')


    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Z Score dF/F")
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.savefig(os.path.join(save_directory, graph_name + ".png"))
    #plt.show()
    plt.close()


def plot_roi_activity_difference(control_vis_1, control_vis_2, neurexin_vis_1, neurexin_vis_2, save_directory, start_window, stop_window, roi_list, graph_name):
        # Load Atlas
        atlas = Mean_Activity_Utils.load_atlas()
        atlas = np.abs(atlas)
        plt.imshow(atlas)
        plt.show()

        # Get Mean in ROI5
        control_vis_1 = Plotting_Functions.get_roi_mean(control_vis_1, atlas, roi_list)
        control_vis_2 = Plotting_Functions.get_roi_mean(control_vis_2, atlas, roi_list)
        neurexin_vis_1 = Plotting_Functions.get_roi_mean(neurexin_vis_1, atlas, roi_list)
        neurexin_vis_2 = Plotting_Functions.get_roi_mean(neurexin_vis_2, atlas, roi_list)

        # Get Difference
        control_difference = np.subtract(control_vis_1, control_vis_2)
        neurexin_difference = np.subtract(neurexin_vis_1, neurexin_vis_2)

        # Get Mean and SD
        control_mean, control_lower_bound, control_upper_bound = Mean_Activity_Utils.get_mean_sd(control_difference)
        neurexin_mean, neurexin_lower_bound, neurexin_upper_bound = Mean_Activity_Utils.get_mean_sd(neurexin_difference)

        max_value = np.max(np.concatenate([control_upper_bound, neurexin_upper_bound]))
        max_value = max_value * 1.1

        # Get Signficance
        t_stats, p_values = stats.ttest_ind(control_difference, neurexin_difference, axis=0)
        binary_signficance = np.where(p_values < 0.05, 1, 0)
        signficance_markers = np.multiply(binary_signficance, max_value * 1.1)

        # Get X Values
        x_values = list(range(start_window, stop_window))
        x_values = np.multiply(x_values, 37)

        figure_1 = plt.figure(figsize=(10, 5))
        axis_1 = figure_1.add_subplot(1, 1, 1)

        neurexin_pink = (0.76, 0.17, 0.99)
        axis_1.plot(x_values, control_mean, c='royalblue')
        axis_1.plot(x_values, neurexin_mean, c=neurexin_pink)

        axis_1.fill_between(x=x_values, y1=control_lower_bound, y2=control_upper_bound, alpha=0.3, color="royalblue")
        axis_1.fill_between(x=x_values, y1=neurexin_lower_bound, y2=neurexin_upper_bound, alpha=0.3, color=neurexin_pink)

        axis_1.scatter(x_values, signficance_markers, alpha=binary_signficance, color='royalblue', marker='s')

        axis_1.axvline(0, c='k', linestyle='dashed')
        axis_1.set_xlabel("Time (ms)")
        axis_1.set_ylabel("Z Score dF/F")
        axis_1.spines[['right', 'top']].set_visible(False)

        plt.savefig(os.path.join(save_directory, graph_name + ".png"))
        plt.close()


def compare_mean_activity_pipeline(data_root,
                                    session_list,
                                    output_root,
                                    learning_time,
                                    start_window,
                                    stop_window):

    # Get Group Mean Activity
    get_group_mean_activity(data_root, session_list, os.path.join(output_root, learning_time), start_window, stop_window)

    # Load Activity
    vis_1 = np.load(os.path.join(output_root, learning_time, "Mean_Activity", "Group_Vis_1_Activity.npy"))
    vis_2 = np.load(os.path.join(output_root, learning_time, "Mean_Activity", "Group_Vis_2_Activity.npy"))

    # View Comparisons
    visualise_activity_comparison(vis_1, vis_2, os.path.join(output_root, learning_time, "Activity Maps"), start_window, stop_window)


def compare_genotype_rois(output_root,
                          learning_time,
                          graph_name,
                          start_window,
                          stop_window):

    # Load Activity
    control_vis_1 = np.load(os.path.join(output_root, "Control_" + learning_time, "Mean_Activity", "Group_Vis_1_Activity.npy"))
    control_vis_2 = np.load(os.path.join(output_root, "Control_" + learning_time, "Mean_Activity", "Group_Vis_2_Activity.npy"))
    neurexin_vis_1 = np.load(os.path.join(output_root, "Neurexin_" + learning_time, "Mean_Activity", "Group_Vis_1_Activity.npy"))
    neurexin_vis_2 = np.load(os.path.join(output_root, "Neurexin_" + learning_time, "Mean_Activity", "Group_Vis_2_Activity.npy"))

    # Plot ROIs
    save_directory = os.path.join(output_root, "ROI_Plots")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plot_roi_activity(control_vis_1, control_vis_2, neurexin_vis_1, neurexin_vis_2, save_directory, start_window, stop_window, [14, 15, 16], graph_name)
    #plot_roi_activity_difference(control_vis_1, control_vis_2, neurexin_vis_1, neurexin_vis_2, save_directory, start_window, stop_window, [14, 15, 16], graph_name)


# Select Analysis Details
frame_period = 37
start_window_ms = -1000
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
neurexin_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"

output_root = r"C:\Learning_Mean_Activity\Time_Matched"


# Assign Session Lists
control_early_list = Session_List.control_early_list
control_mid_list = Session_List.control_mid_list
control_late_list = Session_List.control_late_list

neurexin_early_list = Session_List.neurexin_early_list
neurexin_mid_list = Session_List.neurexin_mid_list
neurexin_late_list = Session_List.neurexin_late_list


"""
compare_mean_activity_pipeline(control_data_root, control_early_list, output_root, "Control_Early", start_window, stop_window)
compare_mean_activity_pipeline(control_data_root, control_mid_list, output_root, "Control_Mid", start_window, stop_window)
compare_mean_activity_pipeline(control_data_root, control_late_list, output_root, "Control_Late", start_window, stop_window)

compare_mean_activity_pipeline(neurexin_data_root, neurexin_early_list, output_root, "Neurexin_Early", start_window, stop_window)
compare_mean_activity_pipeline(neurexin_data_root, neurexin_mid_list, output_root, "Neurexin_Mid", start_window, stop_window)
compare_mean_activity_pipeline(neurexin_data_root, neurexin_late_list, output_root, "Neurexin_Late", start_window, stop_window)
"""

compare_genotype_rois(output_root,"Early","M2_Activity_Early", start_window, stop_window)
compare_genotype_rois(output_root,"Mid","M2_Activity_Mid", start_window, stop_window)
compare_genotype_rois(output_root,"Late","M2_Activity_Late", start_window, stop_window)

