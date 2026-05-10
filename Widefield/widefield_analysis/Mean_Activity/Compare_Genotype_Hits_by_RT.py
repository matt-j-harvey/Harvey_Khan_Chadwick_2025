import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Session_List
import Mean_Activity_Utils
import Plotting_Functions



def get_session_mean_activity(data_root, session, start_window, stop_window, rt_window_start, rt_window_stop):

    # Load Onsets
    onsets = Mean_Activity_Utils.get_hit_onsets_rt_window(data_root, session, rt_window_start, rt_window_stop)

    if len(onsets) > 1:

        # Load Activity Matrix
        activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
        activity_matrix = np.transpose(activity_matrix)

        # Get Data Tensors
        activity_tensor = Mean_Activity_Utils.get_data_tensor(activity_matrix, onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)
        print("activity_tensor", np.shape(activity_tensor))


        if np.shape(activity_tensor)[0] > 1:

            # Get Trial Mean
            activity_mean = np.mean(activity_tensor, axis=0)

            # Reconstruct Into Pixel Space
            activity_mean = Mean_Activity_Utils.reconstruct_regressor_into_pixel_space(data_root, session, activity_mean)

            return activity_mean



def get_group_mean_activity(data_root, session_list, start_window, stop_window, rt_window_start, rt_window_stop):

    group_list = []
    for mouse in session_list:
        mouse_list = []

        for session in mouse:
            print(session)
            session_activity = get_session_mean_activity(data_root, session, start_window, stop_window, rt_window_start, rt_window_stop)

            if isinstance(session_activity, np.ndarray):
                mouse_list.append(session_activity)
                print("session_activity", np.shape(session_activity))

        if len(mouse_list) == 1:
            mouse_mean = mouse_list[0]
            group_list.append(mouse_mean)

        elif len(mouse_list) > 1:
            mouse_list = np.array(mouse_list)
            mouse_mean = np.mean(mouse_list, axis=0)
            group_list.append(mouse_mean)

    group_list = np.array(group_list)
    return group_list



def visualise_activity_comparison(group_1_activity, group_2_activity, output_root, start_window, stop_window):

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 37)

    # Get Means and Diffs
    group_1_mean = np.mean(group_1_activity, axis=0)
    group_2_mean = np.mean(group_2_activity, axis=0)
    difference = np.subtract(group_2_mean, group_1_mean)

    # Create Save Directories
    save_directory_list = [os.path.join(output_root, "Wildtype_CR_Activity_Map"),
                           os.path.join(output_root, "Neurexin_CR_Activity_Map"),
                           os.path.join(output_root, "Genotype_CR_Diff_Activity_Map"),]

    for save_directory in save_directory_list:
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

    # Plot These
    Plotting_Functions.visualise_mean_regressor(group_1_mean, save_directory_list[0], x_values, magnitude=[0, 1.2], cmap=plt.get_cmap("inferno"))
    Plotting_Functions.visualise_mean_regressor(group_2_mean, save_directory_list[1], x_values, magnitude=[0, 1.2], cmap=plt.get_cmap("inferno"))
    Plotting_Functions.visualise_mean_regressor(difference, save_directory_list[2],   x_values, magnitude=[-0.8, 0.8], cmap=Mean_Activity_Utils.get_musall_cmap())




def plot_roi_activity(control, neurexin, save_directory, start_window, stop_window, roi_list, graph_name, rt_window_start, rt_window_stop):

    # Load Atlas
    atlas = Mean_Activity_Utils.load_atlas()
    atlas = np.abs(atlas)

    # Get Mean in ROI5
    control = Plotting_Functions.get_roi_mean(control, atlas, roi_list)
    neurexin = Plotting_Functions.get_roi_mean(neurexin, atlas, roi_list)

    # Get Mean and SD
    control_mean, control_lower_bound, control_upper_bound = Mean_Activity_Utils.get_mean_sd(control)
    neurexin_mean, neurexin_lower_bound, neurexin_upper_bound = Mean_Activity_Utils.get_mean_sd(neurexin)

    max_value = np.max(np.concatenate([control_upper_bound, neurexin_upper_bound]))
    max_value = max_value * 1.1

    # Get Signficance
    t_stats, p_values = stats.ttest_ind(control, neurexin, axis=0)
    binary_signficance = np.where(p_values < 0.05, 1, 0)
    signficance_markers = np.multiply(binary_signficance, max_value * 1.1)

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    figure_1 = plt.figure(figsize=(5,5))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    neurexin_pink = (0.76, 0.17, 0.99)
    axis_1.plot(x_values, control_mean, c='royalblue')
    axis_1.plot(x_values, neurexin_mean, c=neurexin_pink)

    axis_1.fill_between(x=x_values, y1=control_lower_bound, y2=control_upper_bound, alpha=0.3, color="royalblue")
    axis_1.fill_between(x=x_values, y1=neurexin_lower_bound, y2=neurexin_upper_bound, alpha=0.3, color=neurexin_pink)

    axis_1.scatter(x_values, signficance_markers, alpha=binary_signficance, color='Grey', marker='s')

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Z Score dF/F")
    axis_1.spines[['right', 'top']].set_visible(False)

    # Add lick Line
    lick_delta = np.subtract(rt_window_stop, rt_window_start)
    lick_time = np.add(rt_window_start, lick_delta)
    axis_1.axvline(lick_time, c='k', linestyle='dashed')

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, graph_name + ".svg"))
    #plt.show()
    plt.close()





def post_learning_activity_pipeline(control_data_root,
                                    neurexin_data_root,
                                    control_session_list,
                                    neurexin_session_list,
                                    output_root,
                                    start_window,
                                    stop_window,
                                    rt_window_start_list,
                                    rt_window_stop_list):

    n_rt_bins = len(rt_window_start_list)
    for bin_index in range(n_rt_bins):

        # Get Bin Start and Stop
        bin_start = rt_window_start_list[bin_index]
        bin_stop = rt_window_stop_list[bin_index]

        # Get Group Activity
        wt_activity = get_group_mean_activity(control_data_root, control_session_list, start_window, stop_window, bin_start, bin_stop)
        hom_activity = get_group_mean_activity(neurexin_data_root, neurexin_session_list, start_window, stop_window, bin_start, bin_stop)

        plot_roi_activity(wt_activity, hom_activity, output_root, start_window, stop_window, [14, 15, 16], str(bin_start) + "_" + str(bin_stop), bin_start, bin_stop)
        #get_group_mean_activity(neurexin_data_root, neurexin_session_list,  os.path.join(output_root, "Hom"), start_window, stop_window)


    # Visualise Activity Maps
    #visualise_activity_comparison(control_vis_2, hom_vis_2, output_root, start_window, stop_window)

    # Plot M2 Activity
    plot_roi_activity(control_vis_2, hom_vis_2, output_root, start_window, stop_window, [14, 15, 16], "Genotype M2 Activity")

#
#
# # Select Analysis Details
# frame_period = 36
# start_window_ms = -2800
# stop_window_ms = 2500
# start_window = int(start_window_ms/frame_period)
# stop_window = int(stop_window_ms/frame_period)
#
# rt_bin_starts = list(range(500, 2250, 250))
# rt_bin_stops = np.add(rt_bin_starts, 250)
#
# control_post_learning_list = Session_List.control_post_learning_discrimination
# neurexin_post_learning_list = Session_List.neurexin_post_learning_discrimination
#
# control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
# neurexin_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
# output_root = r"C:\Learning_Mean_Activity\Post_Learning_Discrimination_Only\Genotype_Hit_By_RT"
#
# post_learning_activity_pipeline(control_data_root,
#                                     neurexin_data_root,
#                                     control_post_learning_list,
#                                     neurexin_post_learning_list,
#                                     output_root,
#                                     start_window,
#                                     stop_window,
#                                     rt_bin_starts,
#                                     rt_bin_stops)
#


# Select Analysis Details
frame_period = 36
start_window_ms = -1000
stop_window_ms = 2000
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

rt_bin_starts = list(range(500, 2250, 250))
rt_bin_stops = np.add(rt_bin_starts, 250)

#control_post_learning_list = Session_List.control_pre_learning
#neurexin_post_learning_list = Session_List.neurexin_pre_learning_list

#control_post_learning_list = Session_List.control_intermediate_learning
#neurexin_post_learning_list = Session_List.neurexin_intermediate_learning

control_post_learning_list = Session_List.control_post_learning_discrimination
neurexin_post_learning_list = Session_List.neurexin_post_learning_discrimination

control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
neurexin_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
output_root = r"C:\Learning_Mean_Activity\Post_Learning_Discrimination_Only\Post_Genotype_Hit_By_RT"

post_learning_activity_pipeline(control_data_root,
                                    neurexin_data_root,
                                    control_post_learning_list,
                                    neurexin_post_learning_list,
                                    output_root,
                                    start_window,
                                    stop_window,
                                    rt_bin_starts,
                                    rt_bin_stops)