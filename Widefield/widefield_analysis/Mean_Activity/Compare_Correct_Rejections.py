
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from widefield_analysis.utils import session_list, widefield_utils, plotting_functions

"""
import Session_List
import Mean_Activity_Utils
import Plotting_Functions
"""



def get_session_mean_activity(data_root, session, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Onsets
    vis_2_onsets = widefield_utils.get_cr_onsets(behaviour_matrix)

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
    activity_matrix = np.transpose(activity_matrix)
    print("activity_matrix", np.shape(activity_matrix))

    # Get Data Tensors
    vis_2_tensor = widefield_utils.get_data_tensor(activity_matrix, vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)

    # Get Trial Mean
    vis_2_mean = np.mean(vis_2_tensor, axis=0)

    # Reconstruct Into Pixel Space
    vis_2_mean = widefield_utils.reconstruct_regressor_into_pixel_space(data_root, session, vis_2_mean)

    return vis_2_mean



def get_group_mean_activity(data_root, session_list, save_directory, filename, start_window, stop_window):

    group_vis_2_list = []
    for mouse in session_list:
        mouse_vis_2_list = []

        for session in mouse:
            vis_2_coefs = get_session_mean_activity(data_root, session, start_window, stop_window)
            mouse_vis_2_list.append(vis_2_coefs)

        mouse_vis_2_list = np.array(mouse_vis_2_list)
        mouse_vis_2_mean = np.mean(mouse_vis_2_list, axis=0)
        group_vis_2_list.append(mouse_vis_2_mean)


    group_vis_2_list = np.array(group_vis_2_list)

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, filename), group_vis_2_list)




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
    #Plotting_Functions.visualise_mean_regressor(group_1_mean, save_directory_list[0], x_values, magnitude=[0, 1.2], cmap=plt.get_cmap("inferno"))
    #Plotting_Functions.visualise_mean_regressor(group_2_mean, save_directory_list[1], x_values, magnitude=[0, 1.2], cmap=plt.get_cmap("inferno"))
    #Plotting_Functions.visualise_mean_regressor(difference, save_directory_list[2],   x_values, magnitude=[-0.8, 0.8], cmap=Mean_Activity_Utils.get_musall_cmap())








def cr_mean_activity_pipeline(wt_data_root, nx_data_root,
                            wt_int_session_list, nx_int_session_list,
                            wt_post_session_list, nx_post_session_list,
                            output_root,
                            start_window, stop_window):

    # Set Save Directory
    save_directory = os.path.join(output_root, "Mean_Activity")

    # Get Group Activity
    #get_group_mean_activity(wt_data_root, wt_int_session_list, save_directory, "wt_int_cr", start_window, stop_window)
    #get_group_mean_activity(wt_data_root, wt_post_session_list, save_directory, "wt_post_cr", start_window, stop_window)
    #get_group_mean_activity(nx_data_root, nx_int_session_list,  save_directory, "nx_int_cr", start_window, stop_window)
    #get_group_mean_activity(nx_data_root, nx_post_session_list, save_directory, "nx_post_cr", start_window, stop_window)

    # Load Activity
    wt_int_cr = np.load(os.path.join(save_directory, "wt_int_cr.npy"))
    wt_post_cr = np.load(os.path.join(save_directory, "wt_post_cr.npy"))
    nx_int_cr = np.load(os.path.join(save_directory, "nx_int_cr.npy"))
    nx_post_cr = np.load(os.path.join(save_directory, "nx_post_cr.npy"))
    print("wt_int_cr", np.shape(wt_int_cr))

    # Plot M2 Activity
    roi_list = [14, 15, 16]
    plotting_functions.plot_roi_activity(wt_int_cr, nx_int_cr, output_root, start_window, stop_window, roi_list, "Genotype M2 Activity Intermediate Learning", ylim=[-0.2, 1.6])
    plotting_functions.plot_roi_activity(wt_post_cr, nx_post_cr, output_root, start_window, stop_window, roi_list, "Genotype M2 Activity Post Learning", ylim=[-0.2, 1.6])

    # Scatter M2 Activity in Window

    comparison_window_start = np.abs(start_window)
    comparison_window_stop = comparison_window_start + 14
    plotting_functions.plot_roi_activity_genotype_learning(wt_int_cr,
                                        wt_post_cr,
                                        nx_int_cr,
                                        nx_post_cr,
                                        output_root,
                                        roi_list,
                                        comparison_window_start, comparison_window_stop,
                                        "M2 Mean Activity CR 500ms")


    #visualise_activity_comparison(control_vis_2, hom_vis_2, output_root, start_window, stop_window)



# Select Analysis Details
frame_period = 36
start_window_ms = -2800
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

wt_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
nx_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"

wt_int_session_list = session_list.control_intermediate_learning
nx_int_session_list = session_list.neurexin_intermediate_learning

wt_post_session_list = session_list.control_post_learning_discrimination
nx_post_session_list = session_list.neurexin_post_learning_discrimination

output_root = r"C:\Learning_Mean_Activity\Correct_Rejection_Mean_Activity"


cr_mean_activity_pipeline(wt_data_root, nx_data_root,
                            wt_int_session_list, nx_int_session_list,
                            wt_post_session_list, nx_post_session_list,
                            output_root,
                            start_window, stop_window)