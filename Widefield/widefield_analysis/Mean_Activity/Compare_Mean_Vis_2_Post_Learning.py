import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from widefield_analysis.Utils import Session_List
"""
import Session_List
import Mean_Activity_Utils
import Plotting_Functions
"""



def get_session_mean_activity(data_root, session, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Onsets
    vis_2_onsets = Mean_Activity_Utils.get_cr_onsets(behaviour_matrix)

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
    activity_matrix = np.transpose(activity_matrix)
    print("activity_matrix", np.shape(activity_matrix))

    # Get Data Tensors
    vis_2_tensor = Mean_Activity_Utils.get_data_tensor(activity_matrix, vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)

    # Get Trial Mean
    vis_2_mean = np.mean(vis_2_tensor, axis=0)

    # Reconstruct Into Pixel Space
    vis_2_mean = Mean_Activity_Utils.reconstruct_regressor_into_pixel_space(data_root, session, vis_2_mean)

    return vis_2_mean



def get_group_mean_activity(data_root, session_list, output_root, start_window, stop_window):

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

    save_directory = os.path.join(output_root, "Mean_Activity")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Group_Vis_2_Activity.npy"), group_vis_2_list)




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








def post_learning_activity_pipeline(control_data_root,
                                    neurexin_data_root,
                                    control_session_list,
                                    neurexin_session_list,
                                    output_root,
                                    start_window,
                                    stop_window):


    # Get Group Activity
    get_group_mean_activity(control_data_root, control_session_list, os.path.join(output_root, "Wildtype"), start_window, stop_window)
    get_group_mean_activity(neurexin_data_root, neurexin_session_list,  os.path.join(output_root, "Hom"), start_window, stop_window)

    # Load Activity
    control_vis_2 = np.load(os.path.join(output_root, "Wildtype", "Mean_Activity", "Group_Vis_2_Activity.npy"))
    hom_vis_2 = np.load(os.path.join(output_root, "Hom", "Mean_Activity", "Group_Vis_2_Activity.npy"))

    # Plot M2 Activity
    plot_roi_activity(control_vis_2, hom_vis_2, output_root, start_window, stop_window, [14, 15, 16], "Genotype M2 Activity")

    # Visualise Activity Maps
    visualise_activity_comparison(control_vis_2, hom_vis_2, output_root, start_window, stop_window)



# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

wt_session_list = Session_List.control_post_learning_discrimination
nx_session_list = Session_List.neurexin_post_learning_discrimination

wt_session_list = Session_List.control_intermediate_learning
nx_session_list = Session_List.neurexin_intermediate_learning

control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
neurexin_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
#output_root = r"C:\Learning_Mean_Activity\Post_Learning_Discrimination_Only\Genotype_CR"

output_root = r"C:\Learning_Mean_Activity\Int_Learning_Discrimination\Genotype_CR"

post_learning_activity_pipeline(control_data_root,
                                neurexin_data_root,
                                wt_session_list,
                                nx_session_list,
                                output_root,
                                start_window,
                                stop_window)


