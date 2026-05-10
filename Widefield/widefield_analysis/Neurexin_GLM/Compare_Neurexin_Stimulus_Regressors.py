import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Session_List
import GLM_Utils
import Plotting_Functions


def get_group_vis_coefs(data_root, session_list, output_root, start_window, stop_window):

    group_vis_1_list = []
    group_vis_2_list = []

    for mouse in session_list:
        mouse_vis_1_list = []
        mouse_vis_2_list = []

        for session in mouse:
            vis_1_coefs, vis_2_coefs = get_session_stim_coefs(data_root, session, output_root, start_window, stop_window)
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

    save_directory = os.path.join(output_root, "Group_Coefs")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Group_Vis_1_Coefs.npy"), group_vis_1_list)
    np.save(os.path.join(save_directory, "Group_Vis_2_Coefs.npy"), group_vis_2_list)

   

def get_session_stim_coefs(data_root, session, output_root, start_window, stop_window):

    # Load Model Coefs
    model_coefs = np.load(os.path.join(output_root, session, "Model_Output", "Model_Coefs.npy"))
    print("model_coefs", np.shape(model_coefs))

    n_timepoints = stop_window - start_window
    vis_1_coefs = model_coefs[:, 0:n_timepoints]
    vis_2_coefs = model_coefs[:, n_timepoints:2*n_timepoints]
    vis_1_coefs = np.transpose(vis_1_coefs)
    vis_2_coefs = np.transpose(vis_2_coefs)

    # Reconstruct Into Pixel Space
    vis_1_coefs = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, vis_1_coefs)
    vis_2_coefs = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, vis_2_coefs)

    return vis_1_coefs, vis_2_coefs




def view_regressors(control_output_root, neurexin_output_root, start_window, stop_window, results_root):

    # Get X Values
    x_values = list(range(start_window, stop_window))
    current_x_values = np.multiply(x_values, 36)

    # Load Coefs
    control_vis_1 = np.load(os.path.join(control_output_root, "Group_Coefs", "Group_Vis_1_Coefs.npy"))
    control_vis_2 = np.load(os.path.join(control_output_root, "Group_Coefs", "Group_Vis_2_Coefs.npy"))
    neurexin_vis_1 = np.load(os.path.join(neurexin_output_root, "Group_Coefs", "Group_Vis_1_Coefs.npy"))
    neurexin_vis_2 = np.load(os.path.join(neurexin_output_root, "Group_Coefs", "Group_Vis_2_Coefs.npy"))

    # Get Mean Coefs
    control_vis_1_mean = np.mean(control_vis_1, axis=0)
    control_vis_2_mean = np.mean(control_vis_2, axis=0)
    neurexin_vis_1_mean = np.mean(neurexin_vis_1, axis=0)
    neurexin_vis_2_mean = np.mean(neurexin_vis_2, axis=0)

    # Linearly Interpolate
    #control_mean, new_x_values = GLM_Utils.linearly_interpolate(control_mean, 36, 10, current_x_values, 2)
    #neurexin_mean, new_x_values = GLM_Utils.linearly_interpolate(neurexin_mean, 36, 10, current_x_values, 2)
    #mean_diff, new_x_values = GLM_Utils.linearly_interpolate(mean_diff, 36, 10, current_x_values, 2)
    #print("new x values", new_x_values)

    # Create Save Directories
    save_directory_list = [
        os.path.join(results_root, "Group_Regressor_Maps", "control_vis_1_mean"),
        os.path.join(results_root, "Group_Regressor_Maps", "control_vis_2_mean"),
        os.path.join(results_root, "Group_Regressor_Maps", "neurexin_vis_1_mean"),
        os.path.join(results_root, "Group_Regressor_Maps", "neurexin_vis_2_mean"),
    ]

    for directory in save_directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    Plotting_Functions.visualise_mean_regressor(control_vis_1_mean,  save_directory_list[0], current_x_values, magnitude=[0, 1], cmap=plt.get_cmap('inferno'))
    Plotting_Functions.visualise_mean_regressor(control_vis_2_mean, save_directory_list[1], current_x_values, magnitude=[0, 1], cmap=plt.get_cmap('inferno'))
    Plotting_Functions.visualise_mean_regressor(neurexin_vis_1_mean, save_directory_list[2], current_x_values, magnitude=[0, 1], cmap=plt.get_cmap('inferno'))
    Plotting_Functions.visualise_mean_regressor(neurexin_vis_2_mean, save_directory_list[3], current_x_values, magnitude=[0, 1], cmap=plt.get_cmap('inferno'))


# Select Analysis Details
frame_period = 37
start_window_ms = -2800
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)
regressor_list = ["vis_1_correct", "vis_2_correct"]

# Set Directories
control_session_list = Session_List.control_all_post_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"

hom_session_list = Session_List.neurexin_all_post_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Neurexin_GLM\Post_Learning\Homs"


results_root = r"C:\Analysis_Output\Neurexin_GLM\Post_Learning\Group_Results"


#get_group_vis_coefs(control_data_root, control_session_list, control_output_root, start_window, stop_window)
#get_group_vis_coefs(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window)

view_regressors(control_output_root, hom_output_root, start_window, stop_window, results_root)
