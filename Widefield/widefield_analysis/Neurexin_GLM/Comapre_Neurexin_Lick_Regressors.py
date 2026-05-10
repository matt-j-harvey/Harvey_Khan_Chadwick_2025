import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Session_List
import GLM_Utils
import Plotting_Functions


def get_group_lick_coefs(data_root, session_list, output_root, start_window, stop_window):

    group_list = []
    for mouse in session_list:
        mouse_list = []

        for session in mouse:
            session_coefs = get_session_lick_coefs(data_root, session, output_root, start_window, stop_window)

            """
            session_baseline = session_coefs[0:3]
            session_baseline = np.mean(session_baseline, axis=0)
            session_coefs = np.subtract(session_coefs, session_baseline)
            """
            mouse_list.append(session_coefs)

        mouse_list = np.array(mouse_list)
        mouse_mean = np.mean(mouse_list, axis=0)
        group_list.append(mouse_mean)

    group_list = np.array(group_list)

    save_directory = os.path.join(output_root, "Group_Coefs")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Group_Lick_Coefs.npy"), group_list)

    """
    group_mean = np.mean(group_list, axis=0)
    plt.ion()
    count = 0
    for frame in group_mean:
        plt.imshow(frame, cmap='plasma', vmin=0, vmax=1)
        plt.title(str(count))
        count += 1
        plt.pause(0.1)
        plt.draw()
        plt.pause(0.1)
        plt.clf()
    """

def get_session_lick_coefs(data_root, session, output_root, start_window, stop_window):

    # Load Model Coefs
    model_coefs = np.load(os.path.join(output_root, session, "Model_Output", "Model_Coefs.npy"))
    print("model_coefs", np.shape(model_coefs))

    # Get Lick Coefs:
    n_timepoints = stop_window - start_window
    print("n_timepoints", n_timepoints)

    n_lags = int(np.around(1500 / 36, 0)) * 2
    print("n_lags", n_lags)
    behaviour_coefs = model_coefs[:, n_timepoints * 2:]
    lick_coefs = behaviour_coefs[:, 1:1 + n_lags]
    lick_coefs = np.transpose(lick_coefs)
    print("lick_coefs", np.shape(lick_coefs))

    # Reconstruct Into Pixel Space
    lick_coefs = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, lick_coefs)

    return lick_coefs


def view_regressors(control_output_root, neurexin_output_root, results_root):

    # Get X Values
    n_lags = int(np.around(1500 / 36, 0))
    x_values = list(range(-n_lags, n_lags))
    current_x_values = np.multiply(x_values, 36)

    # Load Coefs
    control_lick_coefs = np.load(os.path.join(control_output_root, "Group_Coefs", "Group_Lick_Coefs.npy"))
    neurexin_lick_coefs = np.load(os.path.join(neurexin_output_root, "Group_Coefs", "Group_Lick_Coefs.npy"))

    graph_save_directory = os.path.join(results_root, "ROI_Graphs")
    if not os.path.exists(graph_save_directory):
        os.makedirs(graph_save_directory)

    print("control_lick_coefs", np.shape(control_lick_coefs))
    print("neurexin_lick_coefs", np.shape(neurexin_lick_coefs))
    Plotting_Functions.compare_roi_across_groups(control_lick_coefs, neurexin_lick_coefs, graph_save_directory, -n_lags, n_lags, [14,15,16], "Lick_Regressor")


    # Get Mean Coefs
    control_mean = np.mean(control_lick_coefs, axis=0)
    neurexin_mean = np.mean(neurexin_lick_coefs, axis=0)
    mean_diff = np.subtract(neurexin_mean, control_mean)

    """
    print("Getting sig diff")
    t_stats, p_values = stats.ttest_ind(control_lick_coefs, neurexin_lick_coefs, axis=0)
    p_values = np.nan_to_num(p_values, nan=1)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    mean_diff = np.multiply(binary_sig, mean_diff)
    """

    # Linearly Interpolate
    control_mean, new_x_values = GLM_Utils.linearly_interpolate(control_mean, 36, 10, current_x_values, 2)
    neurexin_mean, new_x_values = GLM_Utils.linearly_interpolate(neurexin_mean, 36, 10, current_x_values, 2)
    mean_diff, new_x_values = GLM_Utils.linearly_interpolate(mean_diff, 36, 10, current_x_values, 2)

    print("new x values", new_x_values)

    # Create Save Directories
    save_directory_list = [
        os.path.join(results_root, "Group_Regressor_Maps", "Control_Lick"),
        os.path.join(results_root, "Group_Regressor_Maps", "Neurexin_Lick"),
        os.path.join(results_root, "Group_Regressor_Maps", "Lick_Difference"),
    ]

    for directory in save_directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    Plotting_Functions.visualise_mean_regressor(control_mean,  save_directory_list[0], new_x_values, magnitude=[0, 0.6], cmap=plt.get_cmap('inferno'))
    Plotting_Functions.visualise_mean_regressor(neurexin_mean, save_directory_list[1], new_x_values, magnitude=[0, 0.6], cmap=plt.get_cmap('inferno'))
    Plotting_Functions.visualise_mean_regressor(mean_diff,     save_directory_list[2], new_x_values, magnitude=[-0.3, 0.3], cmap=GLM_Utils.get_musall_cmap())


# Select Analysis Details
frame_period = 37
start_window_ms = -2800
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)
regressor_list = ["vis_1_correct", "vis_2_correct"]


# Set Directories
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"
#control_session_list = Session_List.control_all_post_learning
control_session_list = Session_List.control_post_learning_discrimination

hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Neurexin_GLM\Post_Learning\Homs"
#hom_session_list = Session_List.neurexin_all_post_learning
hom_session_list = Session_List.neurexin_post_learning_discrimination

results_root = r"C:\Analysis_Output\Neurexin_GLM\Post_Learning_Discrimination_Only\Group_Results"


#get_group_lick_coefs(control_data_root, control_session_list, control_output_root, start_window, stop_window)
#get_group_lick_coefs(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window)
view_regressors(control_output_root, hom_output_root, results_root)
