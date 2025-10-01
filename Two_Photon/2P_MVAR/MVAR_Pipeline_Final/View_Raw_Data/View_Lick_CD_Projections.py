import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import Get_Data_Tensor



def get_lick_cd_projection(data_root, session, mvar_output_root, onset_file, start_window, stop_window):

    # Load dF/F
    df_matrix = np.load(os.path.join(mvar_output_root, session, "df_over_f_matrix.npy"))

    # Load Onsets
    onset_list = np.load(os.path.join(data_root, session, "Stimuli_Onsets", onset_file + "_onsets.npy"))

    # Get Data Tensor
    data_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

    # Get Mean Response
    mean_response = np.mean(data_tensor, axis=0)

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_output_root, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))

    # Create Save Directory
    save_directory = os.path.join(mvar_output_root, session, "Raw Data Visualisation", "Lick_CD_Projections")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    lick_cd_projection = np.dot(mean_response, lick_cd)
    np.save(os.path.join(save_directory, onset_file + "_lick_cd_projection.npy"), lick_cd_projection)




def plot_all_lick_cds(data_root, session, mvar_output_root, start_window, stop_window):

    lick_cd_directory = os.path.join(mvar_output_root, session, "Raw Data Visualisation", "Lick_CD_Projections")
    visual_context_vis_1_lick_cd = np.load(os.path.join(lick_cd_directory, "visual_context_stable_vis_1_lick_cd_projection.npy"))
    visual_context_vis_2_lick_cd = np.load(os.path.join(lick_cd_directory, "visual_context_stable_vis_2_lick_cd_projection.npy"))
    odour_context_vis_1_lick_cd = np.load(os.path.join(lick_cd_directory, "odour_context_stable_vis_1_lick_cd_projection.npy"))
    odour_context_vis_2_lick_cd = np.load(os.path.join(lick_cd_directory, "odour_context_stable_vis_2_lick_cd_projection.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    period = 1.0/frame_rate

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, period)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(x_values, visual_context_vis_1_lick_cd, c='b')
    axis_1.plot(x_values, visual_context_vis_2_lick_cd, c='r')
    axis_1.plot(x_values, odour_context_vis_1_lick_cd, c='g')
    axis_1.plot(x_values, odour_context_vis_2_lick_cd, c='m')

    axis_1.axvline(0, c='k', linestyle='dashed')
    plt.savefig(os.path.join(lick_cd_directory, "Lick_CD_Projections.png"))
    plt.close()


def get_group_cd(session_list, mvar_output_root, condition_name):

    group_projection_list = []
    for session in session_list:
        session_projection = np.load(os.path.join(mvar_output_root, session,  "Raw Data Visualisation", "Lick_CD_Projections", condition_name + "_lick_cd_projection.npy"))
        group_projection_list.append(session_projection)

    group_projection_list = np.array(group_projection_list)
    mean_group_projection = np.mean(group_projection_list, axis=0)

    group_sem = stats.sem(group_projection_list, axis=0)
    lower_bound = np.subtract(mean_group_projection, group_sem)
    upper_bound = np.add(mean_group_projection, group_sem)

    return mean_group_projection, lower_bound, upper_bound



def view_group_lick_cd(data_root, session_list, mvar_output_root, start_window, stop_window):

    visual_context_vis_1_mean, visual_context_vis_1_lower_bound, visual_context_vis_1_upper_bound = get_group_cd(session_list, mvar_output_root,"visual_context_stable_vis_1")
    visual_context_vis_2_mean, visual_context_vis_2_lower_bound, visual_context_vis_2_upper_bound = get_group_cd(session_list, mvar_output_root,"visual_context_stable_vis_2")
    odour_context_vis_1_mean, odour_context_vis_1_lower_bound, odour_context_vis_1_upper_bound = get_group_cd(session_list, mvar_output_root,"odour_context_stable_vis_1")
    odour_context_vis_2_mean, odour_context_vis_2_lower_bound, odour_context_vis_2_upper_bound = get_group_cd(session_list, mvar_output_root,"odour_context_stable_vis_2")

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session_list[0], "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, period)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(x_values, visual_context_vis_1_mean, c='b')
    axis_1.fill_between(x=x_values, y1=visual_context_vis_1_lower_bound, y2=visual_context_vis_1_upper_bound, alpha=0.4, color='b')

    axis_1.plot(x_values, visual_context_vis_2_mean, c='r')
    axis_1.fill_between(x=x_values, y1=visual_context_vis_2_lower_bound, y2=visual_context_vis_2_upper_bound, alpha=0.4, color='r')

    axis_1.plot(x_values, odour_context_vis_1_mean, c='g')
    axis_1.fill_between(x=x_values, y1=odour_context_vis_1_lower_bound, y2=odour_context_vis_1_upper_bound, alpha=0.4, color='g')

    axis_1.plot(x_values, odour_context_vis_2_mean, c='m')
    axis_1.fill_between(x=x_values, y1=odour_context_vis_2_lower_bound, y2=odour_context_vis_2_upper_bound, alpha=0.4, color='m')

    axis_1.axvline(0, c='k', linestyle='dashed')

    # Cfreate Save Directory
    save_directory = os.path.join(mvar_output_root, "Group_Results", "Raw_Data_Visualisation")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Group_Lick_CD_Projections.png"))
    plt.close()