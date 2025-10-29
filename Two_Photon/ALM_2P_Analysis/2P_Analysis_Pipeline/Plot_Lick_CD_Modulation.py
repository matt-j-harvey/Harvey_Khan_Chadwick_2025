import numpy as np
import os
import matplotlib.pyplot as plt

import ALM_Analysis_Utils



def plot_lick_cd_modulation(data_root, session, output_root):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(output_root, session, "df_over_f_matrix.npy"))

    # Load Lick CD
    lick_cd = np.load(os.path.join(output_root, session, "Lick_Coding", "Lick_Coding_Dimension.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))

    # Load Onsets
    vis_context_vis_1_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))
    vis_context_vis_2_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))
    odr_context_vis_1_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "odour_context_stable_vis_1_onsets.npy"))
    odr_context_vis_2_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "odour_context_stable_vis_2_onsets.npy"))

    # Get Tensors
    start_time = -2.8
    stop_time = 2.5
    start_window = int(start_time * frame_rate)
    stop_window = int(stop_time * frame_rate)

    vis_context_vis_1_tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix, vis_context_vis_1_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    vis_context_vis_2_tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix, vis_context_vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    odr_context_vis_1_tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix, odr_context_vis_1_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    odr_context_vis_2_tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix, odr_context_vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

    # Get Means
    vis_context_vis_1_mean = np.mean(vis_context_vis_1_tensor, axis=0)
    vis_context_vis_2_mean = np.mean(vis_context_vis_2_tensor, axis=0)
    odr_context_vis_1_mean = np.mean(odr_context_vis_1_tensor, axis=0)
    odr_context_vis_2_mean = np.mean(odr_context_vis_2_tensor, axis=0)

    # Get Lick CD Projections
    vis_context_vis_1_cd_projection = np.dot(vis_context_vis_1_mean, lick_cd)
    vis_context_vis_2_cd_projection = np.dot(vis_context_vis_2_mean, lick_cd)
    odr_context_vis_1_cd_projection = np.dot(odr_context_vis_1_mean, lick_cd)
    odr_context_vis_2_cd_projection = np.dot(odr_context_vis_2_mean, lick_cd)

    # Save These
    save_directory = os.path.join(output_root, session, "Contextual_Modulation_Lick_CD")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "vis_context_vis_1_cd_projection"), vis_context_vis_1_cd_projection)
    np.save(os.path.join(save_directory, "vis_context_vis_2_cd_projection"), vis_context_vis_2_cd_projection)
    np.save(os.path.join(save_directory, "odr_context_vis_1_cd_projection"), odr_context_vis_1_cd_projection)
    np.save(os.path.join(save_directory, "odr_context_vis_2_cd_projection"), odr_context_vis_2_cd_projection)
    np.save(os.path.join(save_directory, "start_window"), start_window)
    np.save(os.path.join(save_directory, "stop_window"), stop_window)

    # Plot These
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_rate)

    axis_1.plot(x_values, vis_context_vis_1_cd_projection, c='b')
    axis_1.plot(x_values, vis_context_vis_2_cd_projection, c='r')
    axis_1.plot(x_values, odr_context_vis_1_cd_projection, c='g')
    axis_1.plot(x_values, odr_context_vis_2_cd_projection, c='m')

    axis_1.axvline(0, linestyle='dashed', c='k')
    plt.show()



def get_group_data(output_root, session_list, condition_name):

    group_data = []
    for session in session_list:
        session_data = np.load(os.path.join(output_root, session, "Contextual_Modulation_Lick_CD", condition_name + ".npy"))
        group_data.append(session_data)

    group_data = np.array(group_data)
    group_mean, group_upper_bound, group_lower_bound = ALM_Analysis_Utils.get_sem_and_bounds(group_data)
    return group_mean, group_upper_bound, group_lower_bound



def plot_group_modulation(data_root, session_list, output_root):

    # Load Data
    vis_context_vis_1_mean, vis_context_vis_1_upper_bound, vis_context_vis_1_lower_bound = get_group_data(output_root, session_list, "vis_context_vis_1_cd_projection")
    vis_context_vis_2_mean, vis_context_vis_2_upper_bound, vis_context_vis_2_lower_bound = get_group_data(output_root, session_list, "vis_context_vis_2_cd_projection")
    odr_context_vis_1_mean, odr_context_vis_1_upper_bound, odr_context_vis_1_lower_bound = get_group_data(output_root, session_list, "odr_context_vis_1_cd_projection")
    odr_context_vis_2_mean, odr_context_vis_2_upper_bound, odr_context_vis_2_lower_bound = get_group_data(output_root, session_list, "odr_context_vis_2_cd_projection")

    # Plot These
    figure_1 = plt.figure(figsize=(15,5))
    irrelevant_axis = figure_1.add_subplot(1, 2, 1)
    relevant_axis = figure_1.add_subplot(1, 2, 2)

    # Get X Values
    start_window = np.load(os.path.join(output_root, session_list[0],  "Contextual_Modulation_Lick_CD", "start_window.npy"))
    stop_window = np.load(os.path.join(output_root, session_list[0],  "Contextual_Modulation_Lick_CD", "stop_window.npy"))
    frame_rate = np.load(os.path.join(data_root, session_list[0], "Frame_Rate.npy"))

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 1.0 / frame_rate)

    irrelevant_axis.plot(x_values, odr_context_vis_1_mean, c='b')
    irrelevant_axis.plot(x_values, odr_context_vis_2_mean, c='r')
    irrelevant_axis.fill_between(x_values, odr_context_vis_1_lower_bound, odr_context_vis_1_upper_bound, color='b', alpha=0.5)
    irrelevant_axis.fill_between(x_values, odr_context_vis_2_lower_bound, odr_context_vis_2_upper_bound, color='r', alpha=0.5)

    relevant_axis.plot(x_values, vis_context_vis_1_mean, c='b')
    relevant_axis.plot(x_values, vis_context_vis_2_mean, c='r')
    relevant_axis.fill_between(x_values, vis_context_vis_1_lower_bound, vis_context_vis_1_upper_bound, color='b', alpha=0.5)
    relevant_axis.fill_between(x_values, vis_context_vis_2_lower_bound, vis_context_vis_2_upper_bound, color='r', alpha=0.5)

    irrelevant_axis.axvline(0, linestyle='dashed', c='k')
    relevant_axis.axvline(0, linestyle='dashed', c='k')

    irrelevant_axis.set_ylim([-2, 10])
    relevant_axis.set_ylim([-2, 10])

    irrelevant_axis.set_xlabel("Time (S)")
    irrelevant_axis.set_ylabel("Lick CD Projection")
    irrelevant_axis.spines[['right', 'top']].set_visible(False)

    relevant_axis.set_xlabel("Time (S)")
    relevant_axis.set_ylabel("Lick CD Projection")
    relevant_axis.spines[['right', 'top']].set_visible(False)

    save_directory = os.path.join(output_root, "Group_Plots")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Lick_CD_Contextual_Modualtion.png"))
    plt.close()