import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats, linalg

import Get_Data_Tensor
import Get_Mean_SEM_Bounds
import View_PSTH



def load_df_matrix(base_directory):
    # Load DF Data
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)
    df_matrix = stats.zscore(df_matrix, axis=0)
    df_matrix = np.nan_to_num(df_matrix)
    return df_matrix


def get_lick_coding_dimension(base_directory):

    # Load DF Matrix
    df_matrix = load_df_matrix(base_directory)

    # Load Lick Onsets
    lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1) / frame_rate

    # Get Data Tensor
    start_window = int(-2.5 * frame_rate)
    stop_window = 0
    print("start_window", start_window, "stop_window", stop_window)

    lick_df_tensor = Get_Data_Tensor.get_data_tensor(df_matrix,
                                                     lick_onsets,
                                                     start_window,
                                                     stop_window,
                                                     baseline_correction=True,
                                                     baseline_start=0,
                                                     baseline_stop=5)

    # Get Mean Dimension
    lick_df_tensor = lick_df_tensor[:, -6:]
    lick_vector = np.mean(lick_df_tensor, axis=0)
    lick_vector = np.mean(lick_vector, axis=0)

    # Norm Vector
    vector_length = linalg.norm(lick_vector)
    lick_cd = np.divide(lick_vector, vector_length)

    # Save Lick CD
    save_directory = os.path.join(base_directory, "Coding_Dimensions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Lick_CD.npy"), lick_cd)



def plot_lick_cd_projection(session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=None):


    # Load Frame Rate
    frame_rate = np.load(os.path.join(session_list[0], "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    n_conditions = len(condition_onsets_list)
    group_condition_means = []
    group_condition_lower_bounds = []
    group_condition_higher_bounds = []

    for condition_index in range(n_conditions):

        # Get Onsets File
        onsets_file = condition_onsets_list[condition_index]
        condition_means = []

        for base_directory in session_list:

            # Load DF Matrix
            df_matrix = load_df_matrix(base_directory)
            print("df matrix", np.shape(df_matrix))

            # Load Onsets
            onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file + ".npy"))

            # Load CD
            session_coding_dimension = np.load(os.path.join(base_directory, "Coding_Dimensions", "Lick_CD.npy"))

            # Get Tensors
            tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

            # Get Mean
            mean_activity = np.mean(tensor, axis=0)

            # Get Projection
            session_projection = np.dot(mean_activity, session_coding_dimension)

            # Add To List
            condition_means.append(session_projection)



        condition_means = np.array(condition_means)
        group_mean, lower_bound, upper_bound = Get_Mean_SEM_Bounds.get_sem_and_bounds(condition_means)

        group_condition_means.append(group_mean)
        group_condition_lower_bounds.append(lower_bound)
        group_condition_higher_bounds.append(upper_bound)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, period)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot()

    n_conditions = len(group_condition_means)
    cmap = plt.get_cmap('jet')
    for condition_index in range(n_conditions):
        colour = cmap(float(condition_index) / n_conditions)
        condition_mean = group_condition_means[condition_index]
        condition_lower_bound = group_condition_lower_bounds[condition_index]
        condition_higher_bound = group_condition_higher_bounds[condition_index]
        axis_1.plot(x_values, condition_mean, c=colour, label=condition_onsets_list[condition_index])
        axis_1.fill_between(x=x_values, y1=condition_lower_bound, y2=condition_higher_bound, color=colour, alpha=0.5)

    axis_1.axvline(0, c='k', linestyle='dashed')

    if ylim != None:
        axis_1.set_ylim(ylim)

    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylabel("Lick CD Projection")
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.legend()
    plt.show()


wt_session_list = [
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\72.3C\2024_09_10_Switching",
]


hom_session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK64.1B\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1A\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1B\2024_09_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK72.1E\2024_08_23_Switching",
]




start_window = -18
stop_window = 18
coding_dimension = "Lick_CD"
#condition_onsets_list = ["Lick_Onset_Frames"]

#condition_onsets_list = ["visual_context_stable_vis_1_onsets", "visual_context_stable_vis_2_onsets"]
#condition_onsets_list = ["odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]

#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets"]
condition_onsets_list = ["visual_context_stable_vis_1_onsets", "odour_context_stable_vis_1_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "visual_context_false_alarms_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_1_onsets", "visual_context_false_alarms_onsets"]
#condition_onsets_list = ["odour_context_stable_vis_1_onsets", "visual_context_false_alarms_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_1_onsets", "odour_context_stable_vis_1_onsets","visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets","visual_context_false_alarms_onsets"]


#condition_onsets_list = ["odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["odour_1_not_cued_onsets", "odour_2_not_cued_onsets"]
#condition_onsets_list = ["odour_1_cued_onsets", "odour_1_not_cued_onsets"]
#condition_onsets_list = ["odour_1_preceeded_by_vis_1", "odour_1_preceeded_by_vis_2"]
#condition_onsets_list = ["odour_1_not_cued_onsets", "odour_2_not_cued_onsets"]
#condition_onsets_list = ["odour_2_not_cued_onsets", "visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["odour_1_cued_onsets", "odour_1_not_cued_onsets"]
#condition_onsets_list = ["odour_1_preceeded_by_vis_1", "odour_1_preceeded_by_vis_2", "odour_1_not_cued_onsets"]

#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "visual_context_false_alarms_onsets", "visual_context_stable_vis_1_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets", "visual_context_false_alarms_onsets"]

#condition_onsets_list = ["visual_context_false_alarms_onsets", "visual_context_stable_vis_2_onsets"]


#plot_lick_cd_projection(hom_session_list, condition_onsets_list, coding_dimension, start_window, stop_window)

"""
condition_onsets_list = ["visual_context_stable_vis_1_onsets", "odour_context_stable_vis_1_onsets"]
plot_lick_cd_projection(wt_session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=[-7,10])

condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets"]
plot_lick_cd_projection(wt_session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=[-7,10])
"""
condition_onsets_list = ["visual_context_stable_vis_1_onsets", "visual_context_stable_vis_2_onsets"]
plot_lick_cd_projection(wt_session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=[-7,10])

condition_onsets_list = ["odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]
plot_lick_cd_projection(wt_session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=[-7,10])

"""
condition_onsets_list = ["odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]
plot_lick_cd_projection(wt_session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=[-2.5,5])
"""

condition_onsets_list = ["visual_context_stable_vis_1_onsets", "visual_context_stable_vis_2_onsets"]
plot_lick_cd_projection(wt_session_list, condition_onsets_list, coding_dimension, start_window, stop_window, ylim=[-7,10])