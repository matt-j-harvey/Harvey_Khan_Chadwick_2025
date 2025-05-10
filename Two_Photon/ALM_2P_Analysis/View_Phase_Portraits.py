import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import Get_Data_Tensor
import Get_Mean_SEM_Bounds

def load_df_matrix(base_directory):
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)
    df_matrix = stats.zscore(df_matrix, axis=0)
    df_matrix = np.nan_to_num(df_matrix)
    return df_matrix



def view_phase_portraits(session_list, condition_onsets_list, start_window, stop_window):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(session_list[0], "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    n_conditions = len(condition_onsets_list)
    group_positive_mean_activity_list = []
    group_negative_mean_activity_list = []


    for condition_index in range(n_conditions):
        condition_onsets = condition_onsets_list[condition_index]

        condition_positive_list = []
        condition_negative_list = []

        for base_directory in session_list:

            # Load DF Matrix
            df_matrix = load_df_matrix(base_directory)

            # Load Onsets
            onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_onsets + ".npy"))

            # Load Selected Neurons
            positive_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", "positive_cell_indexes.npy"))
            negative_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", "negative_cell_indexes.npy"))

            # Split DF
            positive_df_matrix = df_matrix[:, positive_neurons]
            negative_df_matrix = df_matrix[:, negative_neurons]

            # Get Tensors
            positive_tensor = Get_Data_Tensor.get_data_tensor(positive_df_matrix, onsets, start_window, stop_window,
                                                     baseline_correction=True, baseline_start=0, baseline_stop=5)

            negative_tensor = Get_Data_Tensor.get_data_tensor(negative_df_matrix, onsets, start_window, stop_window,
                                                     baseline_correction=True, baseline_start=0, baseline_stop=5)

            # Get Means
            positive_mean = np.mean(positive_tensor, axis=0)
            positive_mean = np.mean(positive_mean, axis=1)

            negative_mean = np.mean(negative_tensor, axis=0)
            negative_mean = np.mean(negative_mean, axis=1)

            print("Positive mean", np.shape(positive_mean))
            print("negative_mean", np.shape(negative_mean))

            condition_positive_list.append(positive_mean)
            condition_negative_list.append(negative_mean)

        condition_positive_list = np.array(condition_positive_list)
        condition_negative_list = np.array(condition_negative_list)

        group_positive_mean = np.mean(condition_positive_list, axis=0)
        group_negative_mean = np.mean(condition_negative_list, axis=0)


        plt.plot(group_negative_mean, group_positive_mean)
    plt.show()


session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]


start_window = -18
stop_window = 15
condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets", "visual_context_stable_vis_1_onsets", "visual_context_false_alarms_onsets"]
view_phase_portraits(session_list, condition_onsets_list, start_window, stop_window)
