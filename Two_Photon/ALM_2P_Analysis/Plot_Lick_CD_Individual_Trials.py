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


def plot_lick_cd_individual_trials(base_directory):

    # Load DF Matrix
    df_matrix = load_df_matrix(base_directory)
    print("df matrix", np.shape(df_matrix))

    # Load Lick CD
    lick_cd = np.load(os.path.join(base_directory, "Coding_Dimensions",  "Lick_CD.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1) / frame_rate

    onset_file_list = ["visual_context_stable_vis_1_onsets", "odour_context_stable_vis_1_onsets"] #, "visual_context_false_alarms_onsets"]
    colour_list = ["b", "r", "r"]
    linestyle_list = [None, None, "dashed"]

    # Get CD Projections#
    cd_projections_list = []
    for onset_file in onset_file_list:

        # Load Onsets
        onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onset_file + ".npy"))

        # Get Tensor
        tensor = Get_Data_Tensor.get_data_tensor(df_matrix,
                                                 onsets,
                                                 start_window,
                                                 stop_window,
                                                 baseline_correction=True,
                                                 baseline_start=0,
                                                 baseline_stop=5)

        # Convert Tensor to CD Projection
        cd_projections = []
        for trial in tensor:
            trial_projection = np.dot(trial, lick_cd)
            cd_projections.append(trial_projection)

        cd_projections_list.append(cd_projections)


    # Plot Conditions
    n_conditions = len(onset_file_list)
    for condition_index in range(n_conditions):

        # Get Colour
        condition_colour = colour_list[condition_index]

        # Get Condition Name
        condition_name = onset_file_list[condition_index]

        # Plot Data
        cd_projections = cd_projections_list[condition_index]
        for trial in cd_projections:
            plt.plot(trial, c=condition_colour, linestyle=linestyle_list[condition_index], alpha=0.5)

    plt.show()

start_window = -18
stop_window = 18


session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]

for session in session_list:
    plot_lick_cd_individual_trials(session)
