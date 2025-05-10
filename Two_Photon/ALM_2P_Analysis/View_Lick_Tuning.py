import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import Two_Photon_Utils
import Get_Data_Tensor
import View_PSTH



def view_lick_tuning(base_directory):

    # Load DF Data
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)
    df_matrix = stats.zscore(df_matrix, axis=0)
    df_matrix = np.nan_to_num(df_matrix)
    print("df matrix", np.shape(df_matrix))

    # Load Lick Onsets
    lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"))
    print("lick onsets", len(lick_onsets))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1)/frame_rate

    # Get Data Tensor
    start_window = -int(2.5 * frame_rate)
    stop_window = int(1 * frame_rate)
    print("start_window", start_window, "stop_window", stop_window)

    lick_df_tensor = Get_Data_Tensor.get_data_tensor(df_matrix,
                                                     lick_onsets,
                                                     start_window,
                                                     stop_window,
                                                     baseline_correction=True,
                                                     baseline_start=0,
                                                     baseline_stop=5)

    save_directory = None
    condition_name = "Lick"
    sorting_window_start = int((np.abs(start_window) - frame_rate))
    sorting_window_stop = np.abs(start_window)
    print("sorting_window_start", sorting_window_start, "sorting_window_stop", sorting_window_stop)

    View_PSTH.view_single_psth(lick_df_tensor,
                               start_window,
                               stop_window,
                               period,
                               save_directory,
                               condition_name,
                               sorting_window_start,
                               sorting_window_stop)


session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]


for base_directory in session_list:
    view_lick_tuning(base_directory)