import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import Get_Data_Tensor
import Plotting_Functions


def get_lick_coding_dimension(base_directory):

    # Load DF Data
    df_matrix = np.load(os.path.join(base_directory, "df_over_f_matrix.npy"))
    print("df matrix", np.shape(df_matrix))

    # Load Lick Onsets
    lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"))
    print("lick onsets", len(lick_onsets))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))

    # Get Data Tensor
    start_window = -frame_rate
    stop_window = 0





def view_lick_tuning(data_root, session, mvar_output_root):

    # Load DF Data
    df_matrix = np.load(os.path.join(mvar_output_root, session, "df_over_f_matrix.npy"))
    print("df matrix", np.shape(df_matrix))

    # Load Lick Onsets
    lick_onsets = np.load(os.path.join(mvar_output_root, session, "Behaviour", "Lick_Onset_Frames.npy"))
    print("lick onsets", len(lick_onsets))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    period = float(1)/frame_rate

    # Get Data Tensor
    start_window = -int(2 * frame_rate)
    stop_window = int(1 * frame_rate)
    print("start_window", start_window, "stop_window", stop_window)

    lick_df_tensor = Get_Data_Tensor.get_data_tensor(df_matrix,
                                                     lick_onsets,
                                                     start_window,
                                                     stop_window,
                                                     baseline_correction=True,
                                                     baseline_start=0,
                                                     baseline_stop=5)

    save_directory = os.path.join(mvar_output_root, session, "Lick_Tuning")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    condition_name = "Lick"
    sorting_window_start = int((np.abs(start_window) - frame_rate))
    sorting_window_stop = np.abs(start_window)
    print("sorting_window_start", sorting_window_start, "sorting_window_stop", sorting_window_stop)

    Plotting_Functions.view_single_psth(lick_df_tensor,
                               start_window,
                               stop_window,
                               period,
                               save_directory,
                               condition_name,
                               sorting_window_start,
                               sorting_window_stop)

    # Get Lick Coding Dimension
    lick_cd_window_start = int(abs(start_window) - frame_rate)
    lick_cd_window_stop = abs(start_window)

    # Get Lick Preceeding Tensor
    lick_preceeding_tensor = lick_df_tensor[:, lick_cd_window_start:lick_cd_window_stop]

    # Get Mean Activity
    mean_lick_activity = np.mean(lick_preceeding_tensor, axis=1) # Across Time
    mean_lick_activity = np.mean(mean_lick_activity, axis=0) # Across Trials

    # Normalise
    norm = np.linalg.norm(mean_lick_activity)
    coding_dimension = np.divide(mean_lick_activity, norm)

    # Save
    np.save(os.path.join(save_directory, "Lick_Coding_Dimension.npy"), coding_dimension)
