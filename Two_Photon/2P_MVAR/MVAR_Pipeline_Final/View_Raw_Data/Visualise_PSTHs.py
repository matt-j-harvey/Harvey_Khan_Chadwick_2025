import numpy as np
import os
import Get_Data_Tensor
import Plotting_Functions


def view_psths(data_root, session, mvar_output_root, onset_file, start_window, stop_window):

    # Load dF/F
    df_matrix = np.load(os.path.join(data_root, session, "df_over_f_matrix.npy"))

    # Load Onsets
    onset_list = np.load(os.path.join(data_root, session, "Stimuli_Onsets", onset_file + "_onsets.npy"))

    # Get Data Tensor
    data_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    # Create Save Directory
    save_directory = os.path.join(mvar_output_root, session, "Raw Data Visualisation", "PSTHs")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot PSTH
    Plotting_Functions.view_single_psth(data_tensor,
                               start_window,
                               stop_window,
                               period,
                               save_directory,
                               onset_file,
                               np.abs(start_window),
                               np.abs(start_window) + 5)

    # Plot PSTH Sorted by Lick Coding Dimension
    lick_cd = np.load(os.path.join(mvar_output_root, session, "Raw Data Visualisation", "Lick_Tuning", "Lick_Coding_Dimension.npy"))
    lick_cd_indicies = lick_cd.argsort()
    lick_cd_indicies = np.flip(lick_cd_indicies)
    print("lick_cd_indicies", np.shape(lick_cd_indicies))
    sorted_data_tensor = data_tensor[:, :, lick_cd_indicies]
    print("sorted_data_tensor", np.shape(sorted_data_tensor))
    print("data_tensor", np.shape(data_tensor))

    Plotting_Functions.view_single_psth(sorted_data_tensor,
                               start_window,
                               stop_window,
                               period,
                               save_directory,
                               onset_file + "_sorted_by_lick_cd",
                               np.abs(start_window),
                               np.abs(start_window) + 5)

