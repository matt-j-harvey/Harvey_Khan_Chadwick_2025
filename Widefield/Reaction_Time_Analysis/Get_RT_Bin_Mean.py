import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import Create_Hit_RT_Matrix


def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=5, early_cutoff=3000):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= early_cutoff and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor



def get_session_mean(base_directory, hit_rt_matrix, start_window, stop_window, bin_start, bin_stop):

    # Get Onset List
    selected_onsets = []
    for trial in hit_rt_matrix:
        if trial[2] > bin_start and trial[2] <= bin_stop:
            selected_onsets.append(trial[0])

    if len(selected_onsets) > 0:

        # Load SVT
        svt = np.load(os.path.join(base_directory, "Preprocessed_Data", "Corrected_SVT.npy"))
        registered_u = np.load(os.path.join(base_directory, "Preprocessed_Data", "Registered_U.npy"))
        svt = np.transpose(svt)

        # Get SVT Tensor
        svt_tensor = get_data_tensor(svt, selected_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000)
        svt_tensor = np.nan_to_num(svt_tensor)

        if len(svt_tensor) > 0:

            # Get Mean SVT
            if len(svt_tensor) > 1:
                mean_svt = np.mean(svt_tensor, axis=0)
            else:
                mean_svt = svt_tensor[0]


            reconstructed_data = np.dot(registered_u, np.transpose(mean_svt))


            return reconstructed_data




def get_rt_bin_means(session_list, n_bins, bin_start_list, bin_stop_list, start_window, stop_window, output_directory):

    # Get List Of Mean Tensors

    # Create Save Directory
    save_directory = os.path.join(output_directory, "RT_Bin_Means")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    for bin_index in tqdm(range(n_bins)):

        bin_activity_list = []
        bin_start = bin_start_list[bin_index]
        bin_stop = bin_stop_list[bin_index]

        for mouse in tqdm(session_list):
            mouse_activity_list = []

            for session in mouse:

                # Create Hit RT Matrix
                hit_rt_time_matrix = Create_Hit_RT_Matrix.create_hit_rt_matrix(session)

                # Get Hit Tensor
                session_mean = get_session_mean(session, hit_rt_time_matrix, start_window, stop_window, bin_start, bin_stop)
                if session_mean is None:
                    pass
                else:
                    mouse_activity_list.append(session_mean)

            if len(mouse_activity_list) == 1:
                bin_activity_list.append(mouse_activity_list[0])

            elif len(mouse_activity_list) > 1:
                mouse_activity_list = np.array(mouse_activity_list)
                mouse_mean_activity = np.mean(mouse_activity_list, axis=0)
                bin_activity_list.append(mouse_mean_activity)

        # Save Activity List
        bin_activity_list = np.array(bin_activity_list)
        file_name = str(bin_start_list[bin_index]) + "_to_" + str(bin_stop_list[bin_index]) + ".npy"
        print(file_name, np.shape(bin_activity_list))
        np.save(os.path.join(save_directory, file_name), bin_activity_list)
