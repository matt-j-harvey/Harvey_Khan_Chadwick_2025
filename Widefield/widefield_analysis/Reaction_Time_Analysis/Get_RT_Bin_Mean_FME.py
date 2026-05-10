import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_data_tensor(running_trace, onset_list, start_window, stop_window):

    n_timepoints = np.shape(running_trace)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < n_timepoints:
            trial_data = running_trace[trial_start:trial_stop]
            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor




def get_session_mean(running_trace, selected_onsets, start_window, stop_window):

    # Get SVT Tensor
    running_tensor = get_data_tensor(running_trace, selected_onsets, start_window, stop_window)
    running_tensor = np.nan_to_num(running_tensor)

    if np.shape(running_tensor)[0] > 0:

        # Get Mean SVT
        if len(running_tensor) > 1:
            mean_running = np.mean(running_tensor, axis=0)
        else:
            mean_running = running_tensor[0]

        return mean_running




def get_hit_onsets(behaviour_matrix, rt_bin_start, rt_bin_stop):

    trial_onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_outcome = trial[3]
        trial_rt = trial[23]
        trial_onset = trial[18]

        if trial_type == 1:
            if trial_outcome == 1:
                if trial_onset != None:
                    if trial_rt >= rt_bin_start and trial_rt < rt_bin_stop:
                        trial_onset_list.append(trial_onset)

    return trial_onset_list




def get_fa_onsets(behaviour_matrix, rt_bin_start, rt_bin_stop):

    trial_onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_outcome = trial[3]
        trial_rt = trial[23]
        trial_onset = trial[18]

        if trial_type == 2:
            if trial_outcome == 0:
                if trial_onset != None:
                    if trial_rt >= rt_bin_start and trial_rt < rt_bin_stop:
                        trial_onset_list.append(trial_onset)

    return trial_onset_list




def get_cr_onsets(behaviour_matrix):

    trial_onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_outcome = trial[3]
        trial_onset = trial[18]

        if trial_type == 2:
            if trial_outcome == 1:
                if trial_onset != None:
                        trial_onset_list.append(trial_onset)

    return trial_onset_list


def get_irrel_onsets(behaviour_matrix):

    trial_onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        preceeded_by_irrel = trial[5]
        irrel_type = trial[6]
        ignored_irrel = trial[7]
        trial_outcome = trial[3]
        trial_onset = trial[20]

        # Check Is Odour Trial
        if trial_type == 3 or trial_type == 4:

            # Check Preceeded By Irrel
            if preceeded_by_irrel == 1:

                # Check Irrel Type Unrewarded
                if irrel_type == 2:

                    # Check Ignore Irrel
                    if ignored_irrel == 1:

                        if trial_outcome == 1:
                            if trial_onset != None:
                                trial_onset_list.append(trial_onset)

    return trial_onset_list



def get_rt_bin_mean_fme(data_root, session_list, n_bins, bin_start_list, bin_stop_list, start_window, stop_window, output_directory):

    # Create Save Directory
    save_directory = os.path.join(output_directory, "RT_Bin_Mean_FME")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Iterate Through RT Time Bins
    for bin_index in tqdm(range(n_bins)):
        bin_start = bin_start_list[bin_index]
        bin_stop = bin_stop_list[bin_index]

        bin_hit_activity_list = []
        bin_cr_activity_list = []

        for mouse in tqdm(session_list):
            mouse_hit_activity_list = []
            mouse_cr_activity_list = []

            for session in mouse:
                print(session)

                # Load Behaviour Matrix
                behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

                # Get Onsets
                hit_onsets = get_hit_onsets(behaviour_matrix, bin_start, bin_stop)
                cr_onsets = get_cr_onsets(behaviour_matrix)

                # Load Running Trace
                face_motion_energy = np.load(os.path.join(data_root, session, "Mousecam_Analysis", "Mean_Jaw_Motion_Energy.npy"))

                # Get Tensors
                hit_mean = get_session_mean(face_motion_energy, hit_onsets, start_window, stop_window)
                cr_mean = get_session_mean(face_motion_energy, cr_onsets, start_window, stop_window)

                if isinstance(hit_mean,np.ndarray):
                    mouse_hit_activity_list.append(hit_mean)

                if isinstance(cr_mean,np.ndarray):
                    mouse_cr_activity_list.append(cr_mean)


            if len(mouse_hit_activity_list) == 1:
                bin_hit_activity_list.append(mouse_hit_activity_list[0])

            elif len(mouse_hit_activity_list) > 1:
                mouse_hit_activity_list = np.array(mouse_hit_activity_list)
                mouse_hit_mean_activity = np.mean(mouse_hit_activity_list, axis=0)
                bin_hit_activity_list.append(mouse_hit_mean_activity)

            if len(mouse_cr_activity_list) == 1:
                bin_cr_activity_list.append(mouse_cr_activity_list[0])

            elif len(mouse_cr_activity_list) > 1:
                mouse_cr_activity_list = np.array(mouse_cr_activity_list)
                mouse_cr_mean_activity = np.mean(mouse_cr_activity_list, axis=0)
                bin_cr_activity_list.append(mouse_cr_mean_activity)



        # Save Activity List
        bin_hit_activity_list = np.array(bin_hit_activity_list)
        bin_cr_activity_list = np.array(bin_cr_activity_list)

        file_name = str(bin_start_list[bin_index]) + "_to_" + str(bin_stop_list[bin_index]) + ".npy"
        print(file_name, np.shape(bin_hit_activity_list))

        np.save(os.path.join(save_directory, "Hit_FME_" + file_name), bin_hit_activity_list)
        np.save(os.path.join(save_directory, "Cr_FME"), bin_cr_activity_list)
