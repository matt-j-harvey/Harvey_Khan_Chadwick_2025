import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import Mean_Activity_Utils

def z_score_trial(regressor, pixel_means, pixel_stds):

    # Reconstruct Into Image
    indicies, image_height, image_width = Mean_Activity_Utils.load_tight_mask()
    pixel_means = Mean_Activity_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Mean_Activity_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Flatten Into 1D
    pixel_means = np.reshape(pixel_means, (image_height * image_width))
    pixel_stds = np.reshape(pixel_stds, (image_height * image_width))

    # Z Score Regressor
    regressor = np.subtract(regressor, pixel_means)
    regressor = np.divide(regressor, pixel_stds)

    return regressor


def reconstruct_data(spatial_components, temporal_components):

    # Reshape Spatial Components
    image_height, image_width, n_components = np.shape(spatial_components)
    spatial_components = np.reshape(spatial_components, (image_height * image_width, n_components))
    print("spatial_components", np.shape(spatial_components))
    print("temporal_components", np.shape(temporal_components))

    # Reconstruct Into Pixel Space
    reconstructed_trial = np.matmul(spatial_components, temporal_components)
    reconstructed_trial = np.transpose(reconstructed_trial)

    return reconstructed_trial


def get_mean_temporal_components(behaviour_matrix, temporal_components, bin_start, bin_stop, selected_trial_type, start_window, stop_window):
    trial_tensor = []
    n_timepoints = np.shape(temporal_components)[1]
    for trial in behaviour_matrix:
        trial_id = trial[0]
        trial_type = trial[1]
        trial_rt = trial[23]
        trial_onset_frame = trial[18]

        if trial_rt >= bin_start and trial_rt < bin_stop:
            if trial_type == selected_trial_type:
                if trial_onset_frame != None:
                    if not np.isnan(trial_rt):

                        trial_start = trial_onset_frame + start_window
                        trial_stop = trial_onset_frame + stop_window

                        if trial_start > 0:
                            if trial_stop < n_timepoints:
                                trial_temporal_components = temporal_components[:, trial_start:trial_stop]
                                trial_tensor.append(trial_temporal_components)
    if len(trial_tensor) == 0:
        return None
    if len(trial_tensor) == 1:
        return trial_tensor[0]
    else:
        trial_tensor = np.array(trial_tensor)
        trial_mean = np.mean(trial_tensor, axis=0)
        return trial_mean


def get_session_bin_mean(data_root, session, bin_start, bin_stop, selected_trial_type, start_window, stop_window):

    # Load Neural Data
    temporal_components = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Corrected_SVT.npy"))
    spatial_components = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Registered_U.npy"))
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Mean Temporal Component
    mean_temporal_components = get_mean_temporal_components(behaviour_matrix, temporal_components, bin_start, bin_stop, selected_trial_type, start_window, stop_window)

    # Get Mean Activity
    if mean_temporal_components is not None:
        mean_activity = reconstruct_data(spatial_components, mean_temporal_components)
        mean_activity = z_score_trial(mean_activity, pixel_means, pixel_stds)
        return mean_activity
    else:
        return None

def get_group_mean(data_root, session_list, bin_start, bin_stop, selected_trial_type, start_window, stop_window):

    group_activity = []

    for mouse in session_list:
        mouse_activity = []

        for session in mouse:
            session_activity = get_session_bin_mean(data_root, session, bin_start, bin_stop, selected_trial_type, start_window, stop_window)
            if session_activity is not None:
                mouse_activity.append(session_activity)

        if len(mouse_activity) == 1:
            group_activity.append(mouse_activity[0])
        elif len(mouse_activity) > 1:
            mouse_activity = np.array(mouse_activity)
            mouse_mean = np.mean(mouse_activity, axis=0)
            group_activity.append(mouse_mean)

    if len(group_activity) == 0:
        return None
    elif len(group_activity) == 1:
        return group_activity[0]
    else:
        group_activity = np.array(group_activity)
        group_mean = np.mean(group_activity, axis=0)

    return group_mean


def get_rt_bin_means(wt_data_root,
                     nx_data_root,
                     wt_session_list,
                     nx_session_list,
                     rt_bin_starts,
                     rt_bin_stops,
                     selected_trial_type,
                     start_window,
                     stop_window,
                     output_root):

    wt_rt_bin_means = []
    nx_rt_bin_means = []

    for bin_index in range(len(rt_bin_starts)):
        bin_start = rt_bin_starts[bin_index]
        bin_stop = rt_bin_stops[bin_index]

        # Get Mean Activity For Each Genotype
        wt_mean = get_group_mean(wt_data_root, wt_session_list, bin_start, bin_stop, selected_trial_type, start_window, stop_window)
        nx_mean = get_group_mean(nx_data_root, nx_session_list, bin_start, bin_stop, selected_trial_type, start_window, stop_window)
        print("wt_mean", np.shape(wt_mean))
        print("nx_mean", np.shape(nx_mean))

        # Add To List
        wt_rt_bin_means.append(wt_mean)
        nx_rt_bin_means.append(nx_mean)

    # Save Data
    wt_rt_bin_means = np.array(wt_rt_bin_means)
    nx_rt_bin_means = np.array(nx_rt_bin_means)

    save_directory = os.path.join(output_root, str(selected_trial_type))
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "wt_rt_bin_means.npy"), wt_rt_bin_means)
    np.save(os.path.join(save_directory, "nx_rt_bin_means.npy"), nx_rt_bin_means)
