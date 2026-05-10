import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import Opto_GLM_Utils


def z_score_activity_tensor(data_root, session, activity_tensor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()
    pixel_means = Opto_GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Opto_GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    z_scored_activity_tensor = []
    for trial in activity_tensor:
        trial = np.subtract(trial, pixel_means)
        trial = np.divide(trial, pixel_stds)
        trial = np.nan_to_num(trial)
        z_scored_activity_tensor.append(trial)

    z_scored_activity_tensor = np.array(z_scored_activity_tensor)
    return z_scored_activity_tensor


def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=14, early_cutoff=3000):

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



def reconstruct_activity_tensor_into_pixel_space(data_root, session, activity_tensor):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Registered_U.npy"))
    print("spatial_components", np.shape(spatial_components))

    # Reconstrct
    recontructed_tensor = []
    for trial in tqdm(activity_tensor):
        #print("trial", np.shape(trial))
        trial = np.dot(spatial_components, np.transpose(trial))
        trial = np.moveaxis(trial, 2, 0)
        #print("trial", np.shape(trial))
        recontructed_tensor.append(trial)

    reconstructed_tensor = np.array(recontructed_tensor)

    return reconstructed_tensor





def create_activity_tensor(data_root, session, onsets_list, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Corrected_SVT.npy"))
    activity_matrix = np.transpose(activity_matrix)
    #print("Activity Matrix", np.shape(activity_matrix))
    number_of_timepoints, number_of_components = np.shape(activity_matrix)

    # Get Activity Tensors
    activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window,  baseline_correction=True)
    #print("activity_tensor svt", np.shape(activity_tensor))

    # Get Mean In Time Window
    #activity_tensor = activity_tensor[:, mean_window_start:mean_window_stop]
    #activity_tensor = np.mean(activity_tensor, axis=1)
    #print("activity_tensor svt mean window", np.shape(activity_tensor))

    # Reconstruct Into Pixel Space
    activity_tensor = reconstruct_activity_tensor_into_pixel_space(data_root, session, activity_tensor)
    #print("activity_tensor pixel", np.shape(activity_tensor))

    # Z Score
    activity_tensor = z_score_activity_tensor(data_root, session, activity_tensor)
    #print("activity_tensor z score", np.shape(activity_tensor))

    return activity_tensor



def get_activity_tensor_svd(data_root, session, onsets_list, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
    #activity_matrix = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Corrected_SVT.npy"))
    activity_matrix = np.transpose(activity_matrix)

    # Get Activity Tensors
    activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window, baseline_correction=True)

    return activity_tensor