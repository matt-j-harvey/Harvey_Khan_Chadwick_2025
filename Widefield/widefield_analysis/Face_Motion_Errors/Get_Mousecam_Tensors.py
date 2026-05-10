import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


import Opto_GLM_Utils


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





def get_facecam_tensor(data_root, session, onsets_list, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Mousecam_Analysis", "Full_Face_Motion_SVD.npy"))
    #activity_matrix = np.transpose(activity_matrix)
    print("Activity Matrix", np.shape(activity_matrix))
    number_of_timepoints, number_of_components = np.shape(activity_matrix)

    # Get Activity Tensors
    activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window,  baseline_correction=False)


    return activity_tensor



def get_activity_tensor_nmf(data_root, session, onsets_list, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
    activity_matrix = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Corrected_SVT.npy"))
    activity_matrix = np.transpose(activity_matrix)

    # Get Activity Tensors
    activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window, baseline_correction=True)

    return activity_tensor