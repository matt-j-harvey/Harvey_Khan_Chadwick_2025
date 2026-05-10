import os
import pickle
import numpy as np
import matplotlib.pyplot as plt


import Opto_GLM_Utils


def z_score_regressor(data_root, session, regressor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()
    pixel_means = Opto_GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Opto_GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    regressor = np.subtract(regressor, pixel_means)
    regressor = np.divide(regressor, pixel_stds)

    return regressor


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



def reconstruct_regressor_into_pixel_space(data_root, session, regressor):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(data_root, session, "Local_NMF", "Spatial_Components.npy"))
    print("spatial_components", np.shape(spatial_components))

    # Reconstrct
    regressor = np.dot(spatial_components, np.transpose(regressor))
    regressor = np.moveaxis(regressor, -1, 0)
    print("regressor", np.shape(regressor))

    # Z Score
    regressor = z_score_regressor(data_root, session, regressor)
    print("regressor", np.shape(regressor))

    return regressor





def create_activity_tensor(data_root, session, output_root, onsets_file, start_window, stop_window, mean_window_start, mean_window_stop):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
    activity_matrix = np.transpose(activity_matrix)
    number_of_timepoints, number_of_components = np.shape(activity_matrix)

    # Load Onsets
    onsets_list = np.load(os.path.join(output_root, session, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensors
    activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window,  baseline_correction=False)
    print("activity_tensor", np.shape(activity_tensor))

    # Get Mean In Time Window
    #activity_tensor = activity_tensor[:, mean_window_start:mean_window_stop]
    #activity_tensor = np.mean(activity_tensor, axis=1)
    print("activity_tensor", np.shape(activity_tensor))

    """
    # Reconstruct Into Pixel Space
    activity_tensor = reconstruct_regressor_into_pixel_space(data_root, session, activity_tensor)
    print("activity_tensor", np.shape(activity_tensor))

    # Z Score
    activity_tensor = z_score_regressor(data_root, session, activity_tensor)
    print("activity_tensor", np.shape(activity_tensor))
    """

    # Convert Tensor To Array
    activity_tensor = np.array(activity_tensor)

    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "tensor":activity_tensor,
        "start_window": start_window,
        "stop_window": stop_window,
        "onsets_file": onsets_file,
        "mean_window_start":mean_window_start,
        "mean_window_stop":mean_window_stop,
    }

    # Save Trial Tensor
    save_directory = os.path.join(output_root, session, "Activity_Tensors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(save_directory, tensor_name)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)





def create_behaviour_tensor(data_root, session, output_root, onsets_file, start_window, stop_window,  mean_window_start, mean_window_stop):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(output_root, session, "Behaviour", "Behavioural_Regressor_Matrix.npy"))
    number_of_timepoints, number_of_components = np.shape(behaviour_matrix)

    # Load Onsets
    onsets_list = np.load(os.path.join(output_root, session, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensors
    behaviour_tensor = get_data_tensor(behaviour_matrix, onsets_list, start_window, stop_window)

    # Convert Tensor To Array
    behaviour_tensor = np.array(behaviour_tensor)

    # Get Mean In Time Window
    #behaviour_tensor = behaviour_tensor[:, mean_window_start:mean_window_stop]
    #behaviour_tensor = np.mean(behaviour_tensor, axis=1)
    print("behaviour_tensor", np.shape(behaviour_tensor))

    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "tensor":behaviour_tensor,
        "start_window": start_window,
        "stop_window": stop_window,
        "onsets_file": onsets_file,
        "mean_window_start": mean_window_start,
        "mean_window_stop": mean_window_stop,
    }

    # Save Trial Tensor
    save_directory = os.path.join(output_root, session, "Behaviour_Tensors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(save_directory, tensor_name)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)
