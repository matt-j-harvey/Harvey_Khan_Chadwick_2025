import os
import numpy as np
import pickle
import MVAR_Preprocessing_Utils


def moving_average(df_matrix, window_size=2):

    " must be of shape (n_timepoints, n_neurons)"
    n_timepoints, n_neurons = np.shape(df_matrix)
    smoothed_matrix = np.copy(df_matrix)

    for x in range(window_size, n_timepoints):
        data = df_matrix[x-window_size:x]
        data = np.mean(data, axis=0)
        smoothed_matrix[x] = data

    return smoothed_matrix

def create_activity_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root_directory, session, "df_over_f_matrix.npy"))
    #activity_matrix = moving_average(activity_matrix)
    number_of_timepoints, number_of_components = np.shape(activity_matrix)
    print("DF Matrix", np.shape(activity_matrix))

    # Load Onsets
    onsets_list = MVAR_Preprocessing_Utils.load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints)

    # Get Activity Tensors
    activity_tensor = MVAR_Preprocessing_Utils.get_data_tensor(activity_matrix, onsets_list, start_window, stop_window)

    # Convert Tensor To Array
    activity_tensor = np.array(activity_tensor)

    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "tensor":activity_tensor,
        "start_window": start_window,
        "stop_window": stop_window,
        "onsets_file": onsets_file,
    }

    # Save Trial Tensor
    save_directory = os.path.join(mvar_output_directory, session, "Activity_Tensors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(save_directory, tensor_name)
    print("Tensor file", tensor_file)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)


