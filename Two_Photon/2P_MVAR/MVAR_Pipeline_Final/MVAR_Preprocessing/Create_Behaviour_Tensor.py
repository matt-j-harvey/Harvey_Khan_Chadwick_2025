import numpy as np
import os
import pickle

import MVAR_Preprocessing_Utils

def create_behaviour_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(mvar_output_directory, session, "Behaviour", "Behaviour_Matrix.npy"))
    number_of_timepoints, number_of_components = np.shape(behaviour_matrix)
    print("behaviour_matrix", np.shape(behaviour_matrix))

    # Load Onsets
    onsets_list = MVAR_Preprocessing_Utils.load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints)

    # Get Activity Tensors
    behaviour_tensor = MVAR_Preprocessing_Utils.get_data_tensor(behaviour_matrix, onsets_list, start_window, stop_window)
    print("behaviour_tensor", np.shape(behaviour_tensor))

    # Convert Tensor To Array
    behaviour_tensor = np.array(behaviour_tensor)

    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "tensor":behaviour_tensor,
        "start_window": start_window,
        "stop_window": stop_window,
        "onsets_file": onsets_file,
    }

    # Save Trial Tensor
    save_directory = os.path.join(mvar_output_directory, session, "Behaviour_Tensors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(save_directory, tensor_name)
    print("Tensor file", tensor_file)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)
