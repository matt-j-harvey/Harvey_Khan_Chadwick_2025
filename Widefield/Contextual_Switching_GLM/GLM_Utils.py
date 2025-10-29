import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary






def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=5, early_cutoff=3000):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= early_cutoff and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            """
            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)
            """
            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor



def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image




def create_activity_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Activity Matrix
    #activity_matrix = np.load(os.path.join(data_root_directory, session, "Preprocessed_Data", "Corrected_SVT.npy"))
    activity_matrix = np.load(os.path.join(mvar_output_directory, session, "Local_NMF", "Temporal_Components.npy"))

    activity_matrix = np.transpose(activity_matrix)
    number_of_timepoints, number_of_components = np.shape(activity_matrix)

    # Load Onsets
    onsets_list = np.load(os.path.join(mvar_output_directory, session, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensors
    activity_tensor = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window)

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

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)





def create_behaviour_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(mvar_output_directory, session, "Behaviour", "Behavioural_Regressor_Matrix.npy"))
    number_of_timepoints, number_of_components = np.shape(behaviour_matrix)

    # Load Onsets
    onsets_list = np.load(os.path.join(mvar_output_directory, session, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensors
    behaviour_tensor = get_data_tensor(behaviour_matrix, onsets_list, start_window, stop_window)

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

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)
