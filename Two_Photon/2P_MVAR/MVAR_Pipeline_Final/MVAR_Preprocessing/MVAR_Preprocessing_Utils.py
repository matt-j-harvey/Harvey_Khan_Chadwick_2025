import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats


def load_rig_1_channel_dict():

    channel_dict = {

        'Frame Trigger':0,
        'Reward':1,
        'Lick':2,
        'Visual 1':3,
        'Visual 2':4,
        'Odour 1':5,
        'Odour 2':6,
        'Irrelevance':7,
        'Running':8,
        'Trial End':9,
        'Optogenetics':10,
        'Mapping Stim':11,
        'Empty':12,
        'Mousecam':13,

    }

    return channel_dict



def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=5):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor



def load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints):

    onset_file_path = os.path.join(data_root_directory, session, "Stimuli_Onsets", onsets_file)
    raw_onsets_list = np.load(onset_file_path)

    checked_onset_list = []
    for trial_onset in raw_onsets_list:
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window
        if trial_start > 0 and trial_stop < number_of_timepoints:
            checked_onset_list.append(trial_onset)

    return checked_onset_list



def create_activity_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = load_df_matrix(os.path.join(data_root_directory, session), smooth=False, z_score=True)
    number_of_timepoints, number_of_components = np.shape(activity_matrix)
    print("DF Matrix", np.shape(activity_matrix))

    # Load Onsets
    onsets_list = load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints)

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
    print("Tensor file", tensor_file)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)





def create_behaviour_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(mvar_output_directory, session, "Behaviour", "Behaviour_Matrix.npy"))
    number_of_timepoints, number_of_components = np.shape(behaviour_matrix)
    print("behaviour_matrix", np.shape(behaviour_matrix))

    # Load Onsets
    onsets_list = load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints)

    # Get Activity Tensors
    behaviour_tensor = get_data_tensor(behaviour_matrix, onsets_list, start_window, stop_window)
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


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

"""
def load_df_matrix(base_directory, z_score=True, smooth=True, window_size=3):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)

    # Z Score
    if z_score == True:
        df_matrix = stats.zscore(df_matrix, axis=0)
        df_matrix = np.nan_to_num(df_matrix)

    # Smooth Df Matrix
    if smooth == True:
        df_matrix = moving_average_df(df_matrix, window_size)

    return df_matrix

"""


def load_df_matrix():

    """
    Baseline fluorescence F0(t) was computed by:
     smoothing F(t) (causal moving average of 0.375s)
     determining for each time point the minimum value in the preceding 600s time window (120s for slice experiments).

     frame_rate 6.37
    Smoothing Window size = 3
    Moving baseline size = 3822


    """


frame_rate = np.load(r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2a\2024_08_05_Switching\Frame_Rate.npy")
print("frame_rate", frame_rate)

df_matrix = np.load(r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2a\2024_08_05_Switching\df_matrix.npy")

plt.imshow(df_matrix, vmin=0.2, vmax=2)
forceAspect(plt.gca())
plt.show()

