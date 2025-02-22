import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle



def moving_average_df(delta_f_matrix, window_size):

    smoothed_df = []
    n_timepoints = len(delta_f_matrix)

    smoothed_df = np.zeros(np.shape(delta_f_matrix))

    for timepoint_index in range(window_size, n_timepoints):
        timepoint_data = delta_f_matrix[timepoint_index-window_size:timepoint_index]
        timepoint_data = np.mean(timepoint_data, axis=0)
        smoothed_df[timepoint_index] = timepoint_data

    smoothed_df = np.array(smoothed_df)
    return smoothed_df


def sort_raster_by_list(raster, list_to_sort_by):
    # Raster Must Be Of Shape, (N_Timepoints, N_Neurons)
    sorted_indicies = list_to_sort_by.argsort()
    sorted_indicies = np.flip(sorted_indicies)
    sorted_raster = raster[:, sorted_indicies]
    return sorted_raster


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
def sort_matrix(matrix):

    # Cluster Matrix
    Z = ward(pdist(matrix))

    # Get Dendogram Leaf Order
    new_order = leaves_list(Z)

    # Sorted Matrix
    sorted_matrix = matrix[:, new_order][new_order]

    return sorted_matrix
"""



def get_sem_and_bounds(data):
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    #sem = np.std(data, axis=0)

    upper_bound = np.add(mean, sem)
    lower_bound = np.subtract(mean, sem)
    return mean, upper_bound, lower_bound


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


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

def downsample_ai_trace(ai_trace, stack_onsets):

    # Get Average Stack Duration
    stack_duration_list = np.diff(stack_onsets)
    mean_stack_duration = int(np.mean(stack_duration_list))

    downsampled_trace = []
    n_stacks = len(stack_onsets)
    for stack_index in range(n_stacks-1):
        stack_start = stack_onsets[stack_index]
        stack_stop = stack_onsets[stack_index + 1]
        stack_data = ai_trace[stack_start:stack_stop]
        stack_data = np.mean(stack_data)
        downsampled_trace.append(stack_data)

    # Add Last
    final_data = ai_trace[stack_onsets[-1]:stack_onsets[-1] + mean_stack_duration]
    final_data = np.mean(final_data)
    downsampled_trace.append(final_data)

    return downsampled_trace


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




def get_ragged_tensor(df_matrix, onset_list, start_window, baseline_correction=True, baseline_start=0, baseline_stop=5):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    n_trials = len(onset_list)
    for trial_index in range(n_trials):
        trial_start = onset_list[trial_index, 0] + start_window
        trial_stop = onset_list[trial_index, 1]

        if trial_start >= 0 and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

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
    activity_matrix = load_df_matrix(os.path.join(data_root_directory, session))
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



def load_regression_matrix(session, mvar_output_directory, context):

    regression_matrix = np.load(os.path.join(mvar_output_directory,session, "Design_Matricies", context + "_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]

    DesignMatrix = regression_matrix["DesignMatrix"]
    dFtot = regression_matrix["dFtot"]
    Nvar = regression_matrix["Nvar"]
    Nbehav = regression_matrix["Nbehav"]
    Nt = regression_matrix["Nt"]
    Nstim = regression_matrix["N_stim"]
    Ntrials = regression_matrix["N_trials"]
    timewindow = regression_matrix["timewindow"]

    return DesignMatrix, dFtot, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow
