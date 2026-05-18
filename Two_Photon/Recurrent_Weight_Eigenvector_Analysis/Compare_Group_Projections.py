import numpy as np
import os
import pickle

from Shared_Utils.Get_DF import load_df_matrix
import Plotting_Functions.Plotting_Utils as Plotting_Utils


def load_false_alarm_onsets(data_directory):

    # Load Behaviour matrix
    behaviour_matrix = np.load(os.path.join(data_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    flase_alarm_onsets = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_outcome = trial[3]
        troal_onset = trial[18]

        if trial_type == 2:
            if trial_outcome == 0:
                flase_alarm_onsets.append(troal_onset)

    return flase_alarm_onsets


def open_tensor(file_location):

    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        print("session trial dict", session_trial_tensor_dict.keys())
        activity_tensor = session_trial_tensor_dict["tensor"]

        # Swap Axes To Fit Angus Convention
        activity_tensor = np.swapaxes(activity_tensor, 0, 1)

    return activity_tensor




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
    raw_onsets_list = np.load(onset_file_path, allow_pickle=True)

    checked_onset_list = []
    for trial_onset in raw_onsets_list:
        if trial_onset != None:
            trial_start = trial_onset + start_window
            trial_stop = trial_onset + stop_window
            if trial_start > 0 and trial_stop < number_of_timepoints:
                checked_onset_list.append(trial_onset)

    return checked_onset_list


def load_mouse_data(session, data_root, mvar_root, start_window, stop_window):

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_root, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))

    # Load Df Matrix
    df_matrix = load_df_matrix(os.path.join(data_root, session))
    n_timepoints = np.shape(df_matrix)[0]

    # Load Onsets
    #onset_list = load_onsets(data_root, session, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window, n_timepoints)
    onset_list = load_false_alarm_onsets(os.path.join(data_root, session))

    # Get Activity Tensor
    activity_tensor = get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=3)

    # Get Mean Response
    mean_activity = np.mean(activity_tensor, axis=0)

    # Get Lick CD Projection
    lick_cd_projection = np.dot(mean_activity, lick_cd)

    return lick_cd_projection



def load_group_data(session_list, data_root, mvar_root, start_window, stop_window):

    group_data = []
    for mouse in session_list:
        mouse_data = []
        for session in mouse:
            session_data = load_mouse_data(session, data_root, mvar_root, start_window, stop_window)
            mouse_data.append(session_data)
        if len(mouse_data) == 1:
            group_data.append(mouse_data[0])
        else:
            mouse_data = np.mean(np.array(mouse_data), axis=0)
            group_data.append(mouse_data)

    group_data = np.array(group_data)
    return group_data


def compare_projections_group(wt_session_list, wt_data_root, wt_mvar_root, nx_session_list, nx_data_root, nx_mvar_root):

    start_window = -16
    stop_window = 12

    #start_window = -3
    #stop_window = 13

    # Load Group Data
    wt_data = load_group_data(wt_session_list, wt_data_root, wt_mvar_root, start_window, stop_window)
    nx_data = load_group_data(nx_session_list, nx_data_root, nx_mvar_root, start_window, stop_window)
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 1000.0 / 6.37)

    Plotting_Utils.plot_line_graph(wt_data, nx_data, "CR Lick CD Projection", x_value_time=False, ylim=None, set_x_values=x_values)