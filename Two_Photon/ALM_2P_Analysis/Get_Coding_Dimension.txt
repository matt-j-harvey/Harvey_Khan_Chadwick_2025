import numpy as np
import os


def get_activity_tensors(delta_f_matrix, onsets_list, start_window, stop_window):

    n_timepoints = np.shape(delta_f_matrix)[0]

    activity_tensor = []
    for trial_onset in onsets_list:
        trial_start = int(trial_onset + start_window)
        trial_stop = int(trial_onset + stop_window)

        if trial_start > 0 and trial_stop < n_timepoints:
            trial_data = delta_f_matrix[trial_start:trial_stop]
            activity_tensor.append(trial_data)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor



def get_coding_dimension(delta_f_matrix, condition_1_onsets, condition_2_onsets, start_window, stop_window):

    """
    Following Proceedure in:
    Inagaki, Hidehiko K., et al. "Discrete attractor dynamics underlies persistent activity in the frontal cortex." Nature 566.7743 (2019): 212-217.

    1.) Get Average Response In Condition 1
    2.) Get Average Response in Condition 2
    3.) Subtract These
    4.) Normalise to length 1

    Note you can either use two conditions, eg left minus right stimuli or licks
    Or you can use a single condition, and subtract the baseline each time - eg lick minus baseline

    Arguments:
        Delta_f_Matrix = two-dimensional array of shape (N_Timepoints x N_Neurons)
        Condition_1_onsets = list of timepoints for events of interest in category 1 (eg stimuli 1 onsets)
        Condition_2_onsets = list of timepoints for events of interest in category 2 (eg stimuli 2 onsets)
        Start_Window = positve integer indicating how many timepoints before the onset to include
        Stop_Window = positive integer indicating how many timepoints after the onset to include

    """

    # Get Data Tensors - Return 3d tensor of shape (N_Trials, Trial_Length, N_Neurons)
    condition_1_tensor = get_activity_tensors(delta_f_matrix, condition_1_onsets, start_window, stop_window)
    condition_2_tensor = get_activity_tensors(delta_f_matrix, condition_2_onsets, start_window, stop_window)

    # Take Across Trials
    condition_1_trial_mean = np.mean(condition_1_tensor, axis=0)
    condition_2_trial_mean = np.mean(condition_2_tensor, axis=0)

    # Get Differerence (wt)
    wt = np.subtract(condition_1_trial_mean, condition_2_trial_mean)

    # Get Average Difference Across Time
    wt_time_average = np.mean(wt, axis=0)

    # Normalise Vector - This ensures the coding dimension vector has a length of 1, making it comparable across different mice
    norm = np.linalg.norm(wt_time_average)
    coding_dimension = np.divide(wt_time_average, norm)

    return coding_dimension
