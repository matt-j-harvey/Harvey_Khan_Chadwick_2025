import numpy as np


def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5):

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



def get_data_tensor_seperate_starts_stops(df_matrix, trial_start_list, trial_stop_list, start_window, baseline_correction=True, baseline_start=0, baseline_stop=5):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []

    n_trials = len(trial_start_list)
    for trial_index in range(n_trials):
        onset = trial_start_list[trial_index]
        trial_stop = trial_stop_list[trial_index]
        trial_start = onset + start_window

        print("onset",  "trial start", trial_start, "trial stop", trial_stop)

        if trial_start >= 0 and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

    return tensor






def get_data_tensor_seperate_baseline_onsets(df_matrix, trial_onset_list, baseline_onset_list, start_window, stop_window, baseline_start=0, baseline_stop=5):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []

    n_trials = len(trial_onset_list)
    for trial_index in range(n_trials):

        trial_onset = trial_onset_list[trial_index]
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window

        trial_baseline_onset = baseline_onset_list[trial_index]
        trial_baseline_start = trial_baseline_onset + baseline_start
        trial_baseline_stop = trial_baseline_onset + baseline_stop

        if trial_baseline_start >= 0 and trial_baseline_stop < n_timepoints:
            if trial_start >= 0 and trial_stop < n_timepoints:

                baseline_data = df_matrix[trial_baseline_start:trial_baseline_stop]
                baseline_mean = np.mean(baseline_data, axis=0)

                trial_data = df_matrix[trial_start:trial_stop]
                trial_data = np.subtract(trial_data, baseline_mean)

                tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor
