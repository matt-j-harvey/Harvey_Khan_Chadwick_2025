import numpy as np

def get_activity_tensors(delta_f_matrix, onsets_list, start_window, stop_window, baseline_correct, baseline_window):

    n_timepoints = np.shape(delta_f_matrix)[0]

    activity_tensor = []
    for trial_onset in onsets_list:
        trial_start = int(trial_onset + start_window)
        trial_stop = int(trial_onset + stop_window)

        if trial_start > 0 and trial_stop < n_timepoints:
            trial_data = delta_f_matrix[trial_start:trial_stop]

            if baseline_correct == True:
                trial_baseline = trial_data[0:baseline_window]
                trial_baseline = np.mean(trial_baseline, axis=0)
                trial_data = np.subtract(trial_data, trial_baseline)

            activity_tensor.append(trial_data)

    activity_tensor = np.array(activity_tensor)
    return activity_tensor
