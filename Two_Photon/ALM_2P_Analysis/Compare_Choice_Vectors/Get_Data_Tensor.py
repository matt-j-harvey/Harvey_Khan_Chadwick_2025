import os

import numpy as np



def get_lick_tensors(data_directory, output_directory, start_window, stop_window):

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Activity_Tensors")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Df Matrix
    df_matrix = np.load(os.path.join(data_directory, "df_over_f_matrix.npy"))
    print("df matrix", np.shape(df_matrix))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_directory, "Frame_Rate.npy"))
    start_window_frames = int(start_window * frame_rate)
    stop_window_frames = int(stop_window * frame_rate)
    print("start_window_frames", start_window_frames)
    print("stop_window_frames", stop_window_frames)

    # Load Onsets
    vis_context_lick_onsets = np.load(os.path.join(output_directory, "Lick_Onsets", "Visual_Context_Lick_Onsets.npy"))
    odour_context_lick_onsets = np.load(os.path.join(output_directory, "Lick_Onsets", "Odour_Context_Lick_Onsets.npy"))
    combined_lick_onsets = np.concatenate([vis_context_lick_onsets, odour_context_lick_onsets])

    # Get Data Tensors
    visual_lick_tensor   = get_data_tensor(df_matrix, vis_context_lick_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction =True)
    odour_lick_tensor    = get_data_tensor(df_matrix, odour_context_lick_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction =True)
    combined_lick_tensor = get_data_tensor(df_matrix, combined_lick_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction =True)

    # Save These
    np.save(os.path.join(save_directory, "visual_lick_tensor.npy"), visual_lick_tensor)
    np.save(os.path.join(save_directory, "odour_lick_tensor.npy"), odour_lick_tensor)
    np.save(os.path.join(save_directory, "combined_lick_tensor.npy"), combined_lick_tensor)



def get_choice_tensors(data_directory, output_directory, start_window, stop_window):

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Activity_Tensors")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load Df Matrix
    df_matrix = np.load(os.path.join(data_directory, "df_over_f_matrix.npy"))
    print("df matrix", np.shape(df_matrix))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_directory, "Frame_Rate.npy"))
    start_window_frames = int(start_window * frame_rate)
    stop_window_frames = int(stop_window * frame_rate)
    print("start_window_frames", start_window_frames)
    print("stop_window_frames", stop_window_frames)

    # Load Onsets
    vis_context_stable_vis_1_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))
    vis_context_stable_vis_2_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))
    odour_1_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "Odour_1_onset_frames.npy"))
    odour_2_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "Odour_2_onset_frames.npy"))

    # Get Data Tensors
    vis_context_stable_vis_1_tensor = get_data_tensor(df_matrix, vis_context_stable_vis_1_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction=True)
    vis_context_stable_vis_2_tensor = get_data_tensor(df_matrix, vis_context_stable_vis_2_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction=True)
    odour_1_tensor = get_data_tensor(df_matrix, odour_1_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction=True)
    odour_2_tensor = get_data_tensor(df_matrix, odour_2_onsets, start_window=start_window_frames, stop_window=stop_window_frames, baseline_correction=True)

    # Save These
    np.save(os.path.join(save_directory, "vis_context_stable_vis_1_tensor.npy"), vis_context_stable_vis_1_tensor)
    np.save(os.path.join(save_directory, "vis_context_stable_vis_2_tensor.npy"), vis_context_stable_vis_2_tensor)
    np.save(os.path.join(save_directory, "odour_1_tensor.npy"), odour_1_tensor)
    np.save(os.path.join(save_directory, "odour_2_tensor.npy"), odour_2_tensor)





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
