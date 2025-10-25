import numpy as np
import os




def get_vis_1_hit_onsets(behaviour_matrix):

    vis_1_hit_onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_outcome = trial[3]
        trial_onset = trial[18]
        opto_trial = trial[22]

        if trial_onset != None:
            if opto_trial == False:
                if trial_type == 1:
                    if trial_outcome == 1:
                        vis_1_hit_onset_list.append(trial_onset)

    return vis_1_hit_onset_list



def create_stimuli_dictionary():
    channel_index_dictionary = {
        "Photodiode": 0,
        "Reward": 1,
        "Lick": 2,
        "Visual 1": 3,
        "Visual 2": 4,
        "Odour 1": 5,
        "Odour 2": 6,
        "Irrelevance": 7,
        "Running": 8,
        "Trial End": 9,
        "Camera Trigger": 10,
        "Camera Frames": 11,
        "LED 1": 12,
        "LED 2": 13,
        "Mousecam": 14,
        "Optogenetics": 15,
    }
    return channel_index_dictionary




def get_reaction_time(lick_trace, vis_onset, lick_threshold, max_window):

    n_timpoints = len(lick_trace)
    for time_delta in range(max_window):
        if vis_onset + time_delta < n_timpoints:
            if lick_trace[vis_onset + time_delta] >= lick_threshold:
                return time_delta

    return None



def create_rt_time_matrix(vis_1_hit_onsets, lick_trace, lick_threshold, max_window=69):

    rt_time_matrix = []
    for onset in vis_1_hit_onsets:

        reaction_time = get_reaction_time(lick_trace, onset, lick_threshold, max_window)
        if reaction_time != None:
            rt_time_matrix.append([onset, onset + reaction_time, reaction_time * 36])

    rt_time_matrix = np.array(rt_time_matrix)
    return rt_time_matrix



def create_hit_rt_matrix(base_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(base_directory, "Lick_Threshold.npy"))

    # Load Downsampled AI Matrix
    downsampled_ai_matrix = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))

    # Load Stim Dict
    stimuli_dict = create_stimuli_dictionary()

    # Get Lick Trace
    lick_trace = downsampled_ai_matrix[stimuli_dict["Lick"]]

    # Get Vis 1 Hits
    vis_1_hit_onset_list = get_vis_1_hit_onsets(behaviour_matrix)

    # Create hit RT Matrix
    hit_rt_matrix = create_rt_time_matrix(vis_1_hit_onset_list, lick_trace, lick_threshold, max_window=69)

    return hit_rt_matrix


