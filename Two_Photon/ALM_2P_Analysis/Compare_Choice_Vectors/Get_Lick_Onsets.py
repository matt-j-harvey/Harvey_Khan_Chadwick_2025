import os
import numpy as np


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


def get_downsampled_lick_trace(base_directory):

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Behaviour", "AI_Matrix.npy"))

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))

    # Load Stimuli Dict
    stimuli_dict = load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[:, stimuli_dict["Lick"]]

    # Downsample Lick Trace
    downsampled_lick_trace = downsample_ai_trace(lick_trace, stack_onsets)

    return downsampled_lick_trace


def get_next_onset(trace, onset, threshold, max_rt):

    index = 0
    above = False
    while above == False:
        instantaneous_value = trace[onset + index]
        if instantaneous_value > threshold:
            return onset + index

        else:
            index += 1
            if index > max_rt:
                return False


def get_lick_onsets(data_directory, output_directory, min_rt=0.5, max_rt=2.5, lick_threshold=600):

    # Load Lick Trace
    lick_trace = get_downsampled_lick_trace(data_directory)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_directory, "Frame_Rate.npy"))

    # Get Min and Max Rts
    min_rt_frames = min_rt * frame_rate
    max_rt_frames = max_rt * frame_rate

    # Load Visual Hit Onsets
    vis_1_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))

    # Load Odour Hit Onsets
    odour_1_cued_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "odour_1_cued_onsets.npy"))
    odour_1_non_cued_onsets = np.load(os.path.join(data_directory, "Stimuli_Onsets", "odour_1_not_cued_onsets.npy"))
    odour_onsets = list(odour_1_non_cued_onsets)  + list(odour_1_cued_onsets)

    # Get Visual and Odour Lick Onsets
    visual_lick_onsets = []
    for onset in vis_1_onsets:
        trial_lick_onset = get_next_onset(lick_trace, onset, lick_threshold, max_rt_frames)
        print(trial_lick_onset)
        if trial_lick_onset - onset > min_rt_frames:
            visual_lick_onsets.append(trial_lick_onset)

    odour_lick_onsets = []
    for onset in odour_onsets:
        trial_lick_onset = get_next_onset(lick_trace, onset, lick_threshold, max_rt_frames)
        if trial_lick_onset - onset > min_rt_frames:
            odour_lick_onsets.append(trial_lick_onset)

    # Save These
    save_directory = os.path.join(output_directory, "Lick_Onsets")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)


    np.save(os.path.join(save_directory, "Visual_Context_Lick_Onsets.npy"), visual_lick_onsets)
    np.save(os.path.join(save_directory, "Odour_Context_Lick_Onsets.npy"), odour_lick_onsets)

