import numpy as np
import os




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
    print("ai_data", np.shape(ai_data))

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))
    print("stack_onsets", len(stack_onsets))

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



def get_trial_timings(stimuli_onsets, lick_trace, lick_threshold, min_rt_frames, max_rt_frames):

    trial_start_frames = []
    trial_response_frames = []

    for onset in stimuli_onsets:
        trial_lick_onset = get_next_onset(lick_trace, onset, lick_threshold, max_rt_frames)
        print(trial_lick_onset)
        reaction_time = trial_lick_onset - onset
        if reaction_time > min_rt_frames:
            if reaction_time <= max_rt_frames:
                trial_start_frames.append(onset)
                trial_response_frames.append(trial_lick_onset)

    return trial_start_frames, trial_response_frames



def get_choice_timings(data_root, session, output_root, min_rt_frames, max_rt_frames, lick_threshold=600):

    # to 2.5 Seconds b

    # Load Lick Trace
    lick_trace = get_downsampled_lick_trace(os.path.join(data_root, session))

    # Load Visual Hit Onsets
    vis_1_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))

    # Load Odour Hit Onsets
    odour_1_cued_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "odour_1_cued_onsets.npy"))
    odour_1_non_cued_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "odour_1_not_cued_onsets.npy"))
    odour_onsets = list(odour_1_non_cued_onsets) + list(odour_1_cued_onsets)

    # Get Response Times
    vis_1_stimuli_onsets, vis_1_response_onsets = get_trial_timings(vis_1_onsets, lick_trace, lick_threshold, min_rt_frames, max_rt_frames)
    odr_1_stimuli_onsets, odr_1_response_onsets = get_trial_timings(odour_onsets, lick_trace, lick_threshold, min_rt_frames, max_rt_frames)

    # Save These
    save_directory = os.path.join(output_root, session, "Trial_Timings")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "vis_1_stimuli_onsets.npy"), vis_1_stimuli_onsets)
    np.save(os.path.join(save_directory, "vis_1_response_onsets.npy"), vis_1_response_onsets)
    np.save(os.path.join(save_directory, "odr_1_stimuli_onsets.npy"), odr_1_stimuli_onsets)
    np.save(os.path.join(save_directory, "odr_1_response_onsets.npy"), odr_1_response_onsets)



data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Response_Dimension_Comparison"

session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
    ]

for session in session_list:
    get_choice_timings(data_root, session, output_root, min_rt_frames=3, max_rt_frames=18, lick_threshold=600)