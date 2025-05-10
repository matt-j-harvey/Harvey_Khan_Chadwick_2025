import os

import matplotlib.pyplot as plt
import numpy as np

import Two_Photon_Utils


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


def get_rt_time(onset, downsampled_lick_trace, period, lick_threshold):

    n_timepoints = len(downsampled_lick_trace)
    licked = False
    count = 0
    while licked == False:
        if downsampled_lick_trace[onset + count] > lick_threshold:
            return count * period

        else:
            count += 1
            if count == n_timepoints:
                return None





def split_hits_by_rt(base_directory, lick_threshold=600):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1)/frame_rate
    period = np.multiply(period, 1000)

    # Load Vis 1 Hits
    vis_1_hits = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))
    print("stack_onsets", len(stack_onsets))

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Behaviour", "AI_Matrix.npy"))
    print("ai_data", np.shape(ai_data))

    # Load Stimuli Dict
    stimuli_dict = Two_Photon_Utils.load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[:, stimuli_dict["Lick"]]

    # Downsample Lick Trace
    downsampled_lick_trace = downsample_ai_trace(lick_trace, stack_onsets)

    hit_rt_list = []
    rt_dist = []
    for onset in vis_1_hits:
        trial_rt = get_rt_time(onset, downsampled_lick_trace, period, lick_threshold)
        hit_rt_list.append([onset, trial_rt])
        rt_dist.append(trial_rt)

    hit_rt_list = np.array(hit_rt_list)
    print("hit_rt_list", np.shape(hit_rt_list))

    print("rt_dist", rt_dist)
    #plt.title(base_directory)
    #plt.hist(rt_dist)
    #plt.show()

    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Hits_By_RT.npy"), hit_rt_list)

    return rt_dist

session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]

session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK64.1B\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1A\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1B\2024_09_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK72.1E\2024_08_23_Switching",
]


group_rt_dist = []
for session in session_list:
    rt_dist = split_hits_by_rt(session)
    group_rt_dist.append(rt_dist)


group_rt_dist = np.concatenate(group_rt_dist)
plt.hist(group_rt_dist, bins=list(range(500, 3000, 250)))
plt.show()

