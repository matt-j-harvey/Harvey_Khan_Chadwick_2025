import os
import numpy as np
import matplotlib.pyplot as plt

import Two_Photon_Utils
import Get_Data_Tensor


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


def get_onsets(downsampled_trace, threhold, preceeding_window):

    n_timepoints = len(downsampled_trace)
    lick_onsets = []
    for timepoint_index in range(preceeding_window, n_timepoints):

        timepoint_data = downsampled_trace[timepoint_index]
        timepoint_preceeding_window = downsampled_trace[timepoint_index-preceeding_window:timepoint_index]

        if timepoint_data > threhold:
            if np.max(timepoint_preceeding_window) < threhold:
                lick_onsets.append(timepoint_index)

    return lick_onsets



def get_lick_onsets(base_directory, lick_threshold=600):

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Behaviour", "AI_Matrix.npy"))
    print("ai_data", np.shape(ai_data))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    print("frame_rate", frame_rate)

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))
    print("stack_onsets", len(stack_onsets))

    # Load Stimuli Dict
    stimuli_dict = Two_Photon_Utils.load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[:, stimuli_dict["Lick"]]

    # Downsample Lick Trace
    downsampled_lick_trace = downsample_ai_trace(lick_trace, stack_onsets)

    # Get onsets
    preceeding_window = int(np.ceil(2*frame_rate))
    print("preceeding_window", preceeding_window)
    lick_onsets = get_onsets(downsampled_lick_trace, lick_threshold, preceeding_window=preceeding_window)
    print("n licks", len(lick_onsets))

    # Save Onset Frames
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"), lick_onsets)

    """
    plt.plot(downsampled_lick_trace)
    for onset in lick_onsets:
        plt.axvline(onset, c='k', alpha=0.5, linestyle='dashed')

    plt.show()
    """
    # Plot Lick Traces
    """
    lick_trace = lick_trace[stack_onsets[0]:]
    full_x_values = np.linspace(start=0, stop=1, num=len(lick_trace))
    downsampled_x_values = np.linspace(start=0, stop=1, num=len(downsampled_lick_trace))

    # Get Lick Onsets
    plt.plot(full_x_values, lick_trace)
    plt.plot(downsampled_x_values, downsampled_lick_trace)
    plt.axhline(lick_threshold)
    plt.show()
    """




session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]

session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK64.1B\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1A\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1B\2024_09_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK72.1E\2024_08_23_Switching",
]


for base_directory in session_list:
    get_lick_onsets(base_directory)