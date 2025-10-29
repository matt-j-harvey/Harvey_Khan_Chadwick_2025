import os
import tables
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Widefield_Utils import widefield_utils



def get_step_onsets(trace, threshold=1, window=3):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times


def get_offset(onset, stream, threshold=1):

    count = 0
    on = True
    while on:
        if onset + count < len(stream):
            if stream[onset + count] < threshold and count > 10:
                on = False
                return onset + count
            else:
                count += 1

        else:
            return np.nan


def check_not_in_post_opto_window(frame_onset, opto_offsets, post_opto_window=28):
    for offset in opto_offsets:
        if frame_onset >= (offset) and frame_onset < (offset + post_opto_window):
            return False
    return True


def visualise_excluded_frames(included_frames, opto_frace, opto_offsets):
    plt.scatter(included_frames, np.ones(len(included_frames)), c='orange')
    plt.scatter(opto_offsets, np.ones(len(opto_offsets))*0.8, c='r')
    plt.plot(opto_frace)
    plt.show()



def get_opto_offsets(opto_onsets, opto_trace):
    opto_offsets = []
    for onset in opto_onsets:
        corresponding_offset = get_offset(onset, opto_trace)
        opto_offsets.append(corresponding_offset)
    return opto_offsets




def get_frames_outside_of_opto(base_directory, opto_trace_threshold=1):

    """
    Criteria:
    Not during an opto stim
    Not in the 1s following an opto stim -(rebound excitation)
    """
    # Set Save Directory
    save_directory = os.path.join(base_directory, "Stimuli_Onsets")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Downsampled_AI_Matrix_Framewise.npy"))
    print("ai_data", np.shape(ai_data))
    n_frames = np.shape(ai_data)[1]

    # Create Stimuli Dict
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Get Opto Onsets
    opto_trace = ai_data[stimuli_dictionary["Optogenetics"]]
    opto_onsets = get_step_onsets(opto_trace)

    # Get Opto Offsets
    opto_offsets = get_opto_offsets(opto_onsets, opto_trace)

    # Exclude Frames Within Or Shortly After Opto
    included_frames = []
    for frame in range(n_frames-1):

        # Check Its Not During An Opto Stimulus
        if opto_trace[frame] < opto_trace_threshold:
            if check_not_in_post_opto_window(frame, opto_offsets) == True:
                included_frames.append(frame)

    # Save These
    np.save(os.path.join(save_directory, "Frames_Outside_Opto_Window.npy"), included_frames)

    # Visualise These As Sanity Check
    #visualise_excluded_frames(included_frames, opto_trace, opto_offsets)





session_list = [

     "KPGC11.1C/2024_08_22_Switching_V1_Pre_03",
     "KPGC11.1C/2024_08_23_Switching_PPC_Pre_03",
     "KPGC11.1C/2024_08_26_Switching_ProxM_Pre_03",
     "KPGC11.1C/2024_08_28_Switching_PM_Pre_03",
     "KPGC11.1C/2024_08_30_Switching_MM_Pre_03",
     "KPGC11.1C/2024_09_03_Switching_RSC_Pre_03",
     "KPGC11.1C/2024_09_11_Switching_ALM_Pre_03",
     "KPGC11.1C/2024_09_17_Switching_SS_Pre_03",
     "KPGC11.1C/2024_09_20_Switching_V1_Pre_03",

     "KPGC12.2A/2024_09_11_Switching_MM_Pre_03",
     "KPGC12.2A/2024_09_16_Switching_V1_Pre_03",
     "KPGC12.2A/2024_09_17_Switching_PPC_Pre_03",
     "KPGC12.2A/2024_09_18_Switching_MM_Pre_03",
     "KPGC12.2A/2024_09_19_Switching_SS_Pre_03",
     "KPGC12.2A/2024_09_20_Switching_PM_Pre_03",
     "KPGC12.2A/2024_09_25_Switching_RSC_Pre_03",
     "KPGC12.2A/2024_09_28_Switching_ALM_Pre_03",
     "KPGC12.2A/2024_09_29_Switching_ProxM_Pre_03",

     "KPGC12.2B/2024_09_10_Switching_V1_Pre_03",
     "KPGC12.2B/2024_09_11_Switching_MM_Pre_03",
     "KPGC12.2B/2024_09_17_Switching_RSC_Pre_03",
     "KPGC12.2B/2024_09_18_Switching_MM_Pre_03",
     "KPGC12.2B/2024_09_19_Switching_SS_Pre_03",
     "KPGC12.2B/2024_09_20_Switching_PPC_Pre_03",
     "KPGC12.2B/2024_09_24_Switching_PM_Pre_03",
     "KPGC12.2B/2024_09_26_Switching_ALM_Pre_03",
     "KPGC12.2B/2024_09_28_Switching_ProxM_Pre_03",

     #"KPGC3.3E/2023_07_03_Switch_V1_1F_03_Pre",
     #"KPGC3.3E/2023_07_06_Switch_MM_1F_03_Pre",
     #"KPGC3.3E/2023_07_14_Switch_ALM_1F_03_Pre",
     #"KPGC3.3E/2023_07_19_Switch_RSC_1F_03_Pre",
     "KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_28_Switch_V1_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
     "KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
     "KPGC3.3E/2023_08_07_Switch_SS_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre",

     #"KPGC6.2E/2023_07_27_Switch_MM_1F_04_1S_Pre",
     "KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_18_Switch_ALM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_21_Switch_BC_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre",

]




control_session_list = [

    "KPGC12.3B/2024_09_03_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_05_Switching_PPC_Pre_03",
     "KPGC12.3B/2024_09_09_Switching_MM_Pre_03",
     "KPGC12.3B/2024_09_10_Switching_RSC_Pre_03",
     "KPGC12.3B/2024_09_12_Switching_SS_Pre_03",
     "KPGC12.3B/2024_09_16_Switching_ALM_Pre_03",
     "KPGC12.3B/2024_09_17_Switching_PM_Pre_03",
     "KPGC12.3B/2024_09_18_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_19_Switching_Pre_03",
     "KPGC12.3B/2024_09_20_Switching_MM_Pre_03",
     "KPGC12.3B/2024_09_23_Switching_SS_Pre_03",
     "KPGC12.3B/2024_09_28_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_29_Switching_ProxM_Pre_03",

    #"KPGC1.3A/2023_07_07_Switch_V1_1F_03_Pre",
     #"KPGC1.3A/2023_07_13_Switch_MM_1F_03_Pre",
     #"KPGC1.3A/2023_07_18_Switch_ALM_1F_03_Pre",
     #"KPGC1.3A/2023_07_20_Switch_RSC_1F_03_Pre",
     "KPGC1.3A/2023_07_27_Switch_MM_1f_04_1S_Pre",
     "KPGC1.3A/2023_08_01_Switch_V1_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_03_Switch_RSC_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_08_Switch_PM_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_10_Switch_ALM_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_22_Switch_V1_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_24_Switch_PPC_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_28_Switch_BC_1F_04_1S_Pre",

    "KPGC3.4A/2023_08_24_Switch_V1_1F_04_1S_Pre",
     "KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
     "KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",

    "KPGC7.4A/2023_08_25_Switch_V1_1F_04_1S_Pre",
     "KPGC7.4A/2023_08_28_Switch_ALM_1F_04_1S_Pre",
     "KPGC7.4A/2023_08_30_Switch_PM_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_01_Switch_V1_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_04_Switch_MM_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_06_Switch_BC_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_08_Switch_RSC_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
]

data_root_directory = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"

for session in tqdm(control_session_list):
    base_directory = os.path.join(data_root_directory, session)
    get_frames_outside_of_opto(base_directory, opto_trace_threshold=1)