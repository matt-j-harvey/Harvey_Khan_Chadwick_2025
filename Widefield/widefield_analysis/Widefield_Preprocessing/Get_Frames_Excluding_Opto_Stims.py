import os
import tables
import numpy as np
import matplotlib.pyplot as plt

from Behaviour_Analysis import Behaviour_Utils
from Widefield_Utils import widefield_utils


def check_not_in_post_opto_window(frame_onset, opto_offsets, post_opto_window=1000):
    for offset in opto_offsets:
        if frame_onset >= (offset) and frame_onset < (offset + post_opto_window):
            return False
    return True


def visualise_excluded_frames(included_frames, frame_onsets, opto_frace, opto_offsets):
    frame_onsets = np.array(frame_onsets)
    selected_onsets = frame_onsets[included_frames]
    plt.scatter(selected_onsets, np.ones(len(selected_onsets)), c='orange')
    plt.scatter(opto_offsets, np.ones(len(opto_offsets))*0.8, c='r')
    plt.plot(opto_frace)
    plt.show()



def get_opto_offsets(opto_onsets, opto_trace):
    opto_offsets = []
    for onset in opto_onsets:
        corresponding_offset = Behaviour_Utils.get_offset(onset, opto_trace)
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
    ai_data = Behaviour_Utils.load_ai_recorder_file(base_directory)
    stimuli_dictionary = widefield_utils.create_stimuli_dictionary()

    # Get Frame Times
    blue_led_trace = ai_data[stimuli_dictionary["LED 1"]]
    frame_onset_list = Behaviour_Utils.get_step_onsets(blue_led_trace)
    np.save(os.path.join(save_directory, "Frame_Onsets.npy"), frame_onset_list)
    print("Frame Onsets", np.shape(frame_onset_list))

    # Get Opto Onsets
    opto_trace = ai_data[stimuli_dictionary["Optogenetics"]]
    opto_onsets = Behaviour_Utils.get_step_onsets(opto_trace)
    np.save(os.path.join(save_directory, "Opto_Onset_Times.npy"), opto_onsets)

    # Get Opto Offsets
    opto_offsets = get_opto_offsets(opto_onsets, opto_trace)
    np.save(os.path.join(save_directory, "Opto_Offset_Times.npy"), opto_offsets)

    # Match Opto Onsets To Frame Times
    opto_frames = []
    for opto_onset_time in opto_onsets:
        closest_frame_time = widefield_utils.take_closest(frame_onset_list, opto_onset_time)
        closest_frame = frame_onset_list.index(closest_frame_time)
        opto_frames.append(closest_frame)
    np.save(os.path.join(save_directory, "Opto_Onset_Frames.npy"), opto_frames)

    print("Opto Onset Frames", len(opto_frames))

    # Exclude Frames Within Or Shortly After Opto
    included_frames = []
    n_frames = len(frame_onset_list)
    for frame_index in range(n_frames):
        frame_onset = frame_onset_list[frame_index]

        # Check Its Not During An Opto Stimulus
        if opto_trace[frame_onset] < opto_trace_threshold:
            if check_not_in_post_opto_window(frame_onset, opto_offsets) == True:
                included_frames.append(frame_index)

    print("Number Of Frames", np.shape(included_frames))

    # Save These
    np.save(os.path.join(base_directory, "Frames_Outside_Opto_Window.npy"), included_frames)

    # Visualise These As Sanity Check
    #visualise_excluded_frames(included_frames, frame_onset_list, opto_trace, opto_offsets)
