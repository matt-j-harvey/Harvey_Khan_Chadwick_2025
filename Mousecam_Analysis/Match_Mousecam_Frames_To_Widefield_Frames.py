import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import Session_List
import Mousecam_Utils



def check_mousecam_drop_frames(trigger_onsets, base_directory, video_name):

    n_triggers = len(trigger_onsets)
    video_filepath = os.path.join(base_directory, video_name)
    n_frames = Mousecam_Utils.get_n_video_frames(video_filepath)
    if n_triggers == n_frames:
        #print("Yep! :) ", "n_frames: ", n_frames, "n_triggers: ", n_triggers)
        pass
    else:
        print("nope :(",  "n_frames: ", n_frames, "n_triggers: ", n_triggers, "base directory", base_directory)


def get_mousecam_frametimes(base_directory):

    # Load Ai Recorder Data
    ai_data = Mousecam_Utils.load_ai_recorder_file(base_directory)

    # Create Stimuli Dict
    stimuli_dict = Mousecam_Utils.create_stimuli_dictionary()

    # Extract Mousecam Trace
    mousecam_trace = ai_data[stimuli_dict["Mousecam"]]

    # Get Step Onsets
    trigger_onsets = Mousecam_Utils.get_step_onsets(mousecam_trace, threshold=3, window=1)

    # Get Bodycam Video Name
    video_name = Mousecam_Utils.get_bodycam_filename(base_directory)

    # Check Mousecam Dropped Frames
    #check_mousecam_drop_frames(trigger_onsets, base_directory, video_name)

    # Save Frame Times
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), trigger_onsets)


def match_mousecam_to_widefield_frames(base_directory):

    # Get Mousecam Frame Times
    get_mousecam_frametimes(base_directory)

    # Load Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = list(mousecam_frame_times)
    widefield_frame_times = Mousecam_Utils.invert_dictionary(widefield_frame_times)
    widefield_frame_time_keys = list(widefield_frame_times.keys())

    # Get Number of Frames
    number_of_widefield_frames = len(widefield_frame_time_keys)

    # Dictionary - Keys are Widefield Frame Indexes, Values are Closest Mousecam Frame Indexes
    widfield_to_mousecam_frame_dict = {}

    for widefield_frame_index in range(number_of_widefield_frames):
        frame_time = widefield_frame_times[widefield_frame_index]
        closest_mousecam_time = Mousecam_Utils.take_closest(mousecam_frame_times, frame_time)
        closest_mousecam_frame = mousecam_frame_times.index(closest_mousecam_time)
        widfield_to_mousecam_frame_dict[widefield_frame_index] = closest_mousecam_frame

    # Save Directory
    save_directoy = os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy")
    np.save(save_directoy, widfield_to_mousecam_frame_dict)



# Load Session List
session_list = Session_List.nested_session_list
session_list = Session_List.flatten_nested_list(session_list)


session_list = Session_List.control_learning_session_list
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/Expansion/Control_Data"

session_list = Session_List.neurexin_learning_session_list
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

for session in tqdm(session_list):
    match_mousecam_to_widefield_frames(os.path.join(data_root, session))