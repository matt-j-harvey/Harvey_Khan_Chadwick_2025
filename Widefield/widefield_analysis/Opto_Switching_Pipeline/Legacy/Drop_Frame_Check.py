import os
import matplotlib.pyplot as plt
import h5py
import numpy as np

from Widefield_Utils import widefield_utils
from Preprocessing import Preprocessing_Utils

def perform_frame_check(base_directory):

    # Load Downsampled Mask Data - Check Frame Number
    data_file = os.path.join(base_directory, "Motion_Corrected_Mask_Data.hdf5")
    data_container = h5py.File(data_file, mode="r")
    blue_data = data_container["Blue_Data"]
    violet_data = data_container["Violet_Data"]

    blue_frames = np.shape(blue_data)[1]
    violet_frames = np.shape(violet_data)[1]


    # Load Ai Recorder and Extract Camera Frames
    ai_data = widefield_utils.load_ai_recorder_file(base_directory)
    stimuli_dict = widefield_utils.create_stimuli_dictionary()
    led_1_trace = ai_data[stimuli_dict["LED 1"]]
    led_2_trace = ai_data[stimuli_dict["LED 2"]]
    trigger_trace = ai_data[stimuli_dict["Camera Trigger"]]
    camers_frames = ai_data[stimuli_dict['Camera Frames']]
    led_1_onsets = Preprocessing_Utils.get_step_onsets(led_1_trace)
    n_led_1_onsets = len(led_1_onsets)

    print("Blue Frames", blue_frames, "Violet Frames", violet_frames, "LEd Onsets", n_led_1_onsets)

    """
    plt.plot(led_1_trace, c='b')
    plt.plot(led_2_trace, c='m')
    plt.plot(trigger_trace, c='g')
    plt.plot(camers_frames, c='orange')
    plt.scatter(led_1_onsets, np.ones(n_led_1_onsets))
    plt.show()
    """



