import numpy as np
import matplotlib.pyplot as plt
import sys
import h5py
import os
import tables
from scipy import signal, ndimage, stats
from sklearn.neighbors import KernelDensity
import cv2
from matplotlib import gridspec

import Retinotopy_Utils


def get_stimuli_average(corrected_svt, u, indicies, stimuli_onsets, trial_details):

    #Get Data From All Trials
    all_trials = []
    for onset in stimuli_onsets:
        if onset != None:
            trial_data = get_single_trial_trace(onset, corrected_svt, trial_details)
            #trial_data = np.diff(trial_data, axis=0)
            #print("trial Data", np.shape(trial_data))
            all_trials.append(trial_data)
    all_trials = np.array(all_trials)
    all_trials = np.nan_to_num(all_trials)

    trial_average = np.mean(all_trials, axis=0)
    print("Trial Average Shape", np.shape(trial_average))

    trial_average = np.dot(u, np.transpose(trial_average))
    print("Trial Average Shape", np.shape(trial_average))


    image_height, image_width, trial_length = np.shape(trial_average)
    trial_average = np.reshape(trial_average, (image_height * image_width, trial_length))
    trial_average = np.transpose(trial_average)
    print("Trial Average Shape post transpose", np.shape(trial_average))

    trial_average = trial_average[:, indicies]
    print("Trial Average Shape post indicies", np.shape(trial_average))

    return trial_average




def get_single_trial_trace(onset, preprocessed_data, trial_details):

    #Get Trial Details
    trial_start     = trial_details[0]
    trial_end       = trial_details[1]
    window_size     = trial_details[2]

    window_start = onset + trial_start
    window_stop = onset + trial_end

    trial_data = []
    for timepoint in range(window_start, window_stop):
        window_data = preprocessed_data[timepoint - window_size : timepoint + window_size]
        window_mean = np.mean(window_data, axis=0)
        trial_data.append(window_mean)

    trial_data = np.array(trial_data)
    return trial_data



def save_evoked_responses(home_directory, name, matrix, type="Average"):

    # Check Stimuli Evoked Responses Directory Exists
    responses_save_location = home_directory + "/Stimuli_Evoked_Responses"
    if not os.path.exists(responses_save_location):
        os.mkdir(responses_save_location)

    #Create Save Directory
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    #Get File Name
    if type == "Average":
        filename = save_directory + "/" + name + "_Activity_Matrix_Average.npy"
    elif type == "All_Trials":
        filename = save_directory + "/" + name + "_Activity_Matrix_All_Trials.npy"
    np.save(filename, matrix)



def save_trial_details(home_directory, name, stimuli_onsets, trial_details):

    #Get Trial Details
    trial_start     = trial_details[0]
    trial_end       = trial_details[1]
    window_size     = trial_details[2]
    use_baseline    = trial_details[3]

    number_of_trials = np.shape(stimuli_onsets)[0]
    print("Number of trials", number_of_trials)

    current_trial_details = np.zeros((number_of_trials, 4), dtype=int)

    for trial in range(number_of_trials):
        onset = stimuli_onsets[trial]
        window_start = onset + trial_start
        window_stop = onset + trial_end

        current_trial_details[trial, 0] = int(window_start)
        current_trial_details[trial, 1] = int(window_stop)
        current_trial_details[trial, 2] = int(window_size)
        current_trial_details[trial, 3] = int(use_baseline)

    # Create Save Directory
    responses_save_location = home_directory + "/Stimuli_Evoked_Responses"
    save_directory = responses_save_location + "/" + name
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Get File Name
    filename = save_directory + "/" + name + "_Trial_Details.npy"
    np.save(filename, current_trial_details)





def extract_trial_aligned_activity(home_directory):

    # Trial Average Details
    trial_onset_filenames = ["Horizontal_Frame_Onsets.npy", "Vertical_Frame_Onsets.npy"]
    condition_names = ["Horizontal_Sweep", "Vertical_Sweep"]

    trial_start = 0
    trial_end = 300
    window_size = 2
    use_baseline = False
    trial_details = [trial_start, trial_end, window_size, use_baseline]

    responses_save_location = home_directory
    if not os.path.exists(responses_save_location):
        os.mkdir(responses_save_location)

    # Load Mask
    indicies, image_height, image_width = Retinotopy_Utils.load_generous_mask(home_directory)

    #Load Trial Onsets
    condition_1_onset_file = os.path.join(home_directory, "Stimuli_Onsets", trial_onset_filenames[0])
    condition_2_onset_file = os.path.join(home_directory, "Stimuli_Onsets", trial_onset_filenames[1])

    condition_1_onsets = np.load(condition_1_onset_file, allow_pickle=True)
    condition_2_onsets = np.load(condition_2_onset_file, allow_pickle=True)

    condition_1_onsets = np.ndarray.flatten(condition_1_onsets)
    condition_2_onsets = np.ndarray.flatten(condition_2_onsets)

    #Remove Nones
    condition_1_onsets = list(filter(None, condition_1_onsets))
    condition_2_onsets = list(filter(None, condition_2_onsets))

    print("Using FUll Size Data")
    corrected_svt = np.load(os.path.join(home_directory, "Preprocessed_Data", "Corrected_SVT.npy"))
    corrected_svt = np.transpose(corrected_svt)
    u = np.load(os.path.join(home_directory, "Preprocessed_Data", "U.npy"))

    print("corrected_svt Shape", np.shape(corrected_svt))
    print("condition_1_onsets", condition_1_onsets)

    # Extract Average Activity
    condition_1_average = get_stimuli_average(corrected_svt, u, indicies, condition_1_onsets, trial_details)
    save_evoked_responses(home_directory, condition_names[0], condition_1_average, type="Average")
    save_trial_details(home_directory, condition_names[0], condition_1_onsets, trial_details)

    # Repeat for condition 2
    condition_2_average = get_stimuli_average(corrected_svt, u, indicies, condition_2_onsets, trial_details)
    save_evoked_responses(home_directory, condition_names[1], condition_2_average, type="Average")
    save_trial_details(home_directory, condition_names[1], condition_2_onsets, trial_details)



