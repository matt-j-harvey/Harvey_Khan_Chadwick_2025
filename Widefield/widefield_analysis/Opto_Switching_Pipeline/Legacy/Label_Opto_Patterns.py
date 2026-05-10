import numpy as np
import tables
import os


def get_opto_stim_log_file(base_directory):
    file_list = os.listdir(base_directory)
    #print("Base directory", base_directory)
    #print("File list", file_list)
    for file_name in file_list:
        if "Opto_Stim_Log.h5" in file_name:
            return file_name

def load_opto_log_file(base_directory):

    opto_log_filename = get_opto_stim_log_file(base_directory)
    #print("Opto log filename", opto_log_filename)
    opto_log_file = tables.open_file(os.path.join(base_directory, opto_log_filename), mode="r")
    opto_log_stim_images = np.array(opto_log_file.root["Stim_Images"])
    opto_log_timestamps = np.array(opto_log_file.root["Timestamps"])
    opto_log_file.close()
    return opto_log_timestamps, opto_log_stim_images



def get_stimulus_label(stimulus, unique_stimuli):
    number_of_unique_stimuli = len(unique_stimuli)
    for stimulus_index in range(number_of_unique_stimuli):
        if np.array_equal(stimulus, unique_stimuli[stimulus_index]):
            return stimulus_index


def get_stimuli_labels(opto_log_stim_images):

    unique_stimuli = list(np.unique(opto_log_stim_images, axis=0))
    number_of_unique_stimuli = len(unique_stimuli)

    stimuli_labels_list = []
    for stimuli in opto_log_stim_images:
        stimulus_label = get_stimulus_label(stimuli, unique_stimuli)
        stimuli_labels_list.append(stimulus_label)


    return number_of_unique_stimuli, stimuli_labels_list


def label_opto_patterns(base_directory):

    # Load Opto stim log file
    opto_log_timestamps, opto_log_stim_images = load_opto_log_file(base_directory)

    # Get Stim Labels
    number_of_unique_stimuli, stimuli_labels_list = get_stimuli_labels(opto_log_stim_images)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Opto_Pattern_Labels.npy"), stimuli_labels_list)
    #print(stimuli_labels_list)


