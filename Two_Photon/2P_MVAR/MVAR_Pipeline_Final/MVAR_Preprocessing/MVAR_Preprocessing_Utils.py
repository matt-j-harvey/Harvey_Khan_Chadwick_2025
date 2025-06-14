import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats


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



def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=5):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor



def load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints):

    onset_file_path = os.path.join(data_root_directory, session, "Stimuli_Onsets", onsets_file)
    raw_onsets_list = np.load(onset_file_path)

    checked_onset_list = []
    for trial_onset in raw_onsets_list:
        trial_start = trial_onset + start_window
        trial_stop = trial_onset + stop_window
        if trial_start > 0 and trial_stop < number_of_timepoints:
            checked_onset_list.append(trial_onset)

    return checked_onset_list





def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

