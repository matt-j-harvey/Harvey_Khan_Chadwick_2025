import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats
from bisect import bisect_left


def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before



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



def get_sem_and_bounds(data):
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    upper_bound = np.add(mean, sem)
    lower_bound = np.subtract(mean, sem)
    return mean, upper_bound, lower_bound



def get_nearest_frames_to_onsets(onset_time_list, stack_onsets):

    stack_onsets = list(stack_onsets)

    frame_onset_list = []
    for onset in onset_time_list:
        closest_frame_time = take_closest(stack_onsets, onset)
        closest_frame_index = stack_onsets.index(closest_frame_time)
        frame_onset_list.append(closest_frame_index)

    return frame_onset_list
