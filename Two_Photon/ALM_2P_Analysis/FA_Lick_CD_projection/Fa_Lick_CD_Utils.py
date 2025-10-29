import os
import numpy as np
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


def get_vis_hit_lick_onsets(behaviour_matrix):
    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_lick_onset = trial[22]

        if trial_type == 1:
            if trial_correct == 1:
                onset_list.append(trial_lick_onset)
    return onset_list


def get_odour_hit_lick_onsets(behaviour_matrix):
    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        ignore_irrel = trial[7]
        trial_lick_onset = trial[22]

        if trial_type == 3:
            if trial_correct == 1:
                if ignore_irrel == 1:
                    onset_list.append(trial_lick_onset)

    return onset_list


def get_cr_vis_onset_frames(behaviour_matrix):
    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_onset = trial[18]

        if trial_type == 2:
            if trial_correct == 1:
                    onset_list.append(trial_onset)

    return onset_list


def get_fa_vis_onset_frames(behaviour_matrix):
    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_onset = trial[18]

        if trial_type == 2:
            if trial_correct == 0:
                    onset_list.append(trial_onset)

    return onset_list



def get_nearest_frames_to_onsets(onset_time_list, stack_onsets):

    stack_onsets = list(stack_onsets)

    frame_onset_list = []
    for onset in onset_time_list:
        closest_frame_time = take_closest(stack_onsets, onset)
        closest_frame_index = stack_onsets.index(closest_frame_time)
        frame_onset_list.append(closest_frame_index)

    return frame_onset_list



