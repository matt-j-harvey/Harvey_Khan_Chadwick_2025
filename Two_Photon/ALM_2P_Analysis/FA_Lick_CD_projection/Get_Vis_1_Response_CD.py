import numpy as np
import os

import Get_Data_Tensor



def get_hits(behaviour_matrix):

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        onset_frame = trial[18]

        if trial_type == 1:
            if correct == 1:
                onset_list.append(onset_frame)

    return onset_list


def get_vis_1_response_cd(df_matrix, behaviour_matrix):


    # Get Vis 1 Onsets
    vis_1_onsets = get_hits(behaviour_matrix)

    # Get Data Tensor
    start_window = -16
    stop_window = 16
    baseline_correct = True
    baseline_window = 3
    tensor = Get_Data_Tensor.get_activity_tensors(df_matrix, vis_1_onsets, start_window, stop_window, baseline_correct, baseline_window)
    print("tensor", np.shape(tensor))

    # Get Mean
    mean_activity = np.mean(tensor, axis=0)
    print("Mean activity shape", np.shape(mean_activity))

    # Get Pre and Post Window
    prestim_window = mean_activity[np.abs(start_window)-3:np.abs(start_window)]
    prestim_window = np.mean(prestim_window, axis=0)

    post_stim_window = mean_activity[np.abs(start_window):np.abs(start_window) + 3]
    post_stim_window = np.mean(post_stim_window, axis=0)
    delta = np.subtract(post_stim_window, prestim_window)

    return delta