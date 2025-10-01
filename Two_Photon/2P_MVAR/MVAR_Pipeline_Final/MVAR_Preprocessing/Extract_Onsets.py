import numpy as np
import os
import matplotlib.pyplot as plt

import MVAR_Preprocessing_Utils

"""
0 trial_index	
1 trial_type	
2 lick	
3 correct	
4 rewarded	
5 preeceded_by_irrel	
6 irrel_type	
7 ignore_irrel	
8 block_number	
9 first_in_block	
10 in_block_of_stable_performance	
11 onset	
12 stimuli_offset	
13 irrel_onset	
14 irrel_offset	
15 trial_end	
16 photodiode_onset	
17 photodiode_offset	
18 onset_closest_frame	
19 offset_closest_frame	
20 irrel_onset_closest_frame	
21 irrel_offset_closest_frame	
22 lick_onset	
23 reaction_time	
24 reward_onset
"""

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


def extract_lick_onsets(data_root_directory, session, mvar_output_directory):

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(data_root_directory, session, "Behaviour", "Lick_Threshold.npy"))

    # Load Downsampled AI Matrix
    ai_data = np.load(os.path.join(mvar_output_directory, session, "Behaviour", "Downsampled_AI_Matrix_Framewise.npy"))

    # Load Stimuli Dict
    stimuli_dict = MVAR_Preprocessing_Utils.load_rig_1_channel_dict()

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root_directory, session, "Frame_Rate.npy"))

    # Get Lick Trace
    downsampled_lick_trace = ai_data[stimuli_dict["Lick"]]

    # Get onsets
    preceeding_window = int(np.ceil(1.5 * frame_rate))
    lick_onsets = get_onsets(downsampled_lick_trace, lick_threshold, preceeding_window=preceeding_window)

    """
    plt.plot(downsampled_lick_trace)
    plt.scatter(lick_onsets, np.multiply(np.ones(len(lick_onsets)), lick_threshold), c='k')
    plt.show()
    """

    # Save Onset Frames
    np.save(os.path.join(mvar_output_directory, session, "Behaviour", "Lick_Onset_Frames.npy"), lick_onsets)




def extract_odour_onsets(base_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    n_trials = np.shape(behaviour_matrix)[0]
    print("n trials", n_trials)

    odour_1_onsets = []
    odour_2_onsets = []
    for trial_index in range(n_trials):
        trial_data = behaviour_matrix[trial_index]
        trial_type = trial_data[1]
        correct = trial_data[3]
        onset_closest_frame	= trial_data[18]
        ignore_irrel = trial_data[7]

        if onset_closest_frame != None:

            if correct == True:
                if ignore_irrel != False:

                    if trial_type == 3:
                        odour_1_onsets.append(onset_closest_frame)

                    elif trial_type == 4:
                        odour_2_onsets.append(onset_closest_frame)

    # Save These
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Odour_1_onset_frames.npy"), odour_1_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Odour_2_onset_frames.npy"), odour_2_onsets)

