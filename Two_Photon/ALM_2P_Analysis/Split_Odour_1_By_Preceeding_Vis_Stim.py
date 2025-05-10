import numpy as np
import os
import matplotlib.pyplot as plt


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


def split_odour_1_by_preceeding_vis_stim(base_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # If Vis 1, if Hit, if Preceeded by dsiaul stimulus
    odour_1_preceeded_by_vis_1 = []
    odour_1_preceeded_by_vis_2 = []

    n_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(n_trials):
        trial_data = behaviour_matrix[trial_index]
        trial_type = trial_data[1]
        correct = trial_data[3]
        preceeded_by_irrel = trial_data[5]
        irrel_type = trial_data[6]
        ignore_irrel = trial_data[7]
        stimuli_onset_frame = trial_data[18]


        if trial_type == 3:
            if correct == 1:
                if preceeded_by_irrel == 1:
                    if ignore_irrel == 1:

                        if irrel_type == 1:
                            odour_1_preceeded_by_vis_1.append(stimuli_onset_frame)
                        elif irrel_type == 2:
                            odour_1_preceeded_by_vis_2.append(stimuli_onset_frame)

    print("odour_1_preceeded_by_vis_1", odour_1_preceeded_by_vis_1)
    print("odour_1_preceeded_by_vis_2", odour_1_preceeded_by_vis_2)
    print(len(odour_1_preceeded_by_vis_1), len(odour_1_preceeded_by_vis_2))

    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_1_preceeded_by_vis_1.npy"), odour_1_preceeded_by_vis_1)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "odour_1_preceeded_by_vis_2.npy"), odour_1_preceeded_by_vis_2)

session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]

for base_directory in session_list:
    split_odour_1_by_preceeding_vis_stim(base_directory)