import numpy as np
import os


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
11 stimuli_onset	
12 stimuli_offset	
13 irrel_onset	
14 irrel_offset	
15 trial_end	
16 Photodiode Onset	
17 Photodiode Offset	
18 Onset closest Frame	
19 Offset Closest Frame	
20 Irrel Onset Closest Frame	
21 Irrel Offset Closest Frame	
22 Opto Trial	
23 Opto Pattern Label
"""


def extract_cr_fa_onsets(data_directory_root, session):

    # Create Empty Lists
    cr_onsets = []
    fa_onsets = []

    # Load behaviour matrix
    behaviour_matrix = np.load(os.path.join(data_directory_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_stim_onset = trial[18]

        # Visual Context Trials
        if trial_type == 2:
            if trial_stim_onset != None:

                if trial_correct == 1:
                    cr_onsets.append(trial_stim_onset)
                else:
                    fa_onsets.append(trial_stim_onset)

    return cr_onsets, fa_onsets

