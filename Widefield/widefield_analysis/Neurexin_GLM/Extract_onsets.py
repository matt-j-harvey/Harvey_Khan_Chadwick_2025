import numpy as np
import os

import GLM_Utils

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


def extract_onsets(data_directory_root, session, output_directory_root):

    # Load behaviour matrix
    behaviour_matrix = np.load(os.path.join(data_directory_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get All vis 1
    vis_1_onsets = GLM_Utils.get_vis_1_onsets(behaviour_matrix)

    # Get All Vis 2
    vis_2_onsets = GLM_Utils.get_vis_2_onsets(behaviour_matrix)

    # Get Hit Onset Frames
    vis_1_correct_onsets = GLM_Utils.get_hit_onsets(behaviour_matrix)

    # Get Cr Onset Frames
    vis_2_correct_onsets = GLM_Utils.get_cr_onsets(behaviour_matrix)

    # Create Save Directory
    save_directory = os.path.join(output_directory_root, session, "Stimuli_Onsets")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "vis_1_onsets.npy"), vis_1_onsets)
    np.save(os.path.join(save_directory, "vis_2_onsets.npy"), vis_2_onsets)
    np.save(os.path.join(save_directory, "vis_1_correct_onsets.npy"), vis_1_correct_onsets)
    np.save(os.path.join(save_directory, "vis_2_correct_onsets.npy"), vis_2_correct_onsets)
