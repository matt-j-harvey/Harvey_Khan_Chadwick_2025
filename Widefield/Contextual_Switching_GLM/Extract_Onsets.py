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


def extract_stable_control_onsets(data_directory_root, session, output_directory_root):

    # Create Empty Lists
    visual_context_vis_1_onsets = []
    visual_context_vis_2_onsets = []
    odour_context_vis_1_onsets = []
    odour_context_vis_2_onsets = []
    odour_1_onsets = []
    odour_2_onsets = []

    # Load behaviour matrix
    behaviour_matrix = np.load(os.path.join(data_directory_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_opto_status = trial[22]
        trial_stim_onset = trial[18]
        trial_irrel_onset = trial[20]
        trial_irrel_type = trial[6]
        trial_ignore_irrel = trial[7]

        if trial_opto_status == False:
            if trial_correct == 1:

                # Vis 1 Hits
                if trial_type == 1:
                    if trial_stim_onset != None:
                        visual_context_vis_1_onsets.append(trial_stim_onset)

                # Vis 2 CRs
                elif trial_type == 2:
                    if trial_stim_onset != None:
                        visual_context_vis_2_onsets.append(trial_stim_onset)

                # Odr 1 Hits
                elif trial_type == 3:
                    if trial_stim_onset != None:
                        if trial_ignore_irrel != 0:
                            odour_1_onsets.append(trial_stim_onset)

                # Odr 2 Crs
                elif trial_type == 4:
                    if trial_stim_onset != None:
                        if trial_ignore_irrel != 0:
                            odour_2_onsets.append(trial_stim_onset)

                # Irrel Trials
                if trial_type == 3 or trial_type == 4:
                    if trial_ignore_irrel == 1:

                        # Odour Context Vis 1
                        if trial_irrel_type == 1:

                            if trial_irrel_onset != None:
                                odour_context_vis_1_onsets.append(trial_irrel_onset)

                        # Odour Context Vis 1
                        if trial_irrel_type == 2:
                            if trial_irrel_onset != None:
                                odour_context_vis_2_onsets.append(trial_irrel_onset)

    """
    print(  "visual_context_vis_1", len(visual_context_vis_1_onsets),
            "visual_context_vis_2", len(visual_context_vis_2_onsets),
            "odour_context_vis_1", len(odour_context_vis_1_onsets),
            "odour_context_vis_2", len(odour_context_vis_2_onsets))
    """

    # Create Save Directory
    save_directory = os.path.join(output_directory_root, session, "Stimuli_Onsets")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "visual_context_stable_vis_1_control_onsets.npy"), visual_context_vis_1_onsets)
    np.save(os.path.join(save_directory, "visual_context_stable_vis_2_control_onsets.npy"), visual_context_vis_2_onsets)
    np.save(os.path.join(save_directory, "odour_context_stable_vis_1_control_onsets.npy"), odour_context_vis_1_onsets)
    np.save(os.path.join(save_directory, "odour_context_stable_vis_2_control_onsets.npy"), odour_context_vis_2_onsets)
    np.save(os.path.join(save_directory, "odour_1_control_onsets.npy"), odour_1_onsets)
    np.save(os.path.join(save_directory, "odour_2_control_onsets.npy"), odour_2_onsets)