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


def extract_opto_mapping_onsets(data_directory_root, session, output_directory_root):

    # Create Empty Lists
    visual_context_control_onsets = []
    odour_context_control_onsets = []
    visual_context_light_onsets = []
    odour_context_light_onsets = []

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

        if trial_correct == 1:

            # Visual Context Trials
            if trial_type == 1 or trial_type == 2:
                if trial_stim_onset != None:

                    if trial_opto_status == False:
                        visual_context_control_onsets.append(trial_stim_onset)
                    else:
                        visual_context_light_onsets.append(trial_stim_onset)

            # Odour Context Control Trials
            elif trial_type == 3 or trial_type == 4:
                if trial_irrel_onset != None:
                    if trial_irrel_type == 1 or trial_irrel_type == 2:
                        if trial_ignore_irrel == 1:

                            if trial_opto_status == False:
                                odour_context_control_onsets.append(trial_irrel_onset)
                            else:
                                odour_context_light_onsets.append(trial_irrel_onset)

    # Create Save Directory
    save_directory = os.path.join(output_directory_root, session, "Stimuli_Onsets")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "visual_context_control_onsets.npy"), visual_context_control_onsets)
    np.save(os.path.join(save_directory, "odour_context_control_onsets.npy"), odour_context_control_onsets)
    np.save(os.path.join(save_directory, "visual_context_light_onsets.npy"), visual_context_light_onsets)
    np.save(os.path.join(save_directory, "odour_context_light_onsets.npy"), odour_context_light_onsets)


