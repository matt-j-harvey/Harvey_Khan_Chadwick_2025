import matplotlib.pyplot as plt
import numpy as np
import tables
import os

import View_Opto_Aligned_Averages



def split_opto_by_stimuli(base_directory, roi_trials, roi_index, behaviour_matrix):

    # Classify Trials
    visual_block_vis_1_opto_onsets = []
    visual_block_vis_2_opto_onsets = []
    odour_block_vis_1_opto_onsets = []
    odour_block_vis_2_opto_onsets = []

    for trial_index in roi_trials:
        trial_data = behaviour_matrix[trial_index]
        opto_status = trial_data[22]
        trial_type = trial_data[1]
        irrel_type = trial_data[6]
        trial_onset_frame = trial_data[18]
        irrel_onset_frame = trial_data[20]

        if opto_status == True:

            if trial_type == 1:
                visual_block_vis_1_opto_onsets.append(trial_onset_frame)

            elif trial_type == 2:
                visual_block_vis_2_opto_onsets.append(trial_onset_frame)

            elif trial_type == 3 or trial_type == 4:

                if irrel_type == 1:
                    odour_block_vis_1_opto_onsets.append(irrel_onset_frame)

                elif irrel_type == 2:
                    odour_block_vis_2_opto_onsets.append(irrel_onset_frame)

    # Save Onset Lists
    np.save(os.path.join(base_directory, "Stimuli_Onsets", str(roi_index).zfill(3) + "_Vis_Context_Vis_1_Opto_Onset_Frames.npy"), visual_block_vis_1_opto_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", str(roi_index).zfill(3) + "_Vis_Context_Vis_2_Opto_Onset_Frames.npy"), visual_block_vis_2_opto_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", str(roi_index).zfill(3) + "_Odour_Context_Vis_1_Opto_Onset_Frames.npy"), odour_block_vis_1_opto_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", str(roi_index).zfill(3) + "_Odour_Context_Vis_2_Opto_Onset_Frames.npy"), odour_block_vis_2_opto_onsets)




def get_control_onsets(base_directory, behaviour_matrix):

    visual_context_vis_1_control_onsets = []
    visual_context_vis_2_control_onsets = []
    odour_context_vis_1_control_onsets = []
    odour_context_vis_2_control_onsets = []

    n_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(n_trials):
        trial_data = behaviour_matrix[trial_index]
        opto_status = trial_data[22]
        trial_type = trial_data[1]
        irrel_type = trial_data[6]
        trial_onset_frame = trial_data[18]
        irrel_onset_frame = trial_data[20]

        if opto_status == False:

            if trial_type == 1:
                if trial_onset_frame != None:
                    visual_context_vis_1_control_onsets.append(trial_onset_frame)

            elif trial_type == 2:
                if trial_onset_frame != None:
                    visual_context_vis_2_control_onsets.append(trial_onset_frame)

            elif trial_type == 3 or trial_type == 4:

                if irrel_type == 1:
                    if irrel_onset_frame != None:
                        odour_context_vis_1_control_onsets.append(irrel_onset_frame)

                elif irrel_type == 2:
                    if irrel_onset_frame != None:
                        odour_context_vis_2_control_onsets.append(irrel_onset_frame)


    # Save Onset Lists
    visual_context_vis_1_control_onsets = np.array(visual_context_vis_1_control_onsets)
    visual_context_vis_2_control_onsets = np.array(visual_context_vis_2_control_onsets)
    odour_context_vis_1_control_onsets = np.array(odour_context_vis_1_control_onsets)
    odour_context_vis_2_control_onsets = np.array(odour_context_vis_2_control_onsets)


    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Vis_Context_Vis_1_Control_Onset_Frames.npy"), visual_context_vis_1_control_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Vis_Context_Vis_2_Control_Onset_Frames.npy"), visual_context_vis_2_control_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Odour_Context_Vis_1_Control_Onset_Frames.npy"), odour_context_vis_1_control_onsets)
    np.save(os.path.join(base_directory, "Stimuli_Onsets", "Odour_Context_Vis_2_Control_Onset_Frames.npy"), odour_context_vis_2_control_onsets)




def classify_opto_trials(base_directory):

    # Get Stim Labels
    #stimuli_labels_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Opto_Pattern_Labels.npy"))
    #number_of_unique_stimuli = len(np.unique(stimuli_labels_list))

    # Load Beahviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
    n_trials = np.shape(behaviour_matrix)[0]

    # Check Stim Numbers
    number_of_opto_stims = np.sum(behaviour_matrix[:, 22])
    print("Number of opto stims", number_of_opto_stims)

    stimuli_labels_list = np.zeros(number_of_opto_stims, dtype=int)
    print("stimuli_labels_list", stimuli_labels_list)

    number_of_unique_stimuli = len(np.unique(stimuli_labels_list))
    print("number_of_unique_stimuli", number_of_unique_stimuli)


    # Get Trial Indexes For Each Stim
    roi_trials_list = []
    for stim_index in range(number_of_unique_stimuli):
        roi_trials_list.append([])

    current_opto_index = 0
    for trial_index in range(n_trials):
        trial_data = behaviour_matrix[trial_index]
        opto_status = trial_data[22]

        if opto_status == 1:
            trial_roi = stimuli_labels_list[current_opto_index]
            roi_trials_list[trial_roi].append(trial_index)
            current_opto_index += 1

    # Get Control Onsets
    get_control_onsets(base_directory, behaviour_matrix)

    # Classify Onsets
    for stim_index in range(number_of_unique_stimuli):
        split_opto_by_stimuli(base_directory, roi_trials_list[stim_index], stim_index, behaviour_matrix)
        #View_Opto_Aligned_Averages.view_opto_aligned_averages(base_directory, stim_index)


