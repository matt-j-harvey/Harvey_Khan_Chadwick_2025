import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def get_integrated_interaction(stimulus_vector, recurrent_weights, duration=9):
    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_state = np.zeros(n_neurons)
    for x in range(duration):
        trial_vector.append(current_state)
        current_state = np.matmul(recurrent_weights, current_state)
        current_state = np.add(current_state, stimulus_vector)

    trial_vector = np.array(trial_vector)
    return trial_vector


def get_stimuli_recurrent_interactions(mvar_root_directory, session, output_directory, weight_matrix_file):

    stimuli_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
    ]

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Stimuli_Weight_Interactions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_root_directory, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)

    # Load Weight Matrix
    weight_matrix = np.load(os.path.join(output_directory, "Weight_Matricies", weight_matrix_file + ".npy"))

    interaction_list = []
    for stimulus in stimuli_list:

        # Load Vector
        stimulus_vector = np.load(os.path.join(output_directory, "Stimuli Vectors", stimulus + ".npy"))

        # Get Interaction
        stimuli_weight_interaction = get_integrated_interaction(stimulus_vector, weight_matrix)

        # Get Lick CD
        stimuli_weight_interaction_lick_cd = np.dot(stimuli_weight_interaction, lick_cd)

        # Add To List
        interaction_list.append(stimuli_weight_interaction_lick_cd)

    # Save List
    np.save(os.path.join(save_directory, weight_matrix_file + "_Interaction.npy"), interaction_list)


