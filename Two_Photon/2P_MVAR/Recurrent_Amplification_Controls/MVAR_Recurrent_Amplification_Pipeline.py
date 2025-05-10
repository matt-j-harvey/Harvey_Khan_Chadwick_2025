import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

from scipy import stats

import Plotting_Functions



def get_integrated_interaction(stimulus_vector, recurrent_weights):

    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_state = np.zeros(n_neurons)
    for x in range(9):
        trial_vector.append(current_state)
        current_state = np.matmul(recurrent_weights, current_state)
        current_state = np.add(current_state, stimulus_vector)

    trial_vector = np.array(trial_vector)
    
    return trial_vector


def shuffle_recurrent_weights(recurrent_weights, diagonal_weights):
    
    # Fill Diagonal With Shuffled Recurrents
    n_neurons = np.shape(recurrent_weights)[0]
    diag_mask = np.eye(n_neurons)
    off_diag_mask = np.subtract(np.ones(np.shape(diag_mask)), diag_mask)
    off_diag_indicies = np.nonzero(off_diag_mask)
    off_diag_weights = recurrent_weights[off_diag_indicies]
    np.random.shuffle(off_diag_weights)
    shuffled_recurrent_weights = np.copy(diagonal_weights)
    shuffled_recurrent_weights[off_diag_indicies] = off_diag_weights
    return shuffled_recurrent_weights


def norm_vector(vector):
    print("vector", vector)
    plt.hist(vector)
    plt.show()

    norm = np.linalg.norm(vector)
    print("vector norm", norm)
    vector = np.divide(vector, norm)
    return vector


def get_stimuli_weights(model_params, n_neurons, Nt, preceeding_window, norm):

    # Load Stimuli Weights
    stimulus_weight_list = []
    for stimulus_index in range(6):
        stimulus_weight_start = n_neurons + (stimulus_index * Nt)
        stimulus_weight_stop = stimulus_weight_start + Nt
        stimulus_weight_list.append(model_params[:, stimulus_weight_start:stimulus_weight_stop])

    # Extract Vis 1 and 2 Stimuli
    visual_context_vis_1 = stimulus_weight_list[0]
    visual_context_vis_2 = stimulus_weight_list[1]
    odour_context_vis_1 = stimulus_weight_list[2]
    odour_context_vis_2 = stimulus_weight_list[3]

    Plotting_Functions.view_psth(np.transpose(visual_context_vis_1))

    # Get Mean Stimuli Response
    response_window_size = 6
    visual_context_vis_1 = np.mean(visual_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    visual_context_vis_2 = np.mean(visual_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_1 = np.mean(odour_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_2 = np.mean(odour_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)

    # Normalise If Required
    if norm == True:
        visual_context_vis_1 = norm_vector(visual_context_vis_1)
        visual_context_vis_2 = norm_vector(visual_context_vis_2)
        odour_context_vis_1 = norm_vector(odour_context_vis_1)
        odour_context_vis_2 = norm_vector(odour_context_vis_2)



    return visual_context_vis_1, visual_context_vis_2, odour_context_vis_1, odour_context_vis_2



def compare_stimulus_recurrent_interaction(session, output_directory, lick_cd, norm):

    # Load Model Dict
    model_dict = np.load(os.path.join(output_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
    model_params = model_dict["MVAR_Parameters"]
    n_neurons = np.shape(model_params)[0]
    Nt = model_dict["Nt"]
    preceeding_window = int(Nt/2)

    # Load Recurrent Weights
    recurrent_weights = model_params[:, 0:n_neurons]

    # Get Diagonal Only Weights
    diagonal_weights = np.zeros(np.shape(recurrent_weights))
    np.fill_diagonal(diagonal_weights, np.diag(recurrent_weights))

    # Get Shuffled Diagonal Only Weights
    shuffled_recurrent_weights = shuffle_recurrent_weights(recurrent_weights, diagonal_weights)

    # Get Stimuli Weights
    visual_context_vis_1, visual_context_vis_2, odour_context_vis_1, odour_context_vis_2 = get_stimuli_weights(model_params, n_neurons, Nt, preceeding_window, norm)
    
    # View Interaction Between Stimulus Vector and Recurrent Weights
    full_vis_context_vis_1_interaction_vector = get_integrated_interaction(visual_context_vis_1, recurrent_weights)
    full_vis_context_vis_2_interaction_vector = get_integrated_interaction(visual_context_vis_2, recurrent_weights)
    full_odr_context_vis_1_interaction_vector = get_integrated_interaction(odour_context_vis_1, recurrent_weights)
    full_odr_context_vis_2_interaction_vector = get_integrated_interaction(odour_context_vis_2, recurrent_weights)
    full_vis_1_projection = np.dot(full_vis_context_vis_1_interaction_vector, lick_cd)
    full_vis_2_projection = np.dot(full_vis_context_vis_2_interaction_vector, lick_cd)
    full_odr_1_projection = np.dot(full_odr_context_vis_1_interaction_vector, lick_cd)
    full_odr_2_projection = np.dot(full_odr_context_vis_2_interaction_vector, lick_cd)

    # Compare To Diagonal Only
    diagonal_vis_context_vis_1_interaction_vector = get_integrated_interaction(visual_context_vis_1, diagonal_weights)
    diagonal_vis_context_vis_2_interaction_vector = get_integrated_interaction(visual_context_vis_2, diagonal_weights)
    diagonal_odr_context_vis_1_interaction_vector = get_integrated_interaction(odour_context_vis_1, diagonal_weights)
    diagonal_odr_context_vis_2_interaction_vector = get_integrated_interaction(odour_context_vis_2, diagonal_weights)
    diagonal_vis_1_projection = np.dot(diagonal_vis_context_vis_1_interaction_vector, lick_cd)
    diagonal_vis_2_projection = np.dot(diagonal_vis_context_vis_2_interaction_vector, lick_cd)
    diagonal_odr_1_projection = np.dot(diagonal_odr_context_vis_1_interaction_vector, lick_cd)
    diagonal_odr_2_projection = np.dot(diagonal_odr_context_vis_2_interaction_vector, lick_cd)

    # Compare To Shuffled Recurrent Weights
    shuffled_vis_context_vis_1_interaction_vector = get_integrated_interaction(visual_context_vis_1, shuffled_recurrent_weights)
    shuffled_vis_context_vis_2_interaction_vector = get_integrated_interaction(visual_context_vis_2, shuffled_recurrent_weights)
    shuffled_odr_context_vis_1_interaction_vector = get_integrated_interaction(odour_context_vis_1, shuffled_recurrent_weights)
    shuffled_odr_context_vis_2_interaction_vector = get_integrated_interaction(odour_context_vis_2, shuffled_recurrent_weights)
    shuffled_vis_1_projection = np.dot(shuffled_vis_context_vis_1_interaction_vector, lick_cd)
    shuffled_vis_2_projection = np.dot(shuffled_vis_context_vis_2_interaction_vector, lick_cd)
    shuffled_odr_1_projection = np.dot(shuffled_odr_context_vis_1_interaction_vector, lick_cd)
    shuffled_odr_2_projection = np.dot(shuffled_odr_context_vis_2_interaction_vector, lick_cd)


    # Return These
    full_projection_list = [full_vis_1_projection,
                            full_vis_2_projection,
                            full_odr_1_projection,
                            full_odr_2_projection]

    diagonal_projection_list = [diagonal_vis_1_projection,
                                diagonal_vis_2_projection,
                                diagonal_odr_1_projection,
                                diagonal_odr_2_projection]
    
    shuffled_projection_list =  [shuffled_vis_1_projection,
                                 shuffled_vis_2_projection,
                                 shuffled_odr_1_projection,
                                 shuffled_odr_2_projection]

    return full_projection_list, diagonal_projection_list, shuffled_projection_list



def recurrent_amplification_piepline(mvar_output_root, session_list, norm=False):

    lick_cd_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"

    # Get Projections
    full_projection_group_list = []
    diagonal_projection_group_list = []
    shuffled_projection_group_list = []

    for session in session_list:

        # Load Lick CDs
        lick_cd = np.load(os.path.join(lick_cd_directory, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

        full_projection_list, diagonal_projection_list, shuffled_projection_list = compare_stimulus_recurrent_interaction(session, mvar_output_root, lick_cd, norm)
        full_projection_group_list.append(full_projection_list)
        diagonal_projection_group_list.append(diagonal_projection_list)
        shuffled_projection_group_list.append(shuffled_projection_list)

    full_projection_group_list = np.array(full_projection_group_list)
    diagonal_projection_group_list = np.array(diagonal_projection_group_list)
    shuffled_projection_group_list = np.array(shuffled_projection_group_list)

    return diagonal_projection_group_list, full_projection_group_list, shuffled_projection_group_list




# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_No_Preceeding_Lick_Regressor"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


# Run Piepline
diagonal_projection_group_list, full_projection_group_list, shuffled_projection_group_list = recurrent_amplification_piepline(mvar_output_root, control_session_list, norm=True)


# Plot Results
Plotting_Functions.plot_scatter_graph_diff(diagonal_projection_group_list, full_projection_group_list)
Plotting_Functions.plot_stimuli_amplification(diagonal_projection_group_list, full_projection_group_list, ylim=[-2,5])

Plotting_Functions.plot_scatter_graph_diff(shuffled_projection_group_list, full_projection_group_list)
Plotting_Functions.plot_stimuli_amplification(shuffled_projection_group_list, full_projection_group_list, ylim=[-2,5])

