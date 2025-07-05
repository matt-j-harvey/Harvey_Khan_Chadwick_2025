import os
import numpy as np
import matplotlib.pyplot as plt



def norm_vector(vector):
    norm = np.linalg.norm(vector)
    vector = np.divide(vector, norm)
    return vector


def get_stimuli_weights(model_dict, output_directory):

    model_params = model_dict['MVAR_Parameters']
    n_neurons = model_dict['Nvar']
    Nt = model_dict['Nt']
    preceeding_window = int(Nt/2)

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

    # Get Mean Stimuli Response
    response_window_size = 4
    visual_context_vis_1 = np.mean(visual_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    visual_context_vis_2 = np.mean(visual_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_1 = np.mean(odour_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_2 = np.mean(odour_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)

    # Normalise
    visual_context_vis_1 = norm_vector(visual_context_vis_1)
    visual_context_vis_2 = norm_vector(visual_context_vis_2)
    odour_context_vis_1 = norm_vector(odour_context_vis_1)
    odour_context_vis_2 = norm_vector(odour_context_vis_2)

    # Save These
    save_directory = os.path.join(output_directory, "Stimuli Vectors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "visual_context_vis_1.npy"), visual_context_vis_1)
    np.save(os.path.join(save_directory, "visual_context_vis_2.npy"), visual_context_vis_2)
    np.save(os.path.join(save_directory, "odour_context_vis_1.npy"), odour_context_vis_1)
    np.save(os.path.join(save_directory, "odour_context_vis_2.npy"), odour_context_vis_2)