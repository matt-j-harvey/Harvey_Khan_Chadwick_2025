import os
import numpy as np

from Matrix_Analysis_Functions.Matrix_Analysis_Functions import norm_vector

def load_recurrent_weights(mvar_directory, session):

    # Load Model Dictionary
    model_dict = np.load(os.path.join(mvar_directory, session, "Full_Model", "Standard_Model_Dict.npy"), allow_pickle=True)[()]

    # Unpack Dict
    model_params = model_dict['MVAR_Parameters']
    Nt = model_dict['Nt']
    n_neurons = model_dict['Nvar']

    # Load Recurrent Weights
    recurrent_weights = model_params[:, 0:n_neurons]

    return recurrent_weights



def get_stimuli_weights(mvar_directory, session):

    # Load Model Dictionary
    model_dict = np.load(os.path.join(mvar_directory, session, "Full_Model", "Standard_Model_Dict.npy"), allow_pickle=True)[()]

    model_params = model_dict['MVAR_Parameters']
    n_neurons = model_dict['Nvar']
    Nt = model_dict['Nt']
    preceeding_window = int(Nt/2)

    # Load Stimuli Weights
    stimulus_weight_list = []
    for stimulus_index in range(2):
        stimulus_weight_start = n_neurons + (stimulus_index * Nt)
        stimulus_weight_stop = stimulus_weight_start + Nt
        stimulus_weight_list.append(model_params[:, stimulus_weight_start:stimulus_weight_stop])

    # Extract Vis 1 and 2 Stimuli
    visual_context_vis_1 = stimulus_weight_list[0]
    visual_context_vis_2 = stimulus_weight_list[1]


    # Get Mean Stimuli Response
    response_window_size = 6
    visual_context_vis_1_mean = np.mean(visual_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    visual_context_vis_2_mean = np.mean(visual_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)

    # Normalise
    visual_context_vis_1_mean = norm_vector(visual_context_vis_1_mean)
    visual_context_vis_2_mean = norm_vector(visual_context_vis_2_mean)

    return visual_context_vis_1_mean, visual_context_vis_2_mean, visual_context_vis_1, visual_context_vis_2

