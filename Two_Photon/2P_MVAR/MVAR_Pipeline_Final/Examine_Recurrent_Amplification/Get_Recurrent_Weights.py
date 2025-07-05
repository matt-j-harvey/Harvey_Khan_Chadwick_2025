import numpy as np
import os
import matplotlib.pyplot as plt


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



def get_recurrent_weights(model_dict, output_directory):

    # Unpack Dict
    model_params = model_dict['MVAR_Parameters']
    Nt = model_dict['Nt']
    n_neurons = model_dict['Nvar']

    # Load Recurrent Weights
    recurrent_weights = model_params[:, 0:n_neurons]

    # Get Diagonal Only Weights
    diagonal_weights = np.zeros(np.shape(recurrent_weights))
    np.fill_diagonal(diagonal_weights, np.diag(recurrent_weights))

    # Get Shuffled Diagonal Only Weights
    shuffled_recurrent_weights = shuffle_recurrent_weights(recurrent_weights, diagonal_weights)

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Weight_Matricies")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save These
    np.save(os.path.join(save_directory, "recurrent_weights.npy"), recurrent_weights)
    np.save(os.path.join(save_directory, "diagonal_weights.npy"), diagonal_weights)
    np.save(os.path.join(save_directory, "shuffled_recurrent_weights.npy"), shuffled_recurrent_weights)