import numpy as np
import os
import matplotlib.pyplot as plt




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

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Weight_Matricies")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save These
    np.save(os.path.join(save_directory, "recurrent_weights.npy"), recurrent_weights)
    np.save(os.path.join(save_directory, "diagonal_weights.npy"), diagonal_weights)
