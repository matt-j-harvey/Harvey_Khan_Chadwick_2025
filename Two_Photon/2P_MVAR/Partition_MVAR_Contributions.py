import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize

import MVAR_Utils_2P


def get_partial_contribution(design_matrix, mvar_parameters_full, mvar_parameters_partial, Nvar, Ntrials, Nt):

    # Get Full Prediction
    full_prediction = np.matmul(mvar_parameters_full, design_matrix.T)

    # Get Partial Prediction
    partial_prediction = np.matmul(mvar_parameters_partial, design_matrix.T)

    # Get Partial Contribution
    partial_contribution = np.subtract(full_prediction, partial_prediction)
    np.transpose(partial_contribution)

    # Reshape Recurrent Contribution
    total_trials = int(np.sum(Ntrials))
    print("total_trials", total_trials)
    partial_contribution = np.reshape(partial_contribution, (total_trials, Nt, Nvar))

    # Split By Stimuli
    vis_1_partial_contribution = partial_contribution[0:Ntrials[0]]
    vis_2_partial_contribution = partial_contribution[Ntrials[0]:]

    # Get Mean Contribution
    mean_vis_1_partial_contribution = np.mean(vis_1_partial_contribution, axis=0)
    mean_vis_2_partial_contribution = np.mean(vis_2_partial_contribution, axis=0)

    return mean_vis_1_partial_contribution, mean_vis_2_partial_contribution




def partition_model(mvar_directory_root, session, context):

    # Create Save Directory
    save_directory = os.path.join(mvar_directory_root, session, "Partitioned_Contribution", context)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Regression Matricies
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, mvar_directory_root, context)
    print("design_matrix", np.shape(design_matrix))
    print("delta_f_matrix", np.shape(delta_f_matrix))
    print("Nvar", Nvar)
    print("Nbehav", Nbehav)
    print("Nt", Nt)
    print("Nstim", Nstim)

    # Load MVAR Parameters - The Structure is:     DesignMatrix = np.concatenate((stimblocks.T, dFtot_negshift.T, behaviourtot.T), axis=1)  # design matrix
    model_dict = np.load(os.path.join(mvar_directory_root, session, "Full_Model", context + "_Model_Dict.npy"), allow_pickle=True)[()]
    mvar_parameters = model_dict["MVAR_Parameters"] # Structure (N_Neurons, N_Regressors)
    print("mvar_parameters", np.shape(mvar_parameters))

    # Isolate Stimuli Contributions
    stimuli_zeroed_parameters = np.copy(mvar_parameters)
    stimuli_zeroed_parameters[:, 0:(Nstim * Nt)] = 0
    vis_1_stim_contribution, vis_2_stim_contribution = get_partial_contribution(design_matrix, mvar_parameters, stimuli_zeroed_parameters, Nvar, Ntrials, Nt)

    # Isolate Behavioural Contribution
    behaviour_zeroed_parameters = np.copy(mvar_parameters)
    behaviour_zeroed_parameters[:, -Nbehav:] = 0
    vis_1_behaviour_contribution, vis_2_behaviour_contribution = get_partial_contribution(design_matrix, mvar_parameters, behaviour_zeroed_parameters, Nvar, Ntrials, Nt)

    # Isolate Diagonal Contribution
    diagonal_zeroed_parameters = np.copy(mvar_parameters)
    np.fill_diagonal(a=diagonal_zeroed_parameters[:, (Nstim * Nt):-Nbehav], val=np.zeros(Nvar))
    vis_1_diagonal_contribution, vis_2_diagonal_contribution = get_partial_contribution(design_matrix, mvar_parameters, diagonal_zeroed_parameters, Nvar, Ntrials, Nt)

    # Isolate Recurrent Contribution
    recurrent_zeroed_parameters = np.copy(mvar_parameters)
    recurrent_zeroed_parameters[:, (Nstim * Nt):-Nbehav] = 0 # Zero all connectivity
    np.fill_diagonal(a=recurrent_zeroed_parameters[:, (Nstim * Nt):-Nbehav], val=np.diag(mvar_parameters[:, (Nstim * Nt):-Nbehav])) # Add Diagonals Back In
    vis_1_recurrent_contribution, vis_2_recurrent_contribution = get_partial_contribution(design_matrix, mvar_parameters, recurrent_zeroed_parameters, Nvar, Ntrials, Nt)

    # Save These
    np.save(os.path.join(save_directory, "vis_1_stim_contribution.npy"),  vis_1_stim_contribution)
    np.save(os.path.join(save_directory, "vis_2_stim_contribution.npy"),  vis_2_stim_contribution)

    np.save(os.path.join(save_directory, "vis_1_behaviour_contribution.npy"),  vis_1_behaviour_contribution)
    np.save(os.path.join(save_directory, "vis_2_behaviour_contribution.npy"),  vis_2_behaviour_contribution)

    np.save(os.path.join(save_directory, "vis_1_diagonal_contribution.npy"),  vis_1_diagonal_contribution)
    np.save(os.path.join(save_directory, "vis_2_diagonal_contribution.npy"),  vis_2_diagonal_contribution)

    np.save(os.path.join(save_directory, "vis_1_recurrent_contribution.npy"),  vis_1_recurrent_contribution)
    np.save(os.path.join(save_directory, "vis_2_recurrent_contribution.npy"),  vis_2_recurrent_contribution)


