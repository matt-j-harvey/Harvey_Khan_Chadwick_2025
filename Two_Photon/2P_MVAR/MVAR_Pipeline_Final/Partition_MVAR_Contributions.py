import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize

import MVAR_Utils_2P


def visualise_predictions(full_prediction, partial_prediction, contribution):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(3,1,1)
    axis_2 = figure_1.add_subplot(3,1,2)
    axis_3 = figure_1.add_subplot(3,1,3)

    axis_1.imshow(np.transpose(full_prediction))
    axis_2.imshow(np.transpose(partial_prediction))
    axis_3.imshow(np.transpose(contribution))

    MVAR_Utils_2P.forceAspect(axis_1, aspect=3)
    MVAR_Utils_2P.forceAspect(axis_2, aspect=3)
    MVAR_Utils_2P.forceAspect(axis_3, aspect=3)

    plt.show()



def get_partial_contribution(design_matrix, mvar_parameters_full, mvar_parameters_partial, Nvar, Ntrials, Nt):


    # View Mean Stim Weights


    # Get Full Prediction
    full_prediction = np.matmul(mvar_parameters_full, design_matrix.T)
    full_prediction = np.transpose(full_prediction)
    print("full_prediction", np.shape(full_prediction))
    #plt.imshow(full_prediction)
    #plt.show()

    # Get Partial Prediction
    partial_prediction = np.matmul(mvar_parameters_partial, design_matrix.T)
    partial_prediction = np.transpose(partial_prediction)

    # Get Partial Contribution
    partial_contribution = np.subtract(full_prediction, partial_prediction)
    print("partial_contribution", np.shape(partial_contribution))

    #visualise_predictions(full_prediction, partial_prediction, partial_contribution)

    # Reshape Recurrent Contribution
    total_trials = int(np.sum(Ntrials))
    print("total_trials", total_trials)
    partial_contribution = np.reshape(partial_contribution, (total_trials, Nt, Nvar))
    print("partial_contribution", np.shape(partial_contribution))

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

    """
    plt.title("stimuli_zeroed_parameters")
    plt.imshow(np.transpose(stimuli_zeroed_parameters))
    MVAR_Utils_2P.forceAspect(plt.gca(), aspect=1)
    plt.show()
    """

    # Isolate Stimuli Contributions
    stimuli_zeroed_parameters = np.copy(mvar_parameters)
    stimuli_zeroed_parameters[:, Nvar:Nvar + (Nstim * Nt)] = 0
    vis_1_stim_contribution, vis_2_stim_contribution = get_partial_contribution(design_matrix, mvar_parameters, stimuli_zeroed_parameters, Nvar, Ntrials, Nt)

    # Isolate Behavioural Contribution
    behaviour_zeroed_parameters = np.copy(mvar_parameters)
    behaviour_zeroed_parameters[:, -Nbehav:] = 0
    vis_1_behaviour_contribution, vis_2_behaviour_contribution = get_partial_contribution(design_matrix, mvar_parameters, behaviour_zeroed_parameters, Nvar, Ntrials, Nt)

    # Isolate Diagonal Contribution
    diagonal_zeroed_parameters = np.copy(mvar_parameters)
    np.fill_diagonal(a=diagonal_zeroed_parameters[:, 0:Nvar], val=np.zeros(Nvar))
    vis_1_diagonal_contribution, vis_2_diagonal_contribution = get_partial_contribution(design_matrix, mvar_parameters, diagonal_zeroed_parameters, Nvar, Ntrials, Nt)

    """
    plt.title("diagonal_zeroed_parameters")
    plt.imshow(np.transpose(diagonal_zeroed_parameters), cmap="bwr", vmin=-0.01, vmax=0.01)
    MVAR_Utils_2P.forceAspect(plt.gca(), aspect=1)
    plt.show()
    """

    # Isolate Recurrent Contribution
    recurrent_zeroed_parameters = np.copy(mvar_parameters)
    recurrent_zeroed_parameters[:, 0:Nvar] = 0 # Zero all connectivity
    np.fill_diagonal(a=recurrent_zeroed_parameters[:, 0:Nvar], val=np.diag(mvar_parameters[0:Nvar, 0:Nvar])) # Add Diagonals Back In
    vis_1_recurrent_contribution, vis_2_recurrent_contribution = get_partial_contribution(design_matrix, mvar_parameters, recurrent_zeroed_parameters, Nvar, Ntrials, Nt)

    """
    plt.title("recurrent_zeroed_parameters")
    plt.imshow(np.transpose(recurrent_zeroed_parameters))
    MVAR_Utils_2P.forceAspect(plt.gca(), aspect=1)
    plt.show()
    """

    # Save These
    np.save(os.path.join(save_directory, "vis_1_stim_contribution.npy"),  vis_1_stim_contribution)
    np.save(os.path.join(save_directory, "vis_2_stim_contribution.npy"),  vis_2_stim_contribution)

    np.save(os.path.join(save_directory, "vis_1_behaviour_contribution.npy"),  vis_1_behaviour_contribution)
    np.save(os.path.join(save_directory, "vis_2_behaviour_contribution.npy"),  vis_2_behaviour_contribution)

    np.save(os.path.join(save_directory, "vis_1_diagonal_contribution.npy"),  vis_1_diagonal_contribution)
    np.save(os.path.join(save_directory, "vis_2_diagonal_contribution.npy"),  vis_2_diagonal_contribution)

    np.save(os.path.join(save_directory, "vis_1_recurrent_contribution.npy"),  vis_1_recurrent_contribution)
    np.save(os.path.join(save_directory, "vis_2_recurrent_contribution.npy"),  vis_2_recurrent_contribution)


