import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig

import Session_Lists
import Matrix_Analysis_Functions
import Plotting_Functions


def get_cosine_simmilarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim

def norm_vector(vector):
    norm = np.linalg.norm(vector)
    vector = np.divide(vector, norm)
    return vector


def matrix_eigendecomposition(matrix):

    matrix = np.asarray(matrix, dtype=np.complex128)

    eigenvalues, left_mat, right_mat = eig(matrix, left=True, right=True)

    sort_idx = np.argsort(np.real(eigenvalues))[::-1]

    eigenvalues = eigenvalues[sort_idx]
    left_vecs = left_mat[:, sort_idx].T
    right_vecs = right_mat[:, sort_idx].T

    return eigenvalues, left_vecs, right_vecs




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
    visual_context_vis_1 = np.mean(visual_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    visual_context_vis_2 = np.mean(visual_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)

    # Normalise
    visual_context_vis_1 = norm_vector(visual_context_vis_1)
    visual_context_vis_2 = norm_vector(visual_context_vis_2)

    return visual_context_vis_1, visual_context_vis_2



def get_normalised_commutator_norm(matrix):
    commutator = np.subtract(np.matmul(matrix, matrix.T), np.matmul(matrix.T, matrix))
    commutator_norm = np.linalg.norm(commutator, ord='fro')

    matrix_norm = np.linalg.norm(matrix, ord='fro')
    squared_norm = matrix_norm ** 2

    normalised_commutator_norm = np.divide(commutator_norm, squared_norm)
    return normalised_commutator_norm




def henrici_non_normality(matrix):

    # Get Matrix Dimension
    n = matrix.shape[0]

    # Get Eigenvalues
    eigvals = np.linalg.eigvals(matrix)

    # Get Sum of Sqaured absolute eigenvalues
    eig_sq_sum = np.sum(np.abs(eigvals) ** 2)

    # Get Sqaured Frobenium Norm
    fro_sq = np.linalg.norm(matrix, ord="fro") ** 2

    # Get The Difference
    diff = np.real(fro_sq - eig_sq_sum)

    # Take Sqrt
    sqrt_diff =  np.sqrt(diff)

    # Normalise By N
    normalised_sqrt_diff = sqrt_diff / n

    return normalised_sqrt_diff



def get_vector_alignment(vector_list, comparison_vector):

    vector_simmilarities = []
    for vector in vector_list:
        vector = vector.real
        simmilarity = get_cosine_simmilarity(vector, comparison_vector)
        vector_simmilarities.append(simmilarity)

    return vector_simmilarities


def analyse_recurrent_weight_matrix(data_root, mvar_directory, session, output_root):

    # Create Save Directory
    save_directory = os.path.join(output_root, session)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Recurrent Weight Matrix
    recurrent_matrix = load_recurrent_weights(mvar_directory, session)


    # Load Stimulus Weights
    vis_1_weights, vis_2_weights = get_stimuli_weights(mvar_directory, session)

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_directory, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)

    # View Lick CD Decay
    decay_trajectory, decay_total_norm = Matrix_Analysis_Functions.analyse_lick_decay(lick_cd, recurrent_matrix)
    np.save(os.path.join(save_directory, "Lick_CD_Decay.npy"), decay_trajectory)
    np.save(os.path.join(save_directory, "decay_total_norm.npy"), decay_total_norm)

    # Perform Eigendecomposition
    eigenvalues, left_vecs, right_vecs = matrix_eigendecomposition(recurrent_matrix)
    print("max eigenvalue", np.max(eigenvalues))


    # Save Sorted Eigenvalues
    np.save(os.path.join(save_directory, "Sorted_Eigenvalues.npy"), eigenvalues)
    np.save(os.path.join(save_directory, "Sorted_Left_Eigenvectors.npy"), left_vecs)
    np.save(os.path.join(save_directory, "Sorted_Right_Eigenvectors.npy"), right_vecs)

    # Analayse Right Eigenvector Lick Alignment
    right_vector_cosine_simmilarities = get_vector_alignment(right_vecs, lick_cd)
    np.save(os.path.join(save_directory, "Right_Eigenvectors_Lick_Alignment.npy"), right_vector_cosine_simmilarities)

   # Analayse Left Eigenvector Stimuli Alignment
    left_vector_vis_1_simmilarities = get_vector_alignment(left_vecs, vis_1_weights)
    left_vector_vis_2_simmilarities = get_vector_alignment(left_vecs, vis_2_weights)
    np.save(os.path.join(save_directory, "Left_Eigenvectors_Vis_1_Alignment.npy"), left_vector_vis_1_simmilarities)
    np.save(os.path.join(save_directory, "Left_Eigenvectors_Vis_2_Alignment.npy"), left_vector_vis_2_simmilarities)

    # Measure Non-Normality
    non_normality = henrici_non_normality(recurrent_matrix)
    np.save(os.path.join(save_directory, "non_normality.npy"), non_normality)


    # Get Observability and Controlability Gramians
    observability_gramian = Matrix_Analysis_Functions.get_observability_gramian(recurrent_matrix)
    controlability_gramian = Matrix_Analysis_Functions.get_controllability_gramian(recurrent_matrix)

    np.save(os.path.join(save_directory, "observability_gramian.npy"), observability_gramian)
    np.save(os.path.join(save_directory, "controlability_gramian.npy"), controlability_gramian)

    # Perform Eigendecomposition of observability Gramian
    observability_eigenvalues, observability_eigenvectors, _ = matrix_eigendecomposition(observability_gramian)
    np.save(os.path.join(save_directory, "observability_eigenvalues.npy"), observability_eigenvalues)
    np.save(os.path.join(save_directory, "observability_eigenvectors.npy"), observability_eigenvectors)

    # Perform Eigendecomposition of controlability Gramian
    controlability_eigenvalues, controlability_eigenvectors, _ = matrix_eigendecomposition(controlability_gramian)
    np.save(os.path.join(save_directory, "controlability_eigenvalues.npy"), controlability_eigenvalues)
    np.save(os.path.join(save_directory, "controlability_eigenvectors.npy"), controlability_eigenvectors)

    # Check Stimulus Alignment With Observability Eigenvectors
    vis_1_observability_simmilarities = get_vector_alignment(observability_eigenvectors, vis_1_weights)
    vis_2_observability_simmilarities = get_vector_alignment(observability_eigenvectors, vis_2_weights)
    np.save(os.path.join(save_directory, "Observability_Vis_1_Alignment.npy"), vis_1_observability_simmilarities)
    np.save(os.path.join(save_directory, "Observability_Vis_2_Alignment.npy"), vis_2_observability_simmilarities)


    # Check Lick Alignment With Controlability Eigenvectors
    lick_controlability_simmilarities = get_vector_alignment(controlability_eigenvectors, lick_cd)
    np.save(os.path.join(save_directory, "Controlability_lick_Alignment.npy"), lick_controlability_simmilarities)

    # Calculate lick Reachability
    lick_reachability = Matrix_Analysis_Functions.compute_lick_direction_reachability(lick_cd, controlability_gramian)
    np.save(os.path.join(save_directory, "Lick_Reachability.npy"), lick_reachability)


def analyse_group(data_root, mvar_root, session_list, output_root, group_name):

    for mouse in session_list:
        for session in mouse:
            # Analyse Weight Matrix
            analyse_recurrent_weight_matrix(data_root, mvar_root, session, output_root)

    """
    # Plot Eigenspectrums
    Plotting_Functions.plot_eigenspectrums(session_list, output_root, group_name)

    Plotting_Functions.plot_right_alignment(session_list, output_root, group_name)

    Plotting_Functions.plot_left_alignment(session_list, output_root, group_name)

    Plotting_Functions.plot_observability_eigenspectrums(session_list, output_root, group_name)
    

    Plotting_Functions.plot_left_alignment_observability(session_list, output_root, group_name)

    Plotting_Functions.plot_controlability_lick_alignment(session_list, output_root, group_name)
    """



# Model Info
start_window = -10 # How many timepoints before the onset of each stimulus to include
stop_window = 10 # How many timepoints after the onset of each stimulus to include

# Data root Directory
wt_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\ALM 2P\Data\Controls"
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\ALM 2P\Data\Homs"

# MVAR Root Directory
wt_mvar_root = r"C:\Learning_MVAR\WT"
hom_mvar_root = r"C:\Learning_MVAR\Hom"

# Output Directories
wt_output_root = r"C:\Recurrent_Matrix_Analysis\Wt"
hom_output_root = r"C:\Recurrent_Matrix_Analysis\Hom"


# Session Lists
wt_session_list = Session_Lists.wt_session_list
hom_session_list = Session_Lists.hom_session_list



#analyse_group(wt_data_root, wt_mvar_root, wt_session_list, wt_output_root, "wildtype")
#analyse_group(hom_data_root, hom_mvar_root, hom_session_list, hom_output_root, "neurexin")
"""
Plotting_Functions.compare_lick_reachability_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
Plotting_Functions.compare_lick_cd_decay_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
Plotting_Functions.compare_lick_cd_decay_total_norm_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
"""
#Plotting_Functions.compare_eigenspectrum_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
#Plotting_Functions.compare_right_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
#Plotting_Functions.compare_left_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
#Plotting_Functions.compare_non_normality(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

Plotting_Functions.compare_observability_stim_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

#Plotting_Functions.compare_controlability_lick_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

