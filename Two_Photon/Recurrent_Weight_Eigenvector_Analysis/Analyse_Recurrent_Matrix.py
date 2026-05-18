import os
import numpy as np

from Matrix_Analysis_Functions import (Matrix_Analysis_Functions,
                                       Lick_Coupling,
                                       Analyse_Lick_Decay,
                                       Analyse_Stimulus_Transformation,
                                       Get_Prestim_Ramp_Coupling,
                                       Check_Covariance_Alignment_Lick_CD)

from Shared_Utils import Recurrent_Matrix_Analysis_Utils
from Shared_Utils.Get_DF import load_df_matrix



def analyse_recurrent_weight_matrix(data_root, mvar_directory, session, output_root, pre_learning=False):

    # Create Save Directory
    save_directory = os.path.join(output_root, session)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Recurrent Weight Matrix
    recurrent_matrix = Recurrent_Matrix_Analysis_Utils.load_recurrent_weights(mvar_directory, session)

    # Load Stimulus Weights
    vis_1_weights, vis_2_weights, vis_1_time, vis_2_time = Recurrent_Matrix_Analysis_Utils.get_stimuli_weights(mvar_directory, session)

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_directory, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))

    # Load Df Matrix
    df_matrix = load_df_matrix(os.path.join(data_root, session))

    """
    # View Lick CD Decay
    decay_trajectory, decay_total_norm = Analyse_Lick_Decay.analyse_lick_decay(lick_cd, recurrent_matrix)
    np.save(os.path.join(save_directory, "Lick_CD_Decay.npy"), decay_trajectory)
    np.save(os.path.join(save_directory, "decay_total_norm.npy"), decay_total_norm)

    # Perform Eigendecomposition
    eigenvalues, left_vecs, right_vecs = Matrix_Analysis_Functions.matrix_eigendecomposition(recurrent_matrix)
    np.save(os.path.join(save_directory, "Sorted_Eigenvalues.npy"), eigenvalues)
    np.save(os.path.join(save_directory, "Sorted_Left_Eigenvectors.npy"), left_vecs)
    np.save(os.path.join(save_directory, "Sorted_Right_Eigenvectors.npy"), right_vecs)

    # Analayse Right Eigenvector Lick Alignment
    right_vector_cosine_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(right_vecs, lick_cd)
    np.save(os.path.join(save_directory, "Right_Eigenvectors_Lick_Alignment.npy"), right_vector_cosine_simmilarities)

    # Analayse Left Eigenvector Stimuli Alignment
    left_vector_vis_1_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(left_vecs, vis_1_weights)
    left_vector_vis_2_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(left_vecs, vis_2_weights)
    np.save(os.path.join(save_directory, "Left_Eigenvectors_Vis_1_Alignment.npy"), left_vector_vis_1_simmilarities)
    np.save(os.path.join(save_directory, "Left_Eigenvectors_Vis_2_Alignment.npy"), left_vector_vis_2_simmilarities)

    # Measure Non-Normality
    non_normality = Matrix_Analysis_Functions.henrici_non_normality(recurrent_matrix)
    np.save(os.path.join(save_directory, "non_normality.npy"), non_normality)
    
    # Get Observability and Controlability Gramians
    observability_gramian = Matrix_Analysis_Functions.get_observability_gramian(recurrent_matrix)
    controlability_gramian = Matrix_Analysis_Functions.get_controllability_gramian(recurrent_matrix)
    np.save(os.path.join(save_directory, "observability_gramian.npy"), observability_gramian)
    np.save(os.path.join(save_directory, "controlability_gramian.npy"), controlability_gramian)

    # Perform Eigendecomposition of observability Gramian
    observability_eigenvalues, observability_eigenvectors, _ = Matrix_Analysis_Functions.matrix_eigendecomposition(observability_gramian, absolute=True)
    np.save(os.path.join(save_directory, "observability_eigenvalues.npy"), observability_eigenvalues)
    np.save(os.path.join(save_directory, "observability_eigenvectors.npy"), observability_eigenvectors)

    # Perform Eigendecomposition of controlability Gramian
    controlability_eigenvalues, controlability_eigenvectors, _ = Matrix_Analysis_Functions.matrix_eigendecomposition(controlability_gramian, absolute=True)
    np.save(os.path.join(save_directory, "controlability_eigenvalues.npy"), controlability_eigenvalues)
    np.save(os.path.join(save_directory, "controlability_eigenvectors.npy"), controlability_eigenvectors)

    # Check Stimulus Alignment With Observability Eigenvectors
    vis_1_observability_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(observability_eigenvectors, vis_1_weights, absolute=True)
    vis_2_observability_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(observability_eigenvectors, vis_2_weights, absolute=True)
    np.save(os.path.join(save_directory, "Observability_Vis_1_Alignment.npy"), vis_1_observability_simmilarities)
    np.save(os.path.join(save_directory, "Observability_Vis_2_Alignment.npy"), vis_2_observability_simmilarities)

    # Check Lick Alignment With Controlability Eigenvectors
    lick_controlability_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(controlability_eigenvectors, lick_cd, absolute=True)
    np.save(os.path.join(save_directory, "Controlability_lick_Alignment.npy"), lick_controlability_simmilarities)

    # Calculate lick Reachability
    lick_reachability = Matrix_Analysis_Functions.compute_dimension_reachability(lick_cd, controlability_gramian)
    np.save(os.path.join(save_directory, "Lick_Reachability.npy"), lick_reachability)

    # Analyse Effects Of Orthgonal Peturbations
    coupling_effect, single_timestep_effect = Lick_Coupling.analyse_effect_of_orthogonal_dimensions_on_lick_cd(recurrent_matrix, df_matrix, lick_cd)
    np.save(os.path.join(save_directory, "coupling_effect.npy"), coupling_effect)
    np.save(os.path.join(save_directory, "single_timestep_effect.npy"), single_timestep_effect)

    # Analyse Transformation of Stimuli
    vis_1_direct_projection, vis_1_trajectory, vis_1_cosine_simmilarity = Analyse_Stimulus_Transformation.compute_stimulus_transformation_function(vis_1_weights, recurrent_matrix, lick_cd)
    vis_2_direct_projection, vis_2_trajectory, vis_2_cosine_simmilarity = Analyse_Stimulus_Transformation.compute_stimulus_transformation_function(vis_2_weights, recurrent_matrix, lick_cd)
    np.save(os.path.join(save_directory, "vis_1_direct_projection.npy"), vis_1_direct_projection)
    np.save(os.path.join(save_directory, "vis_2_direct_projection.npy"), vis_2_direct_projection)
    np.save(os.path.join(save_directory, "vis_1_transfer_trajectory.npy"), vis_1_trajectory)
    np.save(os.path.join(save_directory, "vis_2_transfer_trajectory.npy"), vis_2_trajectory)
    np.save(os.path.join(save_directory, "vis_1_cosine_simmilarity.npy"), vis_1_cosine_simmilarity)
    np.save(os.path.join(save_directory, "vis_2_cosine_simmilarity.npy"), vis_2_cosine_simmilarity)
   """

    # Analyse Prestim Activity
    direct_projection, lick_projection_trajectory, orthogonal_magnitude_trajectory, preparatory_cosine_similarity = Get_Prestim_Ramp_Coupling.prestim_coupling(session, data_root, df_matrix, recurrent_matrix, lick_cd, pre_learning)
    np.save(os.path.join(save_directory, "Prestim_Coupling.npy"), lick_projection_trajectory)
    np.save(os.path.join(save_directory, "Prestim_Coupling_Orthogonal.npy"), orthogonal_magnitude_trajectory)
    np.save(os.path.join(save_directory, "direct_prestim_projection.npy"), direct_projection)
    np.save(os.path.join(save_directory, "preparatory_cosine_similarity.npy"), preparatory_cosine_similarity)


    # Get Alignment Of Covariance With Lick CD
    #eigenvector_alignment = Check_Covariance_Alignment_Lick_CD.get_covariance_alignment_lick_cd(df_matrix, lick_cd)
    #np.save(os.path.join(save_directory, "covariance_eigenvector_lick_cd_alignment.npy"), eigenvector_alignment)