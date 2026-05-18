import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig
from tqdm import tqdm


import Recurrent_Matrix_Analysis_Utils
import Session_Lists
import Matrix_Analysis_Functions
import Get_DF
import Analyse_Effect_Of_Orthgonal_Dimensions_On_Lick_CD
import Plotting_Functions




def analyse_recurrent_weight_matrix(data_root, mvar_directory, session, output_root):

    # Create Save Directory
    save_directory = os.path.join(output_root, session)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Recurrent Weight Matrix
    recurrent_matrix = Recurrent_Matrix_Analysis_Utils.load_recurrent_weights(mvar_directory, session)

    # Load Stimulus Weights
    vis_1_weights, vis_2_weights, vis_1_time, vis_2_time = Recurrent_Matrix_Analysis_Utils.get_stimuli_weights(mvar_directory, session)
    print("session", session)
    print("vis_1_time", np.shape(vis_1_time))
    print("vis_2_time", np.shape(vis_2_time))

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_directory, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)


    # View Lick CD Decay
    decay_trajectory, decay_total_norm = Matrix_Analysis_Functions.analyse_lick_decay(lick_cd, recurrent_matrix)
    np.save(os.path.join(save_directory, "Lick_CD_Decay.npy"), decay_trajectory)
    np.save(os.path.join(save_directory, "decay_total_norm.npy"), decay_total_norm)
    """
    # Perform Eigendecomposition
    eigenvalues, left_vecs, right_vecs = Matrix_Analysis_Functions.matrix_eigendecomposition(recurrent_matrix)

    # Save Sorted Eigenvalues
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
    observability_eigenvalues, observability_eigenvectors, _ = Matrix_Analysis_Functions.matrix_eigendecomposition(observability_gramian)
    np.save(os.path.join(save_directory, "observability_eigenvalues.npy"), observability_eigenvalues)
    np.save(os.path.join(save_directory, "observability_eigenvectors.npy"), observability_eigenvectors)

    # Perform Eigendecomposition of controlability Gramian
    controlability_eigenvalues, controlability_eigenvectors, _ = Matrix_Analysis_Functions.matrix_eigendecomposition(controlability_gramian)
    np.save(os.path.join(save_directory, "controlability_eigenvalues.npy"), controlability_eigenvalues)
    np.save(os.path.join(save_directory, "controlability_eigenvectors.npy"), controlability_eigenvectors)

    # Check Stimulus Alignment With Observability Eigenvectors
    vis_1_observability_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(observability_eigenvectors, vis_1_weights)
    vis_2_observability_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(observability_eigenvectors, vis_2_weights)
    np.save(os.path.join(save_directory, "Observability_Vis_1_Alignment.npy"), vis_1_observability_simmilarities)
    np.save(os.path.join(save_directory, "Observability_Vis_2_Alignment.npy"), vis_2_observability_simmilarities)

    # Check Lick Alignment With Controlability Eigenvectors
    lick_controlability_simmilarities = Matrix_Analysis_Functions.get_vector_alignment(controlability_eigenvectors, lick_cd)
    np.save(os.path.join(save_directory, "Controlability_lick_Alignment.npy"), lick_controlability_simmilarities)

    # Calculate lick Reachability
    lick_reachability = Matrix_Analysis_Functions.compute_lick_direction_reachability(lick_cd, controlability_gramian)
    np.save(os.path.join(save_directory, "Lick_Reachability.npy"), lick_reachability)

  
    # Analyse Effects Of Orthgonal Peturbations
    df_matrix = Get_DF.load_df_matrix(os.path.join(data_root, session))
    coupling_effect, single_timestep_effect = Analyse_Effect_Of_Orthgonal_Dimensions_On_Lick_CD.analyse_effect_of_orthogonal_dimensions_on_lick_cd(recurrent_matrix, df_matrix, lick_cd)
    np.save(os.path.join(save_directory, "coupling_effect.npy"), coupling_effect)
    np.save(os.path.join(save_directory, "single_timestep_effect.npy"), single_timestep_effect)
    """

    vis_1_direct_projection, vis_1_trajectory = Matrix_Analysis_Functions.compute_stimulus_transformation_function(vis_2_time[:, 8], recurrent_matrix, lick_cd)
    #print("vis_1_trajectory", np.shape(vis_1_trajectory))
    np.save(os.path.join(save_directory, "vis_1_direct_projection.npy"), vis_1_direct_projection)
    np.save(os.path.join(save_directory, "vis_1_transfer_trajectory.npy"), vis_1_trajectory)


def analyse_group(data_root, mvar_root, session_list, output_root, group_name):


    for mouse in tqdm(session_list):

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
wt_data_root =  r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
hom_data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"

# MVAR Root Directory
wt_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR_Multi_Session\WT"
hom_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR_Multi_Session\Homs"

# Output Directories
wt_output_root = r"C:\Recurrent_Matrix_Analysis\Switching\Wt"
hom_output_root = r"C:\Recurrent_Matrix_Analysis\Switching\Hom"


wt_session_list = Session_Lists.wt_switching_session_list
hom_session_list = Session_Lists.hom_switching_session_list


analyse_group(wt_data_root, wt_mvar_root, wt_session_list, wt_output_root, "wildtype")
analyse_group(hom_data_root, hom_mvar_root, hom_session_list, hom_output_root, "neurexin")
"""
Plotting_Functions.compare_lick_cd_decay_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
Plotting_Functions.compare_lick_cd_decay_total_norm_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)


Plotting_Functions.compare_lick_reachability_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)



Plotting_Functions.compare_eigenspectrum_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)


Plotting_Functions.compare_right_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)


Plotting_Functions.compare_left_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

Plotting_Functions.compare_non_normality(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

Plotting_Functions.compare_observability_stim_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

#Plotting_Functions.compare_controlability_lick_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
"""

#Plotting_Functions.compare_coupling_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
#Plotting_Functions.compare_mean_coupling_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
#Plotting_Functions.compare_mean_coupling_single_step_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

Plotting_Functions.compare_direct_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
Plotting_Functions.compare_transfer_trajectories(wt_session_list, wt_output_root, hom_session_list, hom_output_root)