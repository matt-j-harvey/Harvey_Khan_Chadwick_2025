import os


number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import sys

import MVAR_Utils_2P
import Downsample_AI_Matrix_Framewise_2P
import Create_Behaviour_Matrix
import Create_Regression_Matricies
import MVAR_Ridge_Penalty_CV

"""
import Create_Behaviour_Tensors
import Create_Regression_Matricies
import NMF_MVAR_Ridge_Penalty_CV
import Fit_Full_Model_LocaNMF_N_Folds
import Partition_MVAR_Contributions
"""


def get_best_ridge_penalties(output_directory):
    # Get Selection Of Potential Ridge Penalties
    penalty_possible_values = np.load(os.path.join(output_directory, "Ridge_Penalty_Search", "Penalty_Possible_Values.npy"))

    # Load Visual Penalty Matrix
    penalty_matrix = np.load(os.path.join(output_directory, "Ridge_Penalty_Search", "Local_NMF_Ridge_Penalty_Search_Results.npy"))
    best_score = np.max(penalty_matrix)
    score_indexes = np.where(penalty_matrix == best_score)

    stimuli_penalty_index = score_indexes[0]
    behaviour_penalty_index = score_indexes[1]
    interaction_penalty_index = score_indexes[2]

    stimuli_penalty_value = penalty_possible_values[stimuli_penalty_index][0]
    behaviour_penalty_value = penalty_possible_values[behaviour_penalty_index][0]
    interaction_penalty_value = penalty_possible_values[interaction_penalty_index][0]

    return stimuli_penalty_value, behaviour_penalty_value, interaction_penalty_value


def create_ridge_penalty_dictionary(save_directory_root):
    stimuli_penalty, behaviour_penalty, interaction_penalty = get_best_ridge_penalties(save_directory_root)
    print("stimuli_penalty", stimuli_penalty)
    print("behaviour_penalty", behaviour_penalty)
    print("interaction_penalty", interaction_penalty)

    ridge_penalty_dict = {
        "stimuli_penalty": stimuli_penalty,
        "behaviour_penalty": behaviour_penalty,
        "interaction_penalty": interaction_penalty,
    }

    """

    ridge_penalty_dict = {
        "stimuli_penalty":10,
        "behaviour_penalty":100,
        "interaction_penalty":1000,
    }
    """
    return ridge_penalty_dict



def mvar_pipeline(data_root_directory, session_list, mvar_output_directory, start_window, stop_window):

    """
    //// Running The MVAR Pipeline ////
    1.) Generate Activity Tensors
    3.) Generate Behaviour Tensors
    3.) Create Regression Matricies for MVAR From These Tensors
    4.) Perform Ridge Penalty CV to Find Best Ridge Penalty
    5.) Fit Full Model With These Penalties
    6.) Plot Results of The MVAR
    """


    # General Preprocessing

    for session in tqdm(session_list, position=0, desc="Session:"):

        """
        # Downsample AI Matrix
        Downsample_AI_Matrix_Framewise_2P.downsample_ai_matrix(data_root_directory, session, mvar_output_directory)
      
        # Create Behaviour Matrix
        Create_Behaviour_Matrix.create_behaviour_matrix(data_root_directory, session, mvar_output_directory)

        # Create Activity Tensors
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)

        # Create Behaviour Tensors
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)
        """
        # Create Regression Matricies
        Create_Regression_Matricies.create_regression_matricies(session, mvar_output_directory, "visual", start_window, stop_window)
        Create_Regression_Matricies.create_regression_matricies(session, mvar_output_directory, "odour", start_window, stop_window)

        # Perform CV For Each Context
        MVAR_Ridge_Penalty_CV.get_cv_ridge_penalties(session, mvar_output_directory, "visual")
        MVAR_Ridge_Penalty_CV.get_cv_ridge_penalties(session, mvar_output_directory, "odour")

    """
    # Fit Combined Model


    # Get Ridge Penalty Dict
    ridge_penalty_dict = create_ridge_penalty_dictionary(mvar_directory_root)

    # Fit Model
    Fit_Full_Model_LocaNMF_N_Folds.fit_full_model(mvar_directory_root, session_list, ridge_penalty_dict)

    for mouse in tqdm(session_list, position=0, desc="Mouse"):
        for session in tqdm(mouse, position=1, desc="Session"):
            # Partition Model Contributions
            Partition_MVAR_Contributions.parition_mouse_model(mvar_directory_root, session)

    # View Partitioned Contribution
    # Visualise_Partitioned_Contributions.visualise_model(data_directory_root, session, mvar_directory_root)

    # Get ROI Seed MAps
    Get_ROI_Seed_Maps.get_all_roi_seed_maps(mvar_directory_root, session_list)
    """




# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


# Model Info
start_window = -69
stop_window = 56
frame_period = 36

# Control Switching
mvar_pipeline(data_root, control_session_list, mvar_output_root, start_window, stop_window)

