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
import Fit_Full_Model_N_Folds
import Partition_MVAR_Contributions
import Visualise_Partitioned_Contributions


"""
import Create_Behaviour_Tensors
import Create_Regression_Matricies
import NMF_MVAR_Ridge_Penalty_CV
import Fit_Full_Model_LocaNMF_N_Folds
import Partition_MVAR_Contributions
"""




def mvar_pipeline(data_root_directory, session_list, mvar_output_directory, start_window, stop_window, frame_rate):

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
  

        # Create Regression Matricies
        Create_Regression_Matricies.create_regression_matricies(data_root_directory, session, mvar_output_directory, "visual", start_window, stop_window)
        Create_Regression_Matricies.create_regression_matricies(data_root_directory, session, mvar_output_directory, "odour", start_window, stop_window)

        # Perform CV For Each Context
        #MVAR_Ridge_Penalty_CV.get_cv_ridge_penalties(session, mvar_output_directory, "visual")
        #MVAR_Ridge_Penalty_CV.get_cv_ridge_penalties(session, mvar_output_directory, "odour")
     
        # Fit Models
        Fit_Full_Model_N_Folds.fit_full_model(mvar_output_directory, session, "visual")
        Fit_Full_Model_N_Folds.fit_full_model(mvar_output_directory, session, "odour")

        # Partition Contributions
        Partition_MVAR_Contributions.partition_model(mvar_output_directory, session, "visual")
        Partition_MVAR_Contributions.partition_model(mvar_output_directory, session, "odour")
        """

    # View Partitioned Contribution

    #Visualise_Partitioned_Contributions.visualise_component_contribution(data_root_directory, session_list, mvar_output_directory, start_window, stop_window, frame_rate, "stim")
    #Visualise_Partitioned_Contributions.visualise_component_contribution(data_root_directory, session_list, mvar_output_directory, start_window, stop_window, frame_rate, "recurrent")
    #Visualise_Partitioned_Contributions.visualise_component_contribution(data_root_directory, session_list, mvar_output_directory, start_window, stop_window, frame_rate, "diagonal")
    #Visualise_Partitioned_Contributions.visualise_component_contribution(data_root_directory, session_list, mvar_output_directory, start_window, stop_window, frame_rate, "behaviour")


    Visualise_Partitioned_Contributions.check_alignment(data_root_directory, session_list, mvar_output_directory, start_window, stop_window, frame_rate)


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
start_window = -17
stop_window = 12
frame_rate = 6.37

# Control Switching
mvar_pipeline(data_root, control_session_list, mvar_output_root, start_window, stop_window, frame_rate)

