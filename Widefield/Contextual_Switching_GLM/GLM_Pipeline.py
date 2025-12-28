import os
number_of_threads = 2
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm

import Session_List
import GLM_Utils
import Create_Behavioural_Regressor_Matrix
import Extract_Onsets
import Create_Regression_Matricies
import Parameter_Search
import Fit_Full_Model
import View_Group_Average_Results


def run_glm_pipeline(data_directory, session_list, glm_output_directory, start_window, stop_window, regressor_list):

    """
    # Fit Models For Each Session
    for mouse in tqdm(session_list, desc="Mouse"):
        for session in tqdm(mouse, desc="Session"):
            print(session)

            # Create Behaviour Matrix
            Create_Behavioural_Regressor_Matrix.create_behaviour_matrix(data_directory, session, glm_output_directory)

            # Extract Onsets
            Extract_Onsets.extract_stable_control_onsets(data_directory, session, glm_output_directory)

            # Create Activity Tensors
            GLM_Utils.create_activity_tensor(data_directory, session, glm_output_directory, "visual_context_stable_vis_1_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_activity_tensor(data_directory, session, glm_output_directory, "visual_context_stable_vis_2_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_activity_tensor(data_directory, session, glm_output_directory, "odour_context_stable_vis_1_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_activity_tensor(data_directory, session, glm_output_directory, "odour_context_stable_vis_2_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_activity_tensor(data_directory, session, glm_output_directory, "odour_1_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_activity_tensor(data_directory, session, glm_output_directory, "odour_2_control_onsets.npy", start_window, stop_window)

            # Create Behaviour Tensors
            GLM_Utils.create_behaviour_tensor(data_directory, session, glm_output_directory, "visual_context_stable_vis_1_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_behaviour_tensor(data_directory, session, glm_output_directory, "visual_context_stable_vis_2_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_behaviour_tensor(data_directory, session, glm_output_directory, "odour_context_stable_vis_1_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_behaviour_tensor(data_directory, session, glm_output_directory, "odour_context_stable_vis_2_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_behaviour_tensor(data_directory, session, glm_output_directory, "odour_1_control_onsets.npy", start_window, stop_window)
            GLM_Utils.create_behaviour_tensor(data_directory, session, glm_output_directory, "odour_2_control_onsets.npy", start_window, stop_window)

            # Create Regression Matricies
            design_matrix, delta_f_matrix = Create_Regression_Matricies.create_regression_matricies(data_directory, session, glm_output_directory, z_score=False,  baseline_correct=False)

            # Perform Parameter Search
            Parameter_Search.parameter_search(glm_output_directory, session, design_matrix, delta_f_matrix, max_mousecam_components=500)

            # Run Regression
            Fit_Full_Model.fit_full_model(session, glm_output_directory, design_matrix, delta_f_matrix, max_mousecam_components=500)
    """

    # View Group Average Coefs
    View_Group_Average_Results.view_group_average_results(data_directory, session_list, glm_output_directory, start_window, stop_window, regressor_list)






# Set Directories
data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Widefield_Opto"
output_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Analysis_Output\Widefield_GLM"

# Select Analysis Details
frame_period = 36
start_window_ms = -1500
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)
regressor_list = ["Vis_1_Rel", "Vis_2_Rel", "Vis_1_Irrel", "Vis_2_Irrel"]

# Load Session List
session_list = Session_List.nested_session_list

# Run Pipeline
run_glm_pipeline(data_root, session_list, output_root, start_window, stop_window, regressor_list)




