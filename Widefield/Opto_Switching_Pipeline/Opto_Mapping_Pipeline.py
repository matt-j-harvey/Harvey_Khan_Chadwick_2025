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


import Opto_GLM_Utils
import Create_Behavioural_Regressor_Matrix
import Extract_Onsets
import Create_Regression_Matricies
import Parameter_Search
import Fit_Full_Model

"""
import View_Group_Average_Results
"""



def run_glm_pipeline(data_root, experiment, output_root, start_window, stop_window, regressor_list):

    control_session_list = experiment[1]
    opto_session_list = experiment[2]
    session_list = control_session_list + opto_session_list

    # Fit GLM For Each Session
    for session in tqdm(session_list, desc="Mouse"):
        print(session)

        # Create Behaviour Matrix
        Create_Behavioural_Regressor_Matrix.create_behaviour_matrix(data_root, session, output_root)

        # Extract Onsets
        Extract_Onsets.extract_opto_mapping_onsets(data_root, session, output_root)

        # Create Activity Tensors
        Opto_GLM_Utils.create_activity_tensor(data_root, session, output_root, "visual_context_control_onsets.npy", start_window, stop_window)
        Opto_GLM_Utils.create_activity_tensor(data_root, session, output_root, "odour_context_control_onsets.npy", start_window, stop_window)
        Opto_GLM_Utils.create_activity_tensor(data_root, session, output_root, "visual_context_light_onsets.npy", start_window, stop_window)
        Opto_GLM_Utils.create_activity_tensor(data_root, session, output_root, "odour_context_light_onsets.npy", start_window, stop_window)

        # Create Behaviour Tensors
        Opto_GLM_Utils.create_behaviour_tensor(data_root, session, output_root, "visual_context_control_onsets.npy", start_window, stop_window)
        Opto_GLM_Utils.create_behaviour_tensor(data_root, session, output_root, "odour_context_control_onsets.npy", start_window, stop_window)
        Opto_GLM_Utils.create_behaviour_tensor(data_root, session, output_root, "visual_context_light_onsets.npy", start_window, stop_window)
        Opto_GLM_Utils.create_behaviour_tensor(data_root, session, output_root, "odour_context_light_onsets.npy", start_window, stop_window)

        # Create Regression Matricies
        design_matrix, delta_f_matrix = Create_Regression_Matricies.create_regression_matricies(data_root, session, output_root, z_score=False,  baseline_correct=False)
        print("design_matrix", np.shape(design_matrix), "delta_f_matrix", np.shape(delta_f_matrix))

        # Perform Parameter Search
        Parameter_Search.parameter_search(output_root, session, design_matrix, delta_f_matrix, max_mousecam_components=500)

        # Run Regression
        Fit_Full_Model.fit_full_model(session, output_root, design_matrix, delta_f_matrix, max_mousecam_components=500)

    # View Group Average Coefs
    View_Average_Opto_Results.view_group_average_results(data_directory, session_list, glm_output_directory, start_window, stop_window, regressor_list)



"""
For Each Mouse - Fit GLM Estimate Light effect and contextual modulation

    Regressors
    0 = Trial
    1 = Attention
    2 = light
    3 = Attention x Light
"""



# Set Directories
data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Widefield_Opto"
output_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Analysis_Output\Opto_Mapping"

# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 0
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

regressor_list = ["Attention"
                  "Red_Light",
                  "Red_Light_x_Attention"]



experiment_list = [

    ["V1",
    Session_List.v1_opto_session_list,
    Session_List.v1_control_session_list],


    ["PPC",
    Session_List.ppc_opto_session_list,
    Session_List.ppc_control_session_list],

    ["SS",
     Session_List.ss_opto_session_list,
     Session_List.ss_control_session_list],

    ["MM",
     Session_List.mm_opto_session_list,
     Session_List.mm_control_session_list],

    ["ALM",
     Session_List.alm_opto_session_list,
     Session_List.alm_control_session_list],

    ["RSC",
     Session_List.rsc_opto_session_list,
     Session_List.rsc_control_session_list],

    ["PM",
     Session_List.pm_opto_session_list,
     Session_List.pm_control_session_list],

]



# Run Pipeline
for experiment in experiment_list:
    run_glm_pipeline(data_root, experiment, output_root, start_window, stop_window, regressor_list)




