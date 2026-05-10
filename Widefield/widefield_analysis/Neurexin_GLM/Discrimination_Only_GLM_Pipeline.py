import os


"""
number_of_threads = 2
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm


import Session_List
import Create_Behavioural_Regressor_Matrix
import GLM_Utils
import Extract_onsets
import Create_Regression_Matricies
import Parameter_Search
import Fit_Full_Model

"""

import View_Group_Average_Results
"""

def run_discrimination_glm_pipeline(data_root, session_list, output_root, start_window, stop_window, regressor_list):


    # Fit Models For Each Session
    for mouse in tqdm(session_list, desc="Mouse"):
        for session in tqdm(mouse, desc="Session"):
            print(session)

            # Create Behaviour Matrix
            Create_Behavioural_Regressor_Matrix.create_behaviour_matrix(data_root, session, output_root)

            # Extract Onsets
            Extract_onsets.extract_onsets(data_root, session, output_root)

            # Create Tensors
            for regressor in regressor_list:
                GLM_Utils.create_activity_tensor(data_root, session, output_root, regressor + "_onsets.npy", start_window, stop_window)
                GLM_Utils.create_behaviour_tensor(data_root, session, output_root, regressor + "_onsets.npy", start_window, stop_window)

            # Create Regression Matricies
            design_matrix, delta_f_matrix = Create_Regression_Matricies.create_regression_matricies(data_root, session, output_root, regressor_list, z_score=False,  baseline_correct=False)
            print("design_matrix", np.shape(design_matrix), "delta_f_matrix", np.shape(delta_f_matrix))

            # Perform Parameter Search
            print("Performing parameter search")
            Parameter_Search.parameter_search(output_root, session, design_matrix, delta_f_matrix, max_mousecam_components=500)

            # Run Regression
            Fit_Full_Model.fit_full_model(session, output_root, design_matrix, delta_f_matrix, max_mousecam_components=500)


    # View Group Average Coefs
    #View_Group_Average_Results.view_group_average_results(data_directory, session_list, glm_output_directory, start_window, stop_window, regressor_list)


frame_period = 37
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
#regressor_list = ["vis_1_correct", "vis_2_correct"]




# Select Analysis Details - Post Learning
#start_window_ms = -2800
#stop_window_ms = 2500
#control_session_list = Session_List.control_all_post_learning
#control_output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"
#hom_session_list = Session_List.neurexin_all_post_learning
#hom_output_root = r"C:\Neurexin_GLM\Post_Learning\Homs"



# Select Analysis Details -- Intermediate Learning
#start_window_ms = -2500
#stop_window_ms = 2500

#control_session_list = Session_List.control_intermediate_learning
#control_output_root = r"C:\Neurexin_GLM\Intermediate_Learning\Controls"

#hom_session_list = Session_List.neurexin_intermediate_learning
#hom_output_root = r"C:\Neurexin_GLM\Intermediate_Learning\Homs"



# Select Analysis Details -- Pre Learning
start_window_ms = -1500
stop_window_ms = 2500
regressor_list = ["vis_1", "vis_2"]

control_session_list = Session_List.control_pre_learning
control_output_root = r"C:\Neurexin_GLM\Pre_Learning\Controls"

hom_session_list = Session_List.neurexin_pre_learning
hom_output_root = r"C:\Neurexin_GLM\Pre_Learning\Homs"

# Run Pipeline
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)
run_discrimination_glm_pipeline(control_data_root, control_session_list, control_output_root, start_window, stop_window, regressor_list)
run_discrimination_glm_pipeline(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window, regressor_list)




