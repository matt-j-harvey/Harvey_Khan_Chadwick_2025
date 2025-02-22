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


"""
from Widefield_Utils import widefield_utils


import Ridge_Regression_Model_General
import Create_Full_Model_Design_Matrix
import Visualise_Regression_Results
import View_Group_Average_Results
import Test_Result_Significance

sys.path.append(r"/home/matthew/Documents/Github_Code_Clean/Thesis_Code/Utils")
import Create_Data_Tensors
"""



def create_delta_f_matrix(tensor_directory, session, onset_file_list):

    delta_f_matrix = []
    for condition in onset_file_list:

        # Get Tensor Name
        tensor_name = condition.replace("_onsets.npy", "")
        tensor_name = tensor_name.replace("_onset_frames.npy", "")

        # Open Trial Tensor
        session_trial_tensor_dict_path = os.path.join(tensor_directory, session, tensor_name)
        with open(session_trial_tensor_dict_path + ".pickle", 'rb') as handle:
            session_trial_tensor_dict = pickle.load(handle)
            activity_tensor = session_trial_tensor_dict["activity_tensor"]
            activity_tensor = widefield_utils.flatten_tensor(activity_tensor)

        # Add To List
        delta_f_matrix.append(activity_tensor)

    delta_f_matrix = np.vstack(delta_f_matrix)

    return delta_f_matrix



def run_full_model_pipeline(data_directory_root, session, glm_output_diretory_root):

    context_list = ["visual", "odour"]

    # Create Behaviour Matrix
    Create_Behaviour_Matrix.create_behaviour_matrix(data_root_directory, session, mvar_output_directory)

    # Create Activity Tensors
    GLM_Utils.create_activity_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
    GLM_Utils.create_activity_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
    GLM_Utils.create_activity_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
    GLM_Utils.create_activity_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)

    # Create Behaviour Tensors
    GLM_Utils.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
    GLM_Utils.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
    GLM_Utils.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
    GLM_Utils.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)

    # Create Regression Matricies
    for context in context_list:

        design_matrix, delta_f_matrix = Create_Regression_Matricies.create_regression_matricies(session, mvar_directory_root, context)

        # Run Regression
        Ridge_Regression_Model_General.fit_ridge_model(delta_f_matrix, design_matrix, os.path.join(tensor_save_directory, base_directory))

    """
    # Visualise Results
    for mouse in tqdm(selected_session_list, leave=True, position=0, desc="Mouse"):
      for base_directory in tqdm(mouse, leave=True, position=1, desc="Session"):

          data_directory = os.path.join(data_root_diretory, base_directory)
          output_directory = os.path.join(tensor_save_directory, base_directory)
          print("data_directory", data_directory)
          regression_dictionary = np.load(os.path.join(output_directory, "Regression_Dictionary_Simple.npy"), allow_pickle=True)[()]
          design_matrix = np.load(os.path.join(output_directory, "Full_Model_Design_Matrix.npy"))
          design_matrix_key_dict = np.load(os.path.join(output_directory, "design_matrix_key_dict.npy"), allow_pickle=True)[()]
          delta_f_matrix = np.load(os.path.join(output_directory, "Full_Model_Delta_F_Matrix.npy"))

          Visualise_Regression_Results.visualise_regression_results(data_directory, output_directory, regression_dictionary, design_matrix, design_matrix_key_dict, delta_f_matrix, svd_or_nmf=svd_or_nmf)
    
    # View Group Average Coefs
    View_Group_Average_Results.view_average_coef_group(selected_session_list, tensor_save_directory)

    condition_1_coef_name = "visual_context_stable_vis_2"
    condition_2_coef_name = "odour_context_stable_vis_2"
    Test_Result_Significance.test_signifiance(selected_session_list, tensor_save_directory, condition_1_coef_name, condition_2_coef_name, "Vis_2_Context", 0.05)
    Test_Result_Significance.test_signifiance(selected_session_list, tensor_save_directory, condition_1_coef_name, condition_2_coef_name, "Vis_2_Context", 0.1)
 

    condition_1_coef_name = "visual_context_stable_vis_1"
    condition_2_coef_name = "odour_context_stable_vis_1"
    Test_Result_Significance.test_signifiance(selected_session_list, tensor_save_directory, condition_1_coef_name, condition_2_coef_name, "Vis_1_Context", 0.05)
    Test_Result_Significance.test_signifiance(selected_session_list, tensor_save_directory, condition_1_coef_name, condition_2_coef_name, "Vis_1_Context", 0.1)
   """



# Select Analysis Details
# For 2.8 Seconds Pre
# To 2 Seconds Post

frame_period = 36
start_window_ms = -2800
stop_window_ms = 2000
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

# Control Switching
session_list = Session_List.nested_session_list
data_root_diretory = r"/media/matthew/Expansion/Control_Data"
tensor_directory = r"/media/matthew/External_Harddrive_3/Thesis_Analysis/Switching_Analysis/Control_GLM"
run_full_model_pipeline(selected_session_list, analysis_name, data_root_diretory, tensor_directory, svd_or_nmf='nmf')

