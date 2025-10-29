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
from sklearn.linear_model import Ridge, LinearRegression

from Widefield_Utils import widefield_utils

import Session_List
import GLM_Utils
import Create_Behavioural_Regressor_Matrix
import Extract_Onsets
import Create_Regression_Matricies

import Visualise_Regression_Results
import Test_Significance_GLM


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)

def sanity_check_coefs(model_coefs, start_window, stop_window):

    n_timepoints = stop_window - start_window
    vis_context_vis_1 = model_coefs[:, 0:n_timepoints]

    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    for timepoint_index in range(n_timepoints):
        image = vis_context_vis_1[:, timepoint_index]
        image = widefield_utils.create_image_from_data(image, indicies, image_height, image_width)
        plt.imshow(image, vmin=-2, vmax=2, cmap="bwr")
        plt.title("Sanity Check Coefs" + str(timepoint_index))
        plt.show()



def run_regression(session, glm_output_directory, design_matrix, delta_f_matrix):

    # Run Regression
    model = Ridge(alpha=2, fit_intercept=False)
    #model = LinearRegression(fit_intercept=False)
    model.fit(X=design_matrix, y=delta_f_matrix)

    # Extract Coefs
    model_coefs = model.coef_
    print("Model Coefs", np.shape(model_coefs))

    #sanity_check_coefs(model_coefs, start_window, stop_window)

    # Save Coefs
    save_directory = os.path.join(glm_output_directory, session, "Model_Output")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Model_Coefs.npy"), model_coefs)






def run_glm_pipeline(data_directory, session_list, glm_output_directory, start_window, stop_window, context_list = ["visual", "odour"]):


    for session in tqdm(session_list):
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

        """
        # Create Regression Matricies
        design_matrix, delta_f_matrix = Create_Regression_Matricies.create_regression_matricies(data_directory, session, glm_output_directory, z_score=True,  baseline_correct=True)

        print("Design Matrix", np.shape(design_matrix))
        print("Delta F Matrix", np.shape(delta_f_matrix))

        """
        #plt.imshow(np.transpose(design_matrix))
        #forceAspect(plt.gca())
        #plt.show()
        """

        # Run Regression
        run_regression(session, glm_output_directory, design_matrix, delta_f_matrix)
        """

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
start_window_ms = -1500 #-2800
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


# Load Session List
nested_session_list = Session_List.nested_session_list
flat_session_list = Session_List.flatten_nested_list(nested_session_list)

data_root_diretory = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"
output_directory = r"/media/matthew/29D46574463D2856/Paper_Results/Contextual_Swtiching_GLM_2"

run_glm_pipeline(data_root_diretory, flat_session_list, output_directory, start_window, stop_window)

"""


# Get Group Results
Visualise_Regression_Results.extract_model_results(flat_session_list, output_directory, start_window, stop_window)
Visualise_Regression_Results.get_group_results(output_directory, nested_session_list)
Visualise_Regression_Results.view_mean_results(output_directory, start_window, stop_window, frame_period)
"""

# Test Significance
condition_1 = "vis_context_vis_2"
condition_2 = "odr_context_vis_2"
Test_Significance_GLM.test_signficance(nested_session_list, output_directory, condition_1, condition_2, start_window, stop_window)