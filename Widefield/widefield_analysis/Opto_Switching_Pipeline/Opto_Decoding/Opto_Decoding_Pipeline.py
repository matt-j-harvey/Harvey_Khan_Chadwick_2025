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
from sklearn.linear_model import LogisticRegression

import Session_List
import Extract_Onsets
import Get_Activity_Tensors
import Perform_CV_Decoding
"""

import Opto_GLM_Utils
import Create_Behavioural_Regressor_Matrix

import Create_Regression_Matricies_Mean
import Parameter_Search
import Fit_Full_Model
import Get_Group_Average_Results
import View_Average_Opto_Results


import View_Average_Opto_Results
"""

def get_combined_data(condition_1_data, condition_2_data):
    combined_data = np.vstack([condition_1_data, condition_2_data])
    condition_1_labels = np.ones(np.shape(condition_1_data)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_data)[0])
    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])

    print("combined Data", np.shape(combined_data))
    print("combined lables", np.shape(combined_labels))

    return combined_data, combined_labels



def run_opto_mapping_pipeline(data_root, experiment, output_root, start_window, stop_window, regressor_list, comparison_window_start, comparison_window_stop):

    opto_session_list = experiment[1]
    control_session_list = experiment[2]
    session_list =  opto_session_list + control_session_list

    # Perform Decoding for Each Session
    for session in tqdm(session_list, desc="Mouse"):
        print(session)

        # Extract Onsets
        visual_context_light_onsets, odour_context_light_onsets = Extract_Onsets.extract_opto_mapping_onsets(data_root, session, output_root)

        # Create Activity Tensors
        #visual_context_tensor = Get_Activity_Tensors.create_activity_tensor(data_root, session, visual_context_light_onsets, start_window, stop_window, comparison_window_start, comparison_window_stop)
        #odour_context_tensor = Get_Activity_Tensors.create_activity_tensor(data_root, session, odour_context_light_onsets, start_window, stop_window, comparison_window_start, comparison_window_stop)

        visual_context_tensor = Get_Activity_Tensors.get_activity_tensor_nmf(data_root, session, visual_context_light_onsets, start_window, stop_window)
        odour_context_tensor = Get_Activity_Tensors.get_activity_tensor_nmf(data_root, session, odour_context_light_onsets, start_window, stop_window)

        print("visual_context_tensor", np.shape(visual_context_tensor))
        print("odour_context_tensor", np.shape(odour_context_tensor))

        # Combine Data
        combined_data, combined_labels = get_combined_data(visual_context_tensor, odour_context_tensor)

        # Perform Decoding
        n_timepoints = np.shape(combined_data)[1]
        score_list = []
        for timepoint_index in range(n_timepoints):

            # Get Timepoint Data
            timepoint_data = combined_data[:, timepoint_index]

            model = LogisticRegression()

            # Perform Decoding
            average_score, average_coefs = Perform_CV_Decoding.perform_cv(model, x_all=timepoint_data, y_all=combined_labels, n_balance_iterations=20, n_folds=5)
            score_list.append(average_score)

        plt.plot(score_list)
        plt.show()

        """
        visual_mean = np.mean(visual_context_tensor, axis=0)
        odour_mean = np.mean(odour_context_tensor, axis=0)
        diff = np.subtract(visual_mean, odour_mean)

        plt.imshow(visual_mean, cmap="bwr", vmin=-2, vmax=2)
        plt.show()

        plt.imshow(odour_mean, cmap="bwr", vmin=-2, vmax=2)
        plt.show()

        plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        plt.show()
        """

regressor_list = ["Visual_Context_Control",
                  "Visual_Context_Light",
                  "Odour_Context_Control",
                  "Odour_Context_Light"]



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




experiment_list = [

    ["PPC",
    Session_List.ppc_opto_session_list,
    Session_List.ppc_control_session_list],

]




# Set Directories
data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Widefield_Opto"
output_root = r"C:\Analysis_Output\Opto_Decoding"

# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 0
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)
comparison_window_start = 14
comparison_window_stop = comparison_window_start + 28

print("start_window", start_window)
print("stop_window", stop_window)
print("comparison_window_start", comparison_window_start)
print("comparison_window_stop", comparison_window_stop)


# Run Pipeline
for experiment in experiment_list:
    run_opto_mapping_pipeline(data_root, experiment, output_root, start_window, stop_window, regressor_list, comparison_window_start, comparison_window_stop)




