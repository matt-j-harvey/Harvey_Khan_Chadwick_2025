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
from scipy import stats
from sklearn.linear_model import LogisticRegression
from statsmodels.stats import multitest

import Opto_GLM_Utils
import Session_List
import Extract_Onsets
import Get_Activity_Tensors
import Get_Session_Modulation

"""

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




def get_group_modulation_effect(data_root, session_list, start_window, stop_window, comparison_window_start, comparison_window_stop):

    # Perform Decoding for Each Session
    group_modulation_effect = []
    for session in tqdm(session_list):
        session_modulation = Get_Session_Modulation.get_session_modulation(data_root, session, start_window, stop_window, comparison_window_start, comparison_window_stop)
        group_modulation_effect.append(session_modulation)
    group_modulation_effect = np.array(group_modulation_effect)
    return group_modulation_effect



def test_signficance(group_1, group_2):

    # Load Mask
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()

    # Flatten Tensprs
    n_group_1, image_height, image_width = np.shape(group_1)
    group_1 = np.reshape(group_1, (n_group_1, image_height * image_width))
    group_1 = group_1[:, indicies]

    n_group_2, image_height, image_width = np.shape(group_2)
    group_2 = np.reshape(group_2, (n_group_2, image_height * image_width))
    group_2 = group_2[:, indicies]


    t_stats, p_values = stats.ttest_ind(group_1, group_2)
    #print("p_values", np.shape(p_values))
    #rejected, p_values = multitest.fdrcorrection(p_values[0], alpha=0.1)

    t_stats = np.nan_to_num(t_stats, nan=0)
    p_values = np.nan_to_num(p_values, nan=1)

    t_map = np.zeros(image_height * image_width)
    p_map = np.zeros(image_height * image_width)

    t_map[indicies] = t_stats
    p_map[indicies] = p_values

    t_map = np.reshape(t_map, (image_height, image_width))
    p_map = np.reshape(p_map, (image_height, image_width))

    return t_map, p_map


def run_opto_mapping_pipeline(data_root, experiment, output_root, start_window, stop_window, comparison_window_start, comparison_window_stop):

    # Get Session Lists
    opto_session_list = experiment[1]
    control_session_list = experiment[2]

    # Get group modulation effects
    opsin_modulation = get_group_modulation_effect(data_root, opto_session_list, start_window, stop_window, comparison_window_start, comparison_window_stop)
    control_modulation = get_group_modulation_effect(data_root, control_session_list, start_window, stop_window, comparison_window_start, comparison_window_stop)

    print("opsin_modulation", np.shape(opsin_modulation))

    t_map, p_map = test_signficance(opsin_modulation, control_modulation)

    opsin_mean = np.mean(opsin_modulation, axis=0)
    control_mean = np.mean(control_modulation, axis=0)

    mean_diff = np.subtract(opsin_mean, control_mean)
    mean_diff = np.where(p_map < 0.05, mean_diff, 0)
    plt.imshow(mean_diff, cmap="bwr", vmin=-0.2, vmax=0.2)
    plt.show()

    plt.title(experiment[0])
    plt.imshow(t_map, cmap=Opto_GLM_Utils.get_musall_cmap(), vmin=-3, vmax=3)
    plt.show()


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



"""
experiment_list = [

    ["PPC",
    Session_List.ppc_opto_session_list,
    Session_List.ppc_control_session_list],

]
"""



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
    run_opto_mapping_pipeline(data_root, experiment, output_root, start_window, stop_window, comparison_window_start, comparison_window_stop)



