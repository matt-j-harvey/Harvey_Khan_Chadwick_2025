import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import GLM_Utils
import Session_List
import Create_Regression_Matricies
import Plotting_Functions







def get_onsets_reaction_times(onset_list, lick_trace, lick_threshold, max_window=3000):

    reaction_time_list = []
    for onset in onset_list:
        trial_rt = GLM_Utils.get_reaction_time(lick_trace, onset, lick_threshold, max_window)
        if trial_rt != None:
            reaction_time_list.append(trial_rt * 37)

    return reaction_time_list


def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
        onsets = session_trial_tensor_dict["selected_onsets"]
        start_window = session_trial_tensor_dict["start_window"]
        stop_window = session_trial_tensor_dict["stop_window"]
    return activity_tensor, onsets, start_window, stop_window




def get_selected_mousecam_components(glm_output_directory, session):

    # Load Score Matrix
    score_matrix = np.load(os.path.join(glm_output_directory, session, "Parameter_Search", "Ridge_Penalty_Search_Results.npy"))

    # Get Max Indicies
    max_score = np.max(score_matrix)

    max_indicies = np.where(score_matrix == max_score)
    mousecam_index = max_indicies[1][0]

    # Load Possible Values
    mousecam_component_possible_values = np.load(os.path.join(glm_output_directory, session, "Parameter_Search", "Mousecam_component_possible_values.npy"))

    # Get Selected Values
    best_mousecam_n = mousecam_component_possible_values[mousecam_index]

    # Get Design Matrix With Selected Number of Mousecam Component
    return best_mousecam_n




def get_rt_window_trial_indicies(reaction_time_list, rt_window_start, rt_window_stop):
    window_index_list = []
    n_trials = len(reaction_time_list)
    for trial_index in range(n_trials):
        trial_rt = reaction_time_list[trial_index]
        if trial_rt >= rt_window_start and trial_rt < rt_window_stop:
            window_index_list.append(trial_index)
    return window_index_list



def get_model_outputs_by_rt(data_root, session, output_root, rt_window_start_list, rt_window_stop_list, max_mousecam_components):

    # Load Model Coefs and Intercepts
    model_coefs = np.load(os.path.join(output_root, session, "Model_Output", "Model_Coefs.npy"))
    model_intercept = np.load(os.path.join(output_root, session, "Model_Output", "model_intercept.npy"))
    print("model_coefs", np.shape(model_coefs))
    print("model_intercept", np.shape(model_intercept))

    # Get Best Mousecam Components
    best_mousecam_n = get_selected_mousecam_components(output_root, session)
    print("best_mousecam_n", best_mousecam_n)

    # Load Lick Trace
    lick_trace = GLM_Utils.load_lick_trace(data_root, session)

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(data_root, session, "Lick_Threshold.npy"))

    # Load Activity Tensor
    activity_tensor, vis_1_onsets, start_window, stop_window = open_tensor(os.path.join(output_root, session, "Activity_Tensors", "vis_1_correct"))
    n_nmf_components = np.shape(activity_tensor)[2]

    # Load Behaviour Tensor
    behaviour_tensor, vis_1_onsets, start_window, stop_window = open_tensor(os.path.join(output_root, session, "Behaviour_Tensors", "vis_1_correct"))
    print("behaviour_tensor", np.shape(behaviour_tensor))
    print("start_window", start_window, "stop_window", stop_window)

    # Cut To Best N Mousecam Components
    n_behaviour_regressors = np.shape(behaviour_tensor)[2]
    n_non_mousecam_regressors = n_behaviour_regressors - max_mousecam_components
    mousecam_stop = n_non_mousecam_regressors + best_mousecam_n
    behaviour_tensor = behaviour_tensor[:, :, 0:mousecam_stop]

    print("behaviour_tensor best mousecam", np.shape(behaviour_tensor))

    # Get n timepoints
    n_timepoints = stop_window - start_window

    # Get Onset Reaction Times
    reaction_time_list = get_onsets_reaction_times(vis_1_onsets, lick_trace, lick_threshold, max_window=3000)

    # Iterate Through Each RT Window
    n_rt_windows = len(rt_window_start_list)
    for rt_window_index in range(n_rt_windows):
        rt_window_start = rt_window_start_list[rt_window_index]
        rt_window_stop = rt_window_stop_list[rt_window_index]

        # Get Selected Indicies
        window_index_list = get_rt_window_trial_indicies(reaction_time_list, rt_window_start, rt_window_stop)
        n_window_trials = len(window_index_list)

        # If There is More than 1 Trial in this window
        if n_window_trials > 1:

            # Get Real Activity Tensor
            real_activity_tensor = activity_tensor[window_index_list]
            real_activity_mean = np.mean(real_activity_tensor, axis=0)
            real_activity_mean = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, real_activity_mean)

            # Get RT Behaviour Tensor
            rt_window_behaviour_tensor = behaviour_tensor[window_index_list]
            behaviour_regressor = np.vstack(rt_window_behaviour_tensor)

            # Get Mousecam Only Behaviour Regressor
            mousecam_only_regessor = np.copy(behaviour_regressor)
            mousecam_only_regessor[:, 0:n_non_mousecam_regressors] = 0

            # Create Stimuli Regressors
            stimulus_regressor = Create_Regression_Matricies.create_stimuli_regressor([n_window_trials, 0], n_timepoints)

            # Combine Regressors Into Design Matrix
            design_matrix = np.hstack([stimulus_regressor, behaviour_regressor])
            mousecam_only_design_matrix = np.hstack([np.zeros(np.shape(stimulus_regressor)), mousecam_only_regessor])

            # Make Prediction for RT Window
            rt_window_prediction = np.matmul(design_matrix, np.transpose(model_coefs))
            rt_window_prediction_mousecam = np.matmul(mousecam_only_design_matrix, np.transpose(model_coefs))

            # Reshape Back Into Tensor
            rt_window_prediction = np.reshape(rt_window_prediction, (n_window_trials, n_timepoints, n_nmf_components))
            rt_window_prediction_mousecam = np.reshape(rt_window_prediction_mousecam, (n_window_trials, n_timepoints, n_nmf_components))

            # Get RT Bin Mean
            mean_rt_window = np.mean(rt_window_prediction, axis=0)
            mean_rt_window = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, mean_rt_window)

            mean_rt_window_mousecam = np.mean(rt_window_prediction_mousecam, axis=0)
            mean_rt_window_mousecam = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, mean_rt_window_mousecam)

            # Save These
            save_directory = os.path.join(output_root, session, "RT_Window_Predictions")
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            file_name = str(rt_window_start) + "_to_" + str(rt_window_stop)
            np.save(os.path.join(save_directory, file_name + "_predicted.npy"), mean_rt_window)
            np.save(os.path.join(save_directory, file_name + "_real.npy"), real_activity_mean)
            np.save(os.path.join(save_directory, file_name + "_mousecam_only.npy"), mean_rt_window_mousecam)




max_mousecam_components = 500
rt_window_start_list = [500, 750, 1000, 1250, 1500, 1750, 2000]
rt_window_stop_list = np.add(rt_window_start_list, 250)
print("rt_window_start_list", rt_window_start_list)
print("rt_window_stop_list", rt_window_stop_list)

roi_list = [15]
frame_period = 37
start_window_ms = -2800
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

control_session_list = Session_List.control_all_post_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"

hom_session_list = Session_List.neurexin_all_post_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Neurexin_GLM\Post_Learning\Homs"

"""
for session in Session_List.flatten_nested_list(control_session_list):
    get_model_outputs_by_rt(control_data_root, session, control_output_root, rt_window_start_list, rt_window_stop_list, max_mousecam_components)
"""
Plotting_Functions.plot_roi_by_rt_window(control_session_list, control_output_root, start_window, stop_window, rt_window_start_list, rt_window_stop_list, roi_list)

"""
for session in Session_List.flatten_nested_list(hom_session_list):
    get_model_outputs_by_rt(hom_data_root, session, hom_output_root, rt_window_start_list, rt_window_stop_list, max_mousecam_components)
"""

Plotting_Functions.plot_roi_by_rt_window(hom_session_list, hom_output_root, start_window, stop_window, rt_window_start_list, rt_window_stop_list, roi_list)

