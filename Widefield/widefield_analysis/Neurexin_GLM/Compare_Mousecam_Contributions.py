import numpy as np
import os
import pickle
import matplotlib.pyplot as plt

import GLM_Utils
import Session_List
import Create_Regression_Matricies
import Plotting_Functions




def get_group_facecam_activity(data_root, session_list, output_root, start_window, stop_window):

    group_list = []
    for mouse in session_list:
        mouse_list = []

        for session in mouse:
            session_activity = get_session_mousecam_contributions(data_root, session, output_root, start_window, stop_window)
            mouse_list.append(session_activity)

        mouse_list = np.array(mouse_list)
        mouse_mean = np.mean(mouse_list, axis=0)
        group_list.append(mouse_mean)

    group_list = np.array(group_list)

    save_directory = os.path.join(output_root, "Group_Coefs")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Group_Mousecam_Activity_Vis_2.npy"), group_list)


    group_mean = np.mean(group_list, axis=0)
    plt.ion()
    count = 0
    for frame in group_mean:
        plt.imshow(frame, cmap=GLM_Utils.get_musall_cmap(), vmin=-1, vmax=1)
        plt.title(str(count))
        count += 1
        plt.pause(0.1)
        plt.draw()
        plt.pause(0.1)
        plt.clf()#
    plt.ioff()




def get_session_n_mousecam(output_root, session):

    # Load Score Matrix
    score_matrix = np.load(os.path.join(output_root, session, "Parameter_Search", "Ridge_Penalty_Search_Results.npy"))

    # Get Max Indicies
    max_score = np.max(score_matrix)
    max_indicies = np.where(score_matrix == max_score)
    mousecam_index = max_indicies[1][0]

    # Load Possible Values
    mousecam_component_possible_values = np.load(os.path.join(output_root, session, "Parameter_Search", "Mousecam_component_possible_values.npy"))

    # Get Selected Values
    best_mousecam_n = mousecam_component_possible_values[mousecam_index]

    return best_mousecam_n


def get_session_mousecam_contributions(data_root, session, output_root, start_window, stop_window, max_mousecam_components=500):

    # Load Model Coefs
    model_coefs = np.load(os.path.join(output_root, session, "Model_Output", "Model_Coefs.npy"))
    print("model_coefs", np.shape(model_coefs))

    # Load Behaviour Tensor
    behaviour_tensor = GLM_Utils.open_tensor(os.path.join(output_root, session, "Behaviour_Tensors", "vis_2_correct"))
    print("behaviour_tensor", np.shape(behaviour_tensor))

    # Get Facecam Coefs
    n_timepoints = stop_window - start_window
    n_stim_regressors = n_timepoints * 2
    n_behaviour_regressors = np.shape(behaviour_tensor)[2]
    n_non_mousecam_behaviour_regressors = n_behaviour_regressors - max_mousecam_components
    n_mousecam_regressors = get_session_n_mousecam(output_root, session)
    mousecam_regressor_start = n_stim_regressors + n_non_mousecam_behaviour_regressors
    mousecam_regressor_stop = mousecam_regressor_start + n_mousecam_regressors
    mousecam_coefs = model_coefs[:, mousecam_regressor_start:mousecam_regressor_stop]

    # Get Behaviour Tensor Mousecam
    behaviour_tensor = behaviour_tensor[:, :, n_non_mousecam_behaviour_regressors:n_non_mousecam_behaviour_regressors + n_mousecam_regressors]
    n_trials, n_timepoints, n_mousecam_regressors = np.shape(behaviour_tensor)
    behaviour_tensor = np.reshape(behaviour_tensor, (n_trials * n_timepoints, n_mousecam_regressors))

    # Make Prediction
    print("mousecam_coefs", np.shape(mousecam_coefs))
    print("behaviour_tensor", np.shape(behaviour_tensor))
    prediction = np.dot(behaviour_tensor, np.transpose(mousecam_coefs))
    n_nmf_components = np.shape(mousecam_coefs)[0]
    print("prediciton", np.shape(prediction))

    prediction = np.reshape(prediction, (n_trials, n_timepoints, n_nmf_components))
    mean_prediction = np.mean(prediction, axis=0)
    print("mean_prediction", np.shape(mean_prediction))

    mean_prediction = GLM_Utils.reconstruct_regressor_into_pixel_space(data_root, session, mean_prediction)
    return mean_prediction




def view_regressors(control_output_root, neurexin_output_root, start_window, stop_window, results_root):

    # Get X Values
    x_values = list(range(start_window, stop_window))
    current_x_values = np.multiply(x_values, 36)

    # Load Coefs
    control_face_cr = np.load(os.path.join(control_output_root, "Group_Coefs", "Group_Mousecam_Activity_Vis_2.npy"))
    neurexin_face_cr = np.load(os.path.join(neurexin_output_root, "Group_Coefs", "Group_Mousecam_Activity_Vis_2.npy"))

    graph_save_directory = os.path.join(results_root, "ROI_Graphs")
    if not os.path.exists(graph_save_directory):
        os.makedirs(graph_save_directory)

    Plotting_Functions.compare_roi_across_groups(control_face_cr, neurexin_face_cr, graph_save_directory, start_window, stop_window, [14,15,16], "Face CRs")

    # Get Mean Coefs
    control_mean = np.mean(control_face_cr, axis=0)
    neurexin_mean = np.mean(neurexin_face_cr, axis=0)
    mean_diff = np.subtract(neurexin_mean, control_mean)

    """
    print("Getting sig diff")
    t_stats, p_values = stats.ttest_ind(control_lick_coefs, neurexin_lick_coefs, axis=0)
    p_values = np.nan_to_num(p_values, nan=1)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    mean_diff = np.multiply(binary_sig, mean_diff)
    """

    # Linearly Interpolate
    control_mean, new_x_values = GLM_Utils.linearly_interpolate(control_mean, 36, 10, current_x_values, 0)
    neurexin_mean, new_x_values = GLM_Utils.linearly_interpolate(neurexin_mean, 36, 10, current_x_values, 0)
    mean_diff, new_x_values = GLM_Utils.linearly_interpolate(mean_diff, 36, 10, current_x_values, 0)

    print("new x values", new_x_values)

    # Create Save Directories
    save_directory_list = [
        os.path.join(results_root, "Group_Regressor_Maps", "Control_Face_CR"),
        os.path.join(results_root, "Group_Regressor_Maps", "Neurexin_Face_CR"),
        os.path.join(results_root, "Group_Regressor_Maps", "Face_CR_Diff"),
    ]

    for directory in save_directory_list:
        if not os.path.exists(directory):
            os.makedirs(directory)

    #Plotting_Functions.visualise_mean_regressor(control_mean,  save_directory_list[0], new_x_values, magnitude=[0, 0.4], cmap=plt.get_cmap('inferno'))
    #Plotting_Functions.visualise_mean_regressor(neurexin_mean, save_directory_list[1], new_x_values, magnitude=[0, 0.4], cmap=plt.get_cmap('inferno'))
    #Plotting_Functions.visualise_mean_regressor(mean_diff,     save_directory_list[2], new_x_values, magnitude=[-0.2, 0.2], cmap=GLM_Utils.get_musall_cmap())


def compare_roi_mean_over_learning(output_root,
                                   pre_session_list,
                                   pre_start_window,
                                   pre_stop_window,
                                   post_session_list,
                                   post_start_window,
                                   post_stop_window):





max_mousecam_components = 500

frame_period = 37
start_window_ms = -2800 #-2800 for post learning
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


#control_session_list = Session_List.control_all_post_learning
control_session_list = Session_List.control_post_learning_discrimination
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"

#hom_session_list = Session_List.neurexin_all_post_learning
hom_session_list = Session_List.neurexin_post_learning_discrimination
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Neurexin_GLM\Post_Learning\Homs"

results_root = r"C:\Analysis_Output\Neurexin_GLM\Post_Learning_Discrimination_Only\Group_Results"

#get_group_facecam_activity(control_data_root, control_session_list, control_output_root, start_window, stop_window)
#get_group_facecam_activity(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window)
view_regressors(control_output_root, hom_output_root, start_window, stop_window, results_root)



control_session_list = Session_List.control_intermediate_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Neurexin_GLM\Intermediate_Learning\Controls"

hom_session_list = Session_List.neurexin_intermediate_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Neurexin_GLM\Intermediate_Learning\Homs"

results_root = r"C:\Analysis_Output\Neurexin_GLM\Intermediate_Learning_Discrimination_Only\Group_Results"

#get_group_facecam_activity(control_data_root, control_session_list, control_output_root, start_window, stop_window)
#get_group_facecam_activity(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window)
#view_regressors(control_output_root, hom_output_root, start_window, stop_window, results_root)
