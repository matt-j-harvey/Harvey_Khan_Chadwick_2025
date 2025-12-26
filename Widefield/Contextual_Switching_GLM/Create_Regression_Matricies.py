import os

number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import numpy as np
from tqdm import tqdm
import sys
import pickle
import matplotlib.pyplot as plt

import GLM_Utils




def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor



def create_stimuli_regressor(n_trials, n_timepoints):

    n_stimuli = len(n_trials)
    total_trials = np.sum(n_trials)

    #print("total trials", total_trials)
    #print("n_timepoints", n_timepoints)
    #print("n stimuli", n_stimuli)

    combined_stimuli_regressors = []


    for stimulus_index in range(n_stimuli):

        stimulus_regressor = np.zeros((n_trials[stimulus_index] * n_timepoints, n_stimuli * n_timepoints))
        regressor_start = stimulus_index * n_timepoints
        regressor_stop = regressor_start + n_timepoints

        for trial_index in range(n_trials[stimulus_index]):
            trial_start = trial_index * n_timepoints
            trial_stop = trial_start + n_timepoints
            stimulus_regressor[trial_start:trial_stop, regressor_start:regressor_stop] = np.eye(n_timepoints)

        combined_stimuli_regressors.append(stimulus_regressor)

    combined_stimuli_regressors = np.vstack(combined_stimuli_regressors)

    return combined_stimuli_regressors


"""
def reconstruct_df_into_pixel_space(data_directory_root, session, glm_output_root, delta_f_matrix):

    # Load Registered SVD
    reg_u = np.load(os.path.join(data_directory_root, session, "Preprocessed_Data", "Registered_U.npy"))

    # Flatten Reg U
    image_height, image_width, n_components = np.shape(reg_u)
    reg_u = np.reshape(reg_u, (image_height * image_width, n_components))

    # load mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Mask Reg U
    reg_u = reg_u[indicies]

    # Recosntruct
    reconstruction = np.dot(delta_f_matrix, reg_u.T)

    return reconstruction
"""


def reconstruct_tensor_into_pixel_space(data_directory_root, session, tensor):

    # Load Registered SVD
    #reg_u = np.load(os.path.join(data_directory_root, session, "Preprocessed_Data", "Registered_U.npy"))
    spatial_components = np.load(os.path.join(data_directory_root, session, "Local_NMF", "Spatial_Components.npy"))

    # Flatten Reg U
    image_height, image_width, n_components = np.shape(spatial_components)
    spatial_components = np.reshape(spatial_components, (image_height * image_width, n_components))

    # load mask
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()

    # Mask Reg U
    spatial_components = spatial_components[indicies]

    # Flatten Tensor
    n_trials, n_timepoints, n_components = np.shape(tensor)
    tensor = np.reshape(tensor, (n_trials * n_timepoints, n_components))

    # Reconstruct
    tensor = np.dot(tensor, spatial_components.T)

    # Reshape back Into Tensor
    n_pixels = np.shape(indicies)[1]
    tensor = np.reshape(tensor, (n_trials, n_timepoints, n_pixels))

    return tensor


def view_mean_tensor(tensor):

    # load mask
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()

    mean_tensor = np.mean(tensor, axis=0)

    count = 0
    for timepoint in mean_tensor:
        plt.title(str(count))
        reconstruction = GLM_Utils.create_image_from_data(timepoint, indicies, image_height, image_width)
        plt.imshow(reconstruction, cmap=GLM_Utils.get_musall_cmap(), vmin=-0.5, vmax=0.5)
        plt.show()

        count +=1



def z_score_delta_f_matrix(delta_f_matrix, data_directory, session):

    # Load Mean and SD
    session_mean = np.load(os.path.join(data_directory, session, "Z_Scoring", "pixel_means.npy"))
    session_std = np.load(os.path.join(data_directory, session, "Z_Scoring", "pixel_stds.npy"))

    print("delta_f_matrix z score shape", np.shape(delta_f_matrix))

    # Subtract Mean
    delta_f_matrix = np.subtract(delta_f_matrix, session_mean)

    # Divide By STD
    delta_f_matrix = np.divide(delta_f_matrix, session_std)
    delta_f_matrix = np.nan_to_num(delta_f_matrix)

    return delta_f_matrix


def z_score_tensors(data_directory, session, tensor):

    # Load Mean and SD
    session_mean = np.load(os.path.join(data_directory, session, "Z_Scoring", "pixel_means.npy"))
    session_std = np.load(os.path.join(data_directory, session, "Z_Scoring", "pixel_stds.npy"))

    # Flatten Tensor
    n_trials, n_timepoints, n_pixels = np.shape(tensor)
    tensor = np.reshape(tensor, (n_trials * n_timepoints, n_pixels))

    # Subtract Mean
    tensor = np.subtract(tensor, session_mean)

    # Divide By STD
    tensor = np.divide(tensor, session_std)
    tensor = np.nan_to_num(tensor)

    # Reshape Back Into Tensor
    tensor = np.reshape(tensor, (n_trials, n_timepoints, n_pixels))

    return tensor


def baseline_correct_tensor(tensor, baseline_start=0, baseline_stop=14):

    baseline_corrected_tensor = []
    for trial in tensor:
        trial_baseline = trial[baseline_start:baseline_stop]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        baseline_corrected_tensor.append(trial)

    return baseline_corrected_tensor




def create_regression_matricies(data_directory, session, mvar_directory_root, z_score, baseline_correct):

    # Load Activity Tensors
    visual_context_vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "visual" + "_context_stable_vis_1_control"))
    visual_context_vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "visual" + "_context_stable_vis_2_control"))
    odour_context_vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour" + "_context_stable_vis_1_control"))
    odour_context_vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour" + "_context_stable_vis_2_control"))
    print("visual_context_vis_1_activity_tensor", np.shape(visual_context_vis_1_activity_tensor))

    # Reconstruct Tensors Into Pixel Space
    #visual_context_vis_1_activity_tensor = reconstruct_tensor_into_pixel_space(data_directory, session, visual_context_vis_1_activity_tensor)
    #visual_context_vis_2_activity_tensor = reconstruct_tensor_into_pixel_space(data_directory, session, visual_context_vis_2_activity_tensor)
    #odour_context_vis_1_activity_tensor = reconstruct_tensor_into_pixel_space(data_directory, session, odour_context_vis_1_activity_tensor)
    #odour_context_vis_2_activity_tensor = reconstruct_tensor_into_pixel_space(data_directory, session, odour_context_vis_2_activity_tensor)
    #print("visual_context_vis_1_activity_tensor", np.shape(visual_context_vis_1_activity_tensor))

    # Z Score If Required
    if z_score == True:
        visual_context_vis_1_activity_tensor = z_score_tensors(data_directory, session, visual_context_vis_1_activity_tensor)
        visual_context_vis_2_activity_tensor = z_score_tensors(data_directory, session, visual_context_vis_2_activity_tensor)
        odour_context_vis_1_activity_tensor = z_score_tensors(data_directory, session, odour_context_vis_1_activity_tensor)
        odour_context_vis_2_activity_tensor = z_score_tensors(data_directory, session, odour_context_vis_2_activity_tensor)


    # Load Behaviour Tensors
    visual_context_vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "visual" + "_context_stable_vis_1_control"))
    visual_context_vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "visual" + "_context_stable_vis_2_control"))
    odour_context_vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour" + "_context_stable_vis_1_control"))
    odour_context_vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour" + "_context_stable_vis_2_control"))

    # Baseline Correct If Required
    if baseline_correct == True:

        visual_context_vis_1_activity_tensor = baseline_correct_tensor(visual_context_vis_1_activity_tensor)
        visual_context_vis_2_activity_tensor = baseline_correct_tensor(visual_context_vis_2_activity_tensor)
        odour_context_vis_1_activity_tensor = baseline_correct_tensor(odour_context_vis_1_activity_tensor)
        odour_context_vis_2_activity_tensor = baseline_correct_tensor(odour_context_vis_2_activity_tensor)

        visual_context_vis_1_behaviour_tensor = baseline_correct_tensor(visual_context_vis_1_behaviour_tensor)
        visual_context_vis_2_behaviour_tensor = baseline_correct_tensor(visual_context_vis_2_behaviour_tensor)
        odour_context_vis_1_behaviour_tensor = baseline_correct_tensor(odour_context_vis_1_behaviour_tensor)
        odour_context_vis_2_behaviour_tensor = baseline_correct_tensor(odour_context_vis_2_behaviour_tensor)


    #view_mean_tensor(visual_context_vis_1_activity_tensor)

    # Get Trial Structure
    n_visual_context_vis_1_trials = np.shape(visual_context_vis_1_activity_tensor)[0]
    n_visual_context_vis_2_trials = np.shape(visual_context_vis_2_activity_tensor)[0]
    n_odour_context_vis_1_trials = np.shape(odour_context_vis_1_activity_tensor)[0]
    n_odour_context_vis_2_trials = np.shape(odour_context_vis_2_activity_tensor)[0]
    n_timepoints = np.shape(visual_context_vis_1_activity_tensor)[1]

    # Create Behaviour Regressor
    visual_context_vis_1_behaviour_tensor = np.vstack(visual_context_vis_1_behaviour_tensor)
    visual_context_vis_2_behaviour_tensor = np.vstack(visual_context_vis_2_behaviour_tensor)
    odour_context_vis_1_behaviour_tensor = np.vstack(odour_context_vis_1_behaviour_tensor)
    odour_context_vis_2_behaviour_tensor = np.vstack(odour_context_vis_2_behaviour_tensor)
    behaviour_regressor = np.vstack([visual_context_vis_1_behaviour_tensor, visual_context_vis_2_behaviour_tensor, odour_context_vis_1_behaviour_tensor, odour_context_vis_2_behaviour_tensor])

    # Create Stimuli Regressors
    n_trials = [n_visual_context_vis_1_trials, n_visual_context_vis_2_trials, n_odour_context_vis_1_trials, n_odour_context_vis_2_trials]
    stimulus_regressor = create_stimuli_regressor(n_trials, n_timepoints)

    # Combine Regressors Into Design Matrix
    design_matrix = np.hstack([stimulus_regressor, behaviour_regressor])

    # Create Delta F Matrix
    visual_context_vis_1_activity_tensor = np.vstack(visual_context_vis_1_activity_tensor)
    print("visual_context_vis_1_activity_tensor", np.shape(visual_context_vis_1_activity_tensor))
    visual_context_vis_2_activity_tensor = np.vstack(visual_context_vis_2_activity_tensor)
    odour_context_vis_1_activity_tensor = np.vstack(odour_context_vis_1_activity_tensor)
    odour_context_vis_2_activity_tensor = np.vstack(odour_context_vis_2_activity_tensor)
    delta_f_matrix = np.vstack([visual_context_vis_1_activity_tensor, visual_context_vis_2_activity_tensor, odour_context_vis_1_activity_tensor, odour_context_vis_2_activity_tensor])

    return design_matrix, delta_f_matrix