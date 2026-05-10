import os
import numpy as np
from tqdm import tqdm
import sys
import pickle
import matplotlib.pyplot as plt

import Opto_GLM_Utils




def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor


"""
def create_stimuli_regressor(n_trials):

    n_stimuli = len(n_trials)
    total_trials = np.sum(n_trials)

    stimulus_regressor = np.zeros((total_trials, n_stimuli))

    regressor_start = 0
    for stimulus_index in range(n_stimuli):
        regressor_stop = regressor_start + n_trials[stimulus_index]
        stimulus_regressor[regressor_start:regressor_stop, stimulus_index] = 1
        regressor_start = regressor_stop

    return stimulus_regressor
"""


def create_stimuli_regressor(n_trials, n_timepoints):

    n_stimuli = len(n_trials)
    total_trials = np.sum(n_trials)

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



def baseline_correct_tensor(tensor, baseline_start=0, baseline_stop=14):

    baseline_corrected_tensor = []
    for trial in tensor:
        trial_baseline = trial[baseline_start:baseline_stop]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        baseline_corrected_tensor.append(trial)

    return baseline_corrected_tensor



def get_brain_pixels(data_tensor):
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()
    n_trials, image_height, image_width = np.shape(data_tensor)
    data_tensor = np.reshape(data_tensor, (n_trials, image_height * image_width))
    data_tensor = data_tensor[:, indicies]
    print("data_tensor", np.shape(data_tensor))
    return data_tensor



def create_regression_matricies(data_directory, session, mvar_directory_root, z_score, baseline_correct):

    # Load Activity Tensors
    visual_context_control_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "visual" + "_context_control"))
    visual_context_light_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "visual" + "_context_light"))
    odour_context_control_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour" + "_context_control"))
    odour_context_light_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour" + "_context_light"))
    print("visual_context_control_activity_tensor", np.shape(visual_context_control_activity_tensor))

    # Load Behaviour Tensors
    visual_context_control_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "visual" + "_context_control"))
    visual_context_light_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "visual" + "_context_light"))
    odour_context_control_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour" + "_context_control"))
    odour_context_light_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour" + "_context_light"))
    print("visual_context_control_behaviour_tensor", np.shape(visual_context_control_behaviour_tensor))

    # Get Trial Structure
    n_visual_context_control_trials = np.shape(visual_context_control_activity_tensor)[0]
    n_visual_context_light_trials = np.shape(visual_context_light_activity_tensor)[0]
    n_odour_context_control_trials = np.shape(odour_context_control_activity_tensor)[0]
    n_odour_context_light_trials = np.shape(odour_context_light_activity_tensor)[0]
    n_timepoints = np.shape(visual_context_control_activity_tensor)[1]

    # Create Behaviour Regressor
    visual_context_control_behaviour_tensor = np.vstack(visual_context_control_behaviour_tensor)
    visual_context_light_behaviour_tensor = np.vstack(visual_context_light_behaviour_tensor)
    odour_context_control_behaviour_tensor = np.vstack(odour_context_control_behaviour_tensor)
    odour_context_light_behaviour_tensor = np.vstack(odour_context_light_behaviour_tensor)
    behaviour_regressor = np.vstack([visual_context_control_behaviour_tensor,
                                     visual_context_light_behaviour_tensor,
                                     odour_context_control_behaviour_tensor,
                                     odour_context_light_behaviour_tensor])

    # Create Stimuli Regressors
    n_trials = [n_visual_context_control_trials, n_visual_context_light_trials, n_odour_context_control_trials, n_odour_context_light_trials]
    stimulus_regressor = create_stimuli_regressor(n_trials, n_timepoints)
    print("stimulus_regressor", np.shape(stimulus_regressor))

    # Combine Regressors Into Design Matrix
    #design_matrix = np.hstack([stimulus_regressor, behaviour_regressor])
    design_matrix = stimulus_regressor

    # Create Delta F Matrix
    visual_context_control_activity_tensor = np.vstack(visual_context_control_activity_tensor)
    visual_context_light_activity_tensor = np.vstack(visual_context_light_activity_tensor)
    odour_context_control_activity_tensor = np.vstack(odour_context_control_activity_tensor)
    odour_context_light_activity_tensor = np.vstack(odour_context_light_activity_tensor)


    delta_f_matrix = np.vstack([visual_context_control_activity_tensor,
                                visual_context_light_activity_tensor,
                                odour_context_control_activity_tensor,
                                odour_context_light_activity_tensor])
    #delta_f_matrix = np.squeeze(delta_f_matrix)

    return design_matrix, delta_f_matrix