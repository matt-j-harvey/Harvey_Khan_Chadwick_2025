import os
import numpy as np
from tqdm import tqdm
import sys
import pickle

import GLM_Utils




def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor



def create_stimuli_regressor(n_trials, n_timepoints):

    n_stimuli = len(n_trials)
    total_trials = np.sum(n_trials)
    stimuli_regressor = np.zeros((total_trials * n_timepoints, n_stimuli * n_timepoints))

    for stimulus_index in range(n_stimuli):
        regressor_start = stimulus_index * n_timepoints
        regressor_stop = regressor_start + n_timepoints

        for trial_index in range(n_trials[stimulus_index]):
            trial_start = trial_index * n_timepoints
            trial_stop = trial_start + n_timepoints

            stimuli_regressor[trial_start:trial_stop, regressor_start:regressor_stop] = np.eye(n_timepoints)

    return stimuli_regressor


def load_data(mvar_directory_root, session, context):

    # Load Activity Tensors
    vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", context + "_context_stable_vis_1"))
    vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", context + "_context_stable_vis_2"))
    print("vis_1_activity_tensor", np.shape(vis_1_activity_tensor))
    print("vis_2_activity_tensor", np.shape(vis_2_activity_tensor))

    # Load Behaviour Tensors
    vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", context + "_context_stable_vis_1"))
    vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", context + "_context_stable_vis_2"))
    print("vis_1_behaviour_tensor", np.shape(vis_1_behaviour_tensor))
    print("vis_2_behaviour_tensor", np.shape(vis_2_behaviour_tensor))

    return [vis_1_activity_tensor, vis_2_activity_tensor, vis_1_behaviour_tensor, vis_2_behaviour_tensor]




def create_regression_matricies(session, mvar_directory_root, context):

    # Load Data
    [vis_1_activity_tensor,
     vis_2_activity_tensor,
     vis_1_behaviour_tensor,
     vis_2_behaviour_tensor] = load_data(mvar_directory_root, session, context)

    # Create Delta F Matrix
    vis_1_activity_tensor = np.vstack(vis_1_activity_tensor)
    vis_2_activity_tensor = np.vstack(vis_2_activity_tensor)
    delta_f_matrix = np.vstack([vis_1_activity_tensor, vis_2_activity_tensor])

    # Create Behaviour Regressor
    vis_1_behaviour_tensor = np.vstack(vis_1_behaviour_tensor)
    vis_2_behaviour_tensor = np.vstack(vis_2_behaviour_tensor)
    behaviour_regressor = np.vstack([vis_1_behaviour_tensor, vis_2_behaviour_tensor])

    # Create Stimuli Regressors
    n_trials = [np.shape(vis_1_activity_tensor)[0], np.shape(vis_2_activity_tensor)[0]]
    n_timepoints = np.shape(vis_1_activity_tensor)[1]
    stimulus_regressor = create_stimuli_regressor(n_trials, n_timepoints)

    # Combine Regressors Into Design Matrix
    DesignMatrix = np.hstack([stimulus_regressor, behaviour_regressor])

    return DesignMatrix, delta_f_matrix