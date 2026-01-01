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
    vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "vis_1_correct"))
    vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "vis_2_correct"))
    print("vis_1_activity_tensor", np.shape(vis_1_activity_tensor))

    # Load Behaviour Tensors
    vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "vis_1_correct"))
    vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "vis_2_correct"))

    # Baseline Correct If Required
    if baseline_correct == True:

        vis_1_activity_tensor = baseline_correct_tensor(vis_1_activity_tensor)
        vis_2_activity_tensor = baseline_correct_tensor(vis_2_activity_tensor)

        vis_1_behaviour_tensor = baseline_correct_tensor(vis_1_behaviour_tensor)
        vis_2_behaviour_tensor = baseline_correct_tensor(vis_2_behaviour_tensor)


    # Get Trial Structure
    n_vis_1_trials = np.shape(vis_1_activity_tensor)[0]
    n_vis_2_trials = np.shape(vis_2_activity_tensor)[0]
    n_timepoints = np.shape(vis_1_activity_tensor)[1]

    # Create Behaviour Regressor
    vis_1_behaviour_tensor = np.vstack(vis_1_behaviour_tensor)
    vis_2_behaviour_tensor = np.vstack(vis_2_behaviour_tensor)
    behaviour_regressor = np.vstack([vis_1_behaviour_tensor, vis_2_behaviour_tensor])

    # Create Stimuli Regressors
    n_trials = [n_vis_1_trials, n_vis_2_trials]
    stimulus_regressor = create_stimuli_regressor(n_trials, n_timepoints)

    # Combine Regressors Into Design Matrix
    design_matrix = np.hstack([stimulus_regressor, behaviour_regressor])

    # Create Delta F Matrix
    vis_1_activity_tensor = np.vstack(vis_1_activity_tensor)
    vis_2_activity_tensor = np.vstack(vis_2_activity_tensor)
    print("visual_context_vis_1_activity_tensor", np.shape(vis_1_activity_tensor))

    delta_f_matrix = np.vstack([vis_1_activity_tensor, vis_2_activity_tensor])

    return design_matrix, delta_f_matrix