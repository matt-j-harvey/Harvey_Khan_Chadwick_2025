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

import Opto_GLM_Utils




def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor



def create_stimuli_regressor(n_trials, n_timepoints):

    """
                            Trial   Attention   Light   Attention x Light
    Vis Context Control       1         1         0             0
    Vis Context Light         1         1         1             1
    Odr Context Control       1         0         0             0
    Odr Context Light         1         0         1             0
    """

    # Create Empty Regressor Matrix
    n_total_trials = np.sum(n_trials)
    n_stim_regressors = 4
    stimuli_regressor_matrix = np.zeros((n_total_trials * n_timepoints, n_timepoints * n_stim_regressors))
    print("stimuli regressor matrix", np.shape(stimuli_regressor_matrix))
    print("n_timepoints", n_timepoints)

    # First Do Visual Context Control - They have trial and attention
    trial_start = 0
    for trial_index in range(n_trials[0]):
        trial_stop = trial_start + n_timepoints
        #print("trial_start", trial_start, "trial_stop", trial_stop)
        stimuli_regressor_matrix[trial_start:trial_stop, 0 * n_timepoints:1 * n_timepoints] = np.eye(n_timepoints)
        stimuli_regressor_matrix[trial_start:trial_stop, 1 * n_timepoints:2 * n_timepoints] = np.eye(n_timepoints)
        trial_start = trial_stop

    # Next Do Visual Context Light - They have everything
    for trial_index in range(n_trials[1]):
        trial_stop = trial_start + n_timepoints
        stimuli_regressor_matrix[trial_start:trial_stop, 0 * n_timepoints:1 * n_timepoints] = np.eye(n_timepoints)
        stimuli_regressor_matrix[trial_start:trial_stop, 1 * n_timepoints:2 * n_timepoints] = np.eye(n_timepoints)
        stimuli_regressor_matrix[trial_start:trial_stop, 2 * n_timepoints:3 * n_timepoints] = np.eye(n_timepoints)
        stimuli_regressor_matrix[trial_start:trial_stop, 3 * n_timepoints:4 * n_timepoints] = np.eye(n_timepoints)
        trial_start = trial_stop

    # Next Do Odour Context Control - They have Only Trial
    for trial_index in range(n_trials[2]):
        trial_stop = trial_start + n_timepoints
        stimuli_regressor_matrix[trial_start:trial_stop, 0 * n_timepoints:1 * n_timepoints] = np.eye(n_timepoints)
        trial_start = trial_stop

    # Finally Do Odour Context Light - They have Trial and Light
    for trial_index in range(n_trials[3]):
        trial_stop = trial_start + n_timepoints
        stimuli_regressor_matrix[trial_start:trial_stop, 0 * n_timepoints:1 * n_timepoints] = np.eye(n_timepoints)
        stimuli_regressor_matrix[trial_start:trial_stop, 2 * n_timepoints:3 * n_timepoints] = np.eye(n_timepoints)
        trial_start = trial_stop

    #plt.imshow(np.transpose(stimuli_regressor_matrix), cmap="Greys", vmax=0.5)
    #Opto_GLM_Utils.forceAspect(plt.gca())
    #plt.show()
    return stimuli_regressor_matrix




def baseline_correct_tensor(tensor, baseline_start=0, baseline_stop=14):

    baseline_corrected_tensor = []
    for trial in tensor:
        trial_baseline = trial[baseline_start:baseline_stop]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        baseline_corrected_tensor.append(trial)

    return baseline_corrected_tensor




def create_regression_matricies(data_directory, session, mvar_directory_root, z_score, baseline_correct):

    """
    Regressors
    0 = Trial
    1 = Attention
    2 = light
    3 = Attention x Light

                            Trial   Attention   Light   Attention x Light
    Vis Context Control       1         1         0             0
    Odr Context Control       1         0         0             0
    Vis Context Light         1         1         1             1
    Odr Context Light         1         0         1             0

    """

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

    # Combine Regressors Into Design Matrix
    design_matrix = np.hstack([stimulus_regressor, behaviour_regressor])

    # Create Delta F Matrix
    visual_context_control_activity_tensor = np.vstack(visual_context_control_activity_tensor)
    visual_context_light_activity_tensor = np.vstack(visual_context_light_activity_tensor)
    odour_context_control_activity_tensor = np.vstack(odour_context_control_activity_tensor)
    odour_context_light_activity_tensor = np.vstack(odour_context_light_activity_tensor)
    delta_f_matrix = np.vstack([visual_context_control_activity_tensor,
                                visual_context_light_activity_tensor,
                                odour_context_control_activity_tensor,
                                odour_context_light_activity_tensor])

    return design_matrix, delta_f_matrix