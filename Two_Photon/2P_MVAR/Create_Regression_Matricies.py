import os
import numpy as np
from tqdm import tqdm
import sys
import pickle

import MVAR_Utils_2P




def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor


def load_data(data_directory_root, session, mvar_directory_root, context, start_window, stop_window):

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

    # Swap Axes To Fit Angus Convention
    print("Vis 1 activity tensor shape", np.shape(vis_1_activity_tensor))
    vis_1_activity_tensor = np.swapaxes(vis_1_activity_tensor, 0, 1)
    vis_2_activity_tensor = np.swapaxes(vis_2_activity_tensor, 0, 1)
    vis_1_behaviour_tensor = np.swapaxes(vis_1_behaviour_tensor, 0, 1)
    vis_2_behaviour_tensor = np.swapaxes(vis_2_behaviour_tensor, 0, 1)

    # Get Time Data
    window = 1500
    frame_rate = np.load(os.path.join(data_directory_root, session, "Frame_Rate.npy"))
    timestep = (float(1)/frame_rate)*1000
    trial_start_point = np.abs(start_window)
    start = trial_start_point - int(window / timestep)
    stop = trial_start_point + int(window / timestep)
    timewindow = np.array(list(range(start, stop)))

    print("Start Window", start_window)
    print("Stop Window", stop_window)
    print("Window", window)
    print("frame_rate", frame_rate)
    print("timestep", timestep)
    print("start", start)
    print("stop", stop)
    print("Timewindow", timewindow)

    # Combine Stimuli
    delta_f_list = [vis_1_activity_tensor, vis_2_activity_tensor]
    behaviour_list = [vis_1_behaviour_tensor, vis_2_behaviour_tensor]

    # Get Data Details
    Nvar = delta_f_list[0].shape[2]  # number of variables (e.g., neurons, pixels, depending on dataset)
    Nbehav = behaviour_list[0].shape[2]
    Nt = len(timewindow)
    Nstim = len(delta_f_list)
    Ntrials = np.zeros(Nstim)
    for s in range(Nstim):
        Ntrials[s] = delta_f_list[s].shape[1]
    Ntrials = Ntrials.astype(int)

    return Nvar, Nbehav, Nt, Nstim, Ntrials, delta_f_list, behaviour_list, timewindow




def create_regression_matricies(data_directory_root, session, mvar_directory_root, context, start_window, stop_window):

    # Load Data
    Nvar, Nbehav, Nt, Nstim, Ntrials, delta_f_list, behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, context, start_window, stop_window)
    print("Nvar", Nvar)
    print("Nbehav", Nbehav)
    print("Nt", Nt)
    print("Nstim", Nstim)
    print("Ntrials", Ntrials)

    # Create Regression Matricies
    dFmat_concat = []
    dFmat_concat_negshift = []
    behaviour_concat = []
    for s in range(Nstim):
        dFmat_concat.append(np.squeeze(np.reshape(delta_f_list[s][timewindow, :, :], (Nt * Ntrials[s], Nvar), order="F")))  # concatenate trials
        dFmat_concat_negshift.append(np.squeeze(np.reshape(delta_f_list[s][timewindow - 1, :, :], (Nt * Ntrials[s], Nvar), order="F")))
        behaviour_concat.append(np.squeeze(np.reshape(behaviour_list[s][timewindow, :, :], (Nt * Ntrials[s], Nbehav), order="F")))

    dFtot = np.concatenate(dFmat_concat).T  # concatenate stimuli
    dFtot_negshift = np.concatenate(dFmat_concat_negshift).T
    dFtot = np.subtract(dFtot, dFtot_negshift)

    # Create Behaviour Regressor
    behaviourtot = np.concatenate(behaviour_concat).T

    # Create Stimulus Regressor
    Q = np.zeros([Nstim, sum(Ntrials)])
    X = np.concatenate(([0], Ntrials.astype(int)))
    N = np.squeeze(np.cumsum(X))
    for s in range(Nstim):
        Q[s, N[s]:N[s + 1]] = 1
    stimblocks = np.kron(Q, np.eye(Nt))


    # Combine Regressors Into Design Matrix
    DesignMatrix = np.concatenate((dFtot_negshift.T, stimblocks.T, behaviourtot.T), axis=1)  # design matrix

    # Combine Into Dictionary
    regression_matrix_dictionary = {
        "DesignMatrix": DesignMatrix,
        "dFtot": dFtot,
        "Nvar": Nvar,
        "Nbehav": Nbehav,
        "Nt": Nt,
        "N_stim": Nstim,
        "N_trials": Ntrials,
        "timewindow": timewindow
    }

    # Create Save Directory
    save_directory = os.path.join(mvar_directory_root, session, "Design_Matricies")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save Regression Dict
    np.save(os.path.join(save_directory, context + "_Design_Matrix_Dictionary.npy"), regression_matrix_dictionary)

