import os
import numpy as np
import pickle
import matplotlib.pyplot as plt


"""
Other Models
    1.) Previous timestep
    2.) Model no recurrent weights
    3.) Full Model
    4.) Model different recurrent each context
    5.) Model different recurrent each timestep
"""

def create_stimuli_regressors(Nstim, Ntrials, Nt):
    """
    Nstim = int, number of different stimuli
    Ntrials = list of length Nstim, number of trials for each stimulus
    Nt = int, number of timepoints in each trial
    """
    # Create Stimulus Regressor
    Q = np.zeros([Nstim, sum(Ntrials)])
    X = np.concatenate(([0], Ntrials.astype(int)))
    N = np.squeeze(np.cumsum(X))

    for s in range(Nstim):
        Q[s, N[s]:N[s + 1]] = 1
    stimblocks = np.kron(Q, np.eye(Nt))

    return stimblocks


def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]

        # Swap Axes To Fit Angus Convention
        activity_tensor = np.swapaxes(activity_tensor, 0, 1)

    return activity_tensor



def load_session_data(data_directory_root, session, mvar_directory_root, start_window, stop_window):

    # Load Activity Tensors
    vis_context_vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "visual_context_stable_vis_1"))
    vis_context_vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "visual_context_stable_vis_2"))
    odr_context_vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour_context_stable_vis_1"))
    odr_context_vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour_context_stable_vis_2"))
    odr_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "Odour_1"))
    odr_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "Odour_2"))

    # Load Behaviour Tensors
    vis_context_vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "visual_context_stable_vis_1"))
    vis_context_vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "visual_context_stable_vis_2"))
    odr_context_vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour_context_stable_vis_1"))
    odr_context_vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour_context_stable_vis_2"))
    odr_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "Odour_1"))
    odr_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "Odour_2"))

    # Get Time Data
    window = 1500
    frame_rate = np.load(os.path.join(data_directory_root, session, "Frame_Rate.npy"))
    timestep = (float(1) / frame_rate) * 1000
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
    delta_f_list =   [vis_context_vis_1_activity_tensor, vis_context_vis_2_activity_tensor, odr_context_vis_1_activity_tensor, odr_context_vis_2_activity_tensor, odr_1_activity_tensor, odr_2_activity_tensor]
    behaviour_list = [vis_context_vis_1_behaviour_tensor, vis_context_vis_2_behaviour_tensor, odr_context_vis_1_behaviour_tensor, odr_context_vis_2_behaviour_tensor, odr_1_behaviour_tensor, odr_2_behaviour_tensor]

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


def split_negshift_into_seperate_contexts(Ntrials, Nt, Nvar, dFtot_negshift):

    print("dFtot_negshift", np.shape(dFtot_negshift))

    # Count Number Of Trials
    total_trials = np.sum(Ntrials)
    vis_context_trials = np.sum(Ntrials[0:2])
    print("total_trials", total_trials)
    print("vis_context_trials", vis_context_trials)

    # Convert To Timepoints
    total_timepoints = total_trials * Nt
    visual_timepoints = vis_context_trials * Nt

    # Create Empty Matrix
    negshift_matrix = np.zeros((Nvar*2, total_timepoints))

    # Split Preceeding Activity
    negshift_matrix[0:Nvar, 0:visual_timepoints] = dFtot_negshift[:, 0:visual_timepoints]
    negshift_matrix[Nvar:, visual_timepoints:] = dFtot_negshift[:, visual_timepoints:]
    print("negshift_matrix", np.shape(negshift_matrix))

    return negshift_matrix



def create_regression_matrix(data_directory_root, session, mvar_directory_root, start_window, stop_window, model_type="Standard"):

    # Load Data
    Nvar, Nbehav, Nt, Nstim, Ntrials, delta_f_list, behaviour_list, timewindow = load_session_data(data_directory_root, session, mvar_directory_root, start_window, stop_window)
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
    #dFtot = np.subtract(dFtot, dFtot_negshift)

    # Create Behaviour Regressor
    behaviourtot = np.concatenate(behaviour_concat).T

    # Create Stimulus Regressor
    stimblocks = create_stimuli_regressors(Nstim, Ntrials, Nt)

    # Combine Regressors Into Design Matrix
    if model_type == "No_Recurrent":
        DesignMatrix = np.concatenate((stimblocks.T, behaviourtot.T), axis=1)  # design matrix

    elif model_type == "Standard":
        DesignMatrix = np.concatenate((dFtot_negshift.T, stimblocks.T, behaviourtot.T), axis=1)  # design matrix

    elif model_type == "Seperate_Contexts":
        dFtot_negshift = split_negshift_into_seperate_contexts(Ntrials, Nt, Nvar, dFtot_negshift)
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
        "timewindow": timewindow,
        "model_type": model_type,
    }

    # Create Save Directory
    save_directory = os.path.join(mvar_directory_root, session, "Design_Matricies")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save Regression Dict
    np.save(os.path.join(save_directory, model_type + "_Design_Matrix_Dict.npy"), regression_matrix_dictionary)

