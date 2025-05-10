import os
import numpy as np
from tqdm import tqdm
import sys
import pickle
import matplotlib.pyplot as plt

import MVAR_Utils_2P




def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor


def load_data_with_odours(data_directory_root, session, mvar_directory_root, start_window, stop_window):

    # Load Activity Tensors
    vis_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour_context_stable_vis_1"))
    vis_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "odour_context_stable_vis_2"))
    odr_1_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "Odour_1"))
    odr_2_activity_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Activity_Tensors", "Odour_2"))

    # Load Behaviour Tensors
    vis_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour_context_stable_vis_1"))
    vis_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "odour_context_stable_vis_2"))
    odr_1_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "Odour_1"))
    odr_2_behaviour_tensor = open_tensor(os.path.join(mvar_directory_root, session, "Behaviour_Tensors", "Odour_2"))

    print("vis_1_behaviour_tensor", np.shape(vis_1_behaviour_tensor))
    print("vis_2_behaviour_tensor", np.shape(vis_2_behaviour_tensor))

    # Swap Axes To Fit Angus Convention
    print("Vis 1 activity tensor shape", np.shape(vis_1_activity_tensor))
    vis_1_activity_tensor = np.swapaxes(vis_1_activity_tensor, 0, 1)
    vis_2_activity_tensor = np.swapaxes(vis_2_activity_tensor, 0, 1)
    odr_1_activity_tensor = np.swapaxes(odr_1_activity_tensor, 0, 1)
    odr_2_activity_tensor = np.swapaxes(odr_2_activity_tensor, 0, 1)

    vis_1_behaviour_tensor = np.swapaxes(vis_1_behaviour_tensor, 0, 1)
    vis_2_behaviour_tensor = np.swapaxes(vis_2_behaviour_tensor, 0, 1)
    odr_1_behaviour_tensor = np.swapaxes(odr_1_behaviour_tensor, 0, 1)
    odr_2_behaviour_tensor = np.swapaxes(odr_2_behaviour_tensor, 0, 1)

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
    delta_f_list = [vis_1_activity_tensor, vis_2_activity_tensor, odr_1_activity_tensor, odr_2_activity_tensor]
    behaviour_list = [vis_1_behaviour_tensor, vis_2_behaviour_tensor, odr_1_behaviour_tensor, odr_2_behaviour_tensor]

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



def create_regression_matricies_shared_recurrent_weights(data_directory_root, session, mvar_directory_root, start_window, stop_window):

    # Load Data
    Nvar, Nbehav, Nt, N_vis_stim, N_vis_trials, vis_delta_f_list, vis_behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, "visual", start_window, stop_window)
    Nvar, Nbehav, Nt, N_odr_stim, N_odr_trials, odr_delta_f_list, odr_behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, "odour", start_window, stop_window)
    print("N_vis_trials", N_vis_trials, "N_odr_trials", N_odr_trials)
    #print("vis_delta_f_list", len(vis_delta_f_list), "odr_delta_f_list", len(odr_delta_f_list))
    #print("vis_behaviour_list", np.shape(vis_behaviour_list), "odr_behaviour_list", np.shape(odr_behaviour_list))

    Nstim = N_vis_stim + N_odr_stim
    Ntrials = np.concatenate([N_vis_trials, N_odr_trials])
    delta_f_list = vis_delta_f_list + odr_delta_f_list
    behaviour_list = vis_behaviour_list + odr_behaviour_list

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

    print("dFtot", np.shape(dFtot))

    """
    plt.imshow(stimblocks)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

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
    np.save(os.path.join(save_directory,  "Combined_Design_Matrix_Dictionary.npy"), regression_matrix_dictionary)






def create_combined_regression_maticies(data_directory_root, session, mvar_directory_root, start_window, stop_window):

    # Load Data
    Nvar, Nbehav, Nt, N_vis_stim, N_vis_trials, vis_delta_f_list, vis_behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, "visual", start_window, stop_window)
    Nvar, Nbehav, Nt, N_odr_stim, N_odr_trials, odr_delta_f_list, odr_behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, "odour", start_window, stop_window)

    Nstim = N_vis_stim + N_odr_stim
    Ntrials = np.concatenate([N_vis_trials, N_odr_trials])
    behaviour_list = vis_behaviour_list + odr_behaviour_list
    delta_f_list = vis_delta_f_list + odr_delta_f_list

    print("N_vis_trials", N_vis_trials, "N_odr_trials", N_odr_trials)
    print("Nvar", Nvar)
    print("Nbehav", Nbehav)
    print("Nt", Nt)
    print("Nstim", Nstim)
    print("Ntrials", Ntrials)


    # Create Stimulus Regressors
    stimblocks = create_stimuli_regressors(Nstim, Ntrials, Nt)
    print("stimblocks", np.shape(stimblocks))

    """
    plt.imshow(stimblocks)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

    # Create Behaviour Regressors
    behaviour_concat = []
    for s in range(Nstim):
        behaviour_concat.append(np.squeeze(np.reshape(behaviour_list[s][timewindow, :, :], (Nt * Ntrials[s], Nbehav), order="F")))
    behaviourtot = np.concatenate(behaviour_concat).T
    print("behaviour_concat", np.shape(behaviourtot))

    """
    plt.imshow(behaviourtot)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

    # Create DF Matricies
    dFmat_concat = []
    dFmat_concat_negshift = []
    for s in range(Nstim):
        dFmat_concat.append(np.squeeze(np.reshape(delta_f_list[s][timewindow, :, :], (Nt * Ntrials[s], Nvar), order="F")))  # concatenate trials
        dFmat_concat_negshift.append(np.squeeze(np.reshape(delta_f_list[s][timewindow - 1, :, :], (Nt * Ntrials[s], Nvar), order="F")))
    dFtot = np.concatenate(dFmat_concat).T  # concatenate stimuli
    dFtot_negshift = np.concatenate(dFmat_concat_negshift).T
    dFtot = np.subtract(dFtot, dFtot_negshift)
    print("dFtot", np.shape(dFtot))
    print("dFtot_negshift", np.shape(dFtot_negshift))


    # Split dFtot_negshift By Context
    n_vis_timepoints = Nt * np.sum(N_vis_trials)
    split_negshift = np.zeros((Nvar*2, Nt * np.sum(Ntrials)))
    print("split_negshift", np.shape(split_negshift))

    split_negshift[0:Nvar, 0:n_vis_timepoints] = dFtot_negshift[:, 0:n_vis_timepoints]
    split_negshift[Nvar:, n_vis_timepoints:] = dFtot_negshift[:, n_vis_timepoints:]

    """
    plt.imshow(split_negshift)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()

    plt.imshow(dFtot)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

    # Combine Regressors Into Design Matrix
    DesignMatrix = np.concatenate((split_negshift.T, stimblocks.T, behaviourtot.T), axis=1)  # design matrix

    # Combine Into Dictionary
    regression_matrix_dictionary = {
        "DesignMatrix": DesignMatrix,
        "dFtot": dFtot,
        "Nvar": Nvar*2,
        "Nbehav": Nbehav,
        "Nt": Nt,
        "N_stim": Nstim,
        "N_trials": Ntrials,
        "timewindow": timewindow,
        "N_vis_trials":N_vis_trials,
        "N_odr_trials":N_odr_trials
    }

    # Create Save Directory
    save_directory = os.path.join(mvar_directory_root, session, "Design_Matricies")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save Regression Dict
    np.save(os.path.join(save_directory, "Combined_Design_Matrix_Dictionary.npy"), regression_matrix_dictionary)



def create_combined_regression_maticies_odours(data_directory_root, session, mvar_directory_root, start_window, stop_window):

    # Load Data
    Nvar, Nbehav, Nt, N_vis_stim, N_vis_trials, vis_delta_f_list, vis_behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, "visual", start_window, stop_window)
    Nvar, Nbehav, Nt, N_odr_stim, N_odr_trials, odr_delta_f_list, odr_behaviour_list, timewindow = load_data_with_odours(data_directory_root, session, mvar_directory_root, start_window, stop_window)

    Nstim = N_vis_stim + N_odr_stim
    Ntrials = np.concatenate([N_vis_trials, N_odr_trials])
    behaviour_list = vis_behaviour_list + odr_behaviour_list
    delta_f_list = vis_delta_f_list + odr_delta_f_list

    print("N_vis_trials", N_vis_trials, "N_odr_trials", N_odr_trials)
    print("Nvar", Nvar)
    print("Nbehav", Nbehav)
    print("Nt", Nt)
    print("Nstim", Nstim)
    print("Ntrials", Ntrials)


    # Create Stimulus Regressors
    stimblocks = create_stimuli_regressors(Nstim, Ntrials, Nt)
    print("stimblocks", np.shape(stimblocks))

    """
    plt.imshow(stimblocks)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

    # Create Behaviour Regressors
    behaviour_concat = []
    for s in range(Nstim):
        behaviour_concat.append(np.squeeze(np.reshape(behaviour_list[s][timewindow, :, :], (Nt * Ntrials[s], Nbehav), order="F")))
    behaviourtot = np.concatenate(behaviour_concat).T
    print("behaviour_concat", np.shape(behaviourtot))

    """
    plt.imshow(behaviourtot)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

    # Create DF Matricies
    dFmat_concat = []
    dFmat_concat_negshift = []
    for s in range(Nstim):
        dFmat_concat.append(np.squeeze(np.reshape(delta_f_list[s][timewindow, :, :], (Nt * Ntrials[s], Nvar), order="F")))  # concatenate trials
        dFmat_concat_negshift.append(np.squeeze(np.reshape(delta_f_list[s][timewindow - 1, :, :], (Nt * Ntrials[s], Nvar), order="F")))
    dFtot = np.concatenate(dFmat_concat).T  # concatenate stimuli
    dFtot_negshift = np.concatenate(dFmat_concat_negshift).T
    #dFtot = np.subtract(dFtot, dFtot_negshift)
    print("dFtot", np.shape(dFtot))
    print("dFtot_negshift", np.shape(dFtot_negshift))


    # Split dFtot_negshift By Context
    n_vis_timepoints = Nt * np.sum(N_vis_trials)
    split_negshift = np.zeros((Nvar*2, Nt * np.sum(Ntrials)))
    print("split_negshift", np.shape(split_negshift))

    split_negshift[0:Nvar, 0:n_vis_timepoints] = dFtot_negshift[:, 0:n_vis_timepoints]
    split_negshift[Nvar:, n_vis_timepoints:] = dFtot_negshift[:, n_vis_timepoints:]

    """
    plt.imshow(split_negshift)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()

    plt.imshow(dFtot)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """
    # Combine Regressors Into Design Matrix
    DesignMatrix = np.concatenate((split_negshift.T, stimblocks.T, behaviourtot.T), axis=1)  # design matrix

    # Combine Into Dictionary
    regression_matrix_dictionary = {
        "DesignMatrix": DesignMatrix,
        "dFtot": dFtot,
        "Nvar": Nvar*2,
        "Nbehav": Nbehav,
        "Nt": Nt,
        "N_stim": Nstim,
        "N_trials": Ntrials,
        "timewindow": timewindow,
        "N_vis_trials":N_vis_trials,
        "N_odr_trials":N_odr_trials
    }

    # Create Save Directory
    save_directory = os.path.join(mvar_directory_root, session, "Design_Matricies")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save Regression Dict
    np.save(os.path.join(save_directory, "Combined_Design_Matrix_Dictionary.npy"), regression_matrix_dictionary)




def create_regression_matricies_shared_recurrent_weights_odour(data_directory_root, session, mvar_directory_root, start_window, stop_window):

    # Load Data
    Nvar, Nbehav, Nt, N_vis_stim, N_vis_trials, vis_delta_f_list, vis_behaviour_list, timewindow = load_data(data_directory_root, session, mvar_directory_root, "visual", start_window, stop_window)
    Nvar, Nbehav, Nt, N_odr_stim, N_odr_trials, odr_delta_f_list, odr_behaviour_list, timewindow = load_data_with_odours(data_directory_root, session, mvar_directory_root, start_window, stop_window)


    print("N_vis_trials", N_vis_trials, "N_odr_trials", N_odr_trials)
    #print("vis_delta_f_list", len(vis_delta_f_list), "odr_delta_f_list", len(odr_delta_f_list))
    #print("vis_behaviour_list", np.shape(vis_behaviour_list), "odr_behaviour_list", np.shape(odr_behaviour_list))

    Nstim = N_vis_stim + N_odr_stim
    Ntrials = np.concatenate([N_vis_trials, N_odr_trials])
    delta_f_list = vis_delta_f_list + odr_delta_f_list
    behaviour_list = vis_behaviour_list + odr_behaviour_list

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
    Q = np.zeros([Nstim, sum(Ntrials)])
    X = np.concatenate(([0], Ntrials.astype(int)))
    N = np.squeeze(np.cumsum(X))
    for s in range(Nstim):
        Q[s, N[s]:N[s + 1]] = 1
    stimblocks = np.kron(Q, np.eye(Nt))

    print("dFtot", np.shape(dFtot))

    """
    plt.imshow(stimblocks)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

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
    np.save(os.path.join(save_directory,  "Combined_Design_Matrix_Dictionary.npy"), regression_matrix_dictionary)


