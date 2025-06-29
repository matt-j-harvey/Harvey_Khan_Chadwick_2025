


"""
1 - Previous Step Only
2 - Just stimuli and behaviour regressors
3 - With recurrent weights
4 - Different recurrent weights for each context
5 - Time varying recurrent weights
"""




def create_regression_matrix_full_model(data_directory_root, session, mvar_directory_root, start_window, stop_window):

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


