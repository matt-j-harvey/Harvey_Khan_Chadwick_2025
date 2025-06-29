import numpy as np
import os
import matplotlib.pyplot as plt


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_time_x_values(timewindow, frame_rate):

    n_timepoints = len(timewindow)
    start_window = int(n_timepoints / 2)
    x_values = list(range(-start_window, start_window))

    period = 1.0 / frame_rate
    x_values = np.multiply(x_values, period)

    return x_values



def extract_stim_weights(model_dict):

    Nt = model_dict["Nt"]
    model_params = model_dict["MVAR_Parameters"]
    n_neurons = np.shape(model_params)[0]

    stim_weights_list = []
    for x in range(6):
        regressor_start = n_neurons + (x * Nt)
        regressor_stop = regressor_start + Nt
        stim_weights = model_params[:, regressor_start:regressor_stop]
        stim_weights_list.append(stim_weights)

    return stim_weights_list


def load_design_matrix(session, mvar_output_directory, model_type):

    design_matrix = np.load(os.path.join(mvar_output_directory,session, "Design_Matricies", model_type + "_Design_Matrix_Dict.npy"), allow_pickle=True)[()]

    DesignMatrix = design_matrix["DesignMatrix"]
    dFtot = design_matrix["dFtot"]
    Nvar = design_matrix["Nvar"]
    Nbehav = design_matrix["Nbehav"]
    Nt = design_matrix["Nt"]
    Nstim = design_matrix["N_stim"]
    Ntrials = design_matrix["N_trials"]
    timewindow = design_matrix["timewindow"]


    return DesignMatrix, dFtot, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow




