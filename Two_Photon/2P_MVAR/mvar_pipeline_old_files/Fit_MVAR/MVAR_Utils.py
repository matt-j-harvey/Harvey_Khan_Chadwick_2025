import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pickle



def load_regression_matrix(session, mvar_output_directory, context):

    regression_matrix = np.load(os.path.join(mvar_output_directory,session, "Design_Matricies", context + "_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]

    DesignMatrix = regression_matrix["DesignMatrix"]
    dFtot = regression_matrix["dFtot"]
    Nvar = regression_matrix["Nvar"]
    Nbehav = regression_matrix["Nbehav"]
    Nt = regression_matrix["Nt"]
    Nstim = regression_matrix["N_stim"]
    Ntrials = regression_matrix["N_trials"]
    timewindow = regression_matrix["timewindow"]


    return DesignMatrix, dFtot, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow

