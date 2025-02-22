import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy import stats
from sklearn.model_selection import KFold

# Custom Modules
import MVAR_Utils_2P
import Ridge_Model_Class
import Create_Regression_Matricies


def n_fold_fit(model, design_matrix, delta_f_matrix):

    k_fold_object = KFold(n_splits=5)

    # Create Empty Lists
    weights_list = []
    for i, (train_index, test_index) in enumerate(k_fold_object.split(design_matrix)):
        x_train = design_matrix[train_index]
        y_train = delta_f_matrix[train_index]

        print("x_train", np.shape(x_train))
        print("y_train", np.shape(y_train))

        # Fit Model
        model.fit(x_train, y_train)

        # Save Parameters
        model_parameters = model.MVAR_parameters
        weights_list.append(model_parameters)

    mean_weights = np.mean(weights_list, axis=0)
    return mean_weights





def fit_full_model(mvar_directory_root, session, context):

    # Create Output Folder
    save_directory = os.path.join(mvar_directory_root, session, "Full_Model")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Get Ridge Penalties
    ridge_penalty_dict = np.load(os.path.join(mvar_directory_root, session, "Ridge_Penalty_Search", context + "_ridge_penalty_dict.npy"), allow_pickle=True)[()]
    stimuli_penalty = ridge_penalty_dict["stimuli_penalty"]
    behaviour_penalty = ridge_penalty_dict["behaviour_penalty"]
    interaction_penalty = ridge_penalty_dict["interaction_penalty"]

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, mvar_directory_root, context)
    print("Timewindow", len(timewindow))
    delta_f_matrix = np.transpose(delta_f_matrix)
    print("design_matrix", np.shape(design_matrix))
    print("delta_f_matrix", np.shape(delta_f_matrix))

    # Create Model
    model = Ridge_Model_Class.ridge_model(Nvar, Nstim, Nt, Nbehav, Ntrials, interaction_penalty, stimuli_penalty, behaviour_penalty)

    # Fit Model
    mean_weights = n_fold_fit(model, design_matrix, delta_f_matrix)

    # Save Outputs
    model_regression_dictionary = { "MVAR_Parameters": mean_weights,
                                    "Nvar": Nvar,
                                    "Nbehav": Nbehav,
                                    "Nt": Nt,
                                    "N_stim": Nstim,
                                    "Ntrials": Ntrials,
                                    }

    print("Save Directory", save_directory)
    np.save(os.path.join(save_directory, context + "_Model_Dict.npy"), model_regression_dictionary)


