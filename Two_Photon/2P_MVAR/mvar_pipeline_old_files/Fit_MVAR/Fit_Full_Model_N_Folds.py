import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy import stats
from sklearn.model_selection import KFold

# Custom Modules
import MVAR_Utils
import Ridge_Model_Class


def n_fold_fit(model, design_matrix, delta_f_matrix):

    k_fold_object = KFold(n_splits=5)

    # Create Empty Lists
    weights_list = []
    for i, (train_index, test_index) in enumerate(k_fold_object.split(design_matrix)):
        x_train = design_matrix[train_index]
        y_train = delta_f_matrix[train_index]

        # Fit Model
        model.fit(x_train, y_train)

        # Save Parameters
        model_parameters = model.MVAR_parameters
        weights_list.append(model_parameters)

    mean_weights = np.mean(weights_list, axis=0)
    return mean_weights





def sanity_check_model_performance(design_matrix, mvar_parameters, delta_f_matrix):

    print("final design matrix shape", np.shape(design_matrix.T))
    print("final mvar parameters shape", np.shape(mvar_parameters))
    prediction = np.matmul(mvar_parameters, design_matrix.T)
    prediction = np.transpose(prediction)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(2,1,1)
    axis_2 = figure_1.add_subplot(2,1,2)

    axis_1.imshow(np.transpose(prediction))
    axis_2.imshow(np.transpose(delta_f_matrix))

    MVAR_Utils.forceAspect(axis_1)
    MVAR_Utils.forceAspect(axis_2)

    plt.show()




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
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils.load_regression_matrix(session, mvar_directory_root, context)
    print("design_matrix", np.shape(design_matrix))
    print("delta_f_matrix", np.shape(delta_f_matrix))

    print("Timewindow", len(timewindow))
    delta_f_matrix = np.transpose(delta_f_matrix)

    # Create Model
    model = Ridge_Model_Class.ridge_model(Nvar, Nstim, Nt, Nbehav, Ntrials, interaction_penalty, stimuli_penalty, behaviour_penalty)

    # Fit Model
    mean_weights = n_fold_fit(model, design_matrix, delta_f_matrix)

    """
    plt.title("Mean Weights")
    plt.imshow(mean_weights)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """
    #sanity_check_model_performance(design_matrix, mean_weights, delta_f_matrix)

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


