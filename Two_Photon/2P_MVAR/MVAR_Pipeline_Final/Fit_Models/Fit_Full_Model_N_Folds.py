import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from scipy import stats
from sklearn.model_selection import KFold

# Custom Modules
import Model_Fitting_Utils
import Ridge_Model_Class


def get_best_ridge_penalties(output_directory):

    # Get Selection Of Potential Ridge Penalties
    penalty_possible_values = np.load(os.path.join(output_directory, "penalty_possible_values.npy"))

    # Load Visual Penalty Matrix
    penalty_matrix = np.load(os.path.join(output_directory, "Ridge_Penalty_Search_Results.npy"))
    best_score = np.max(penalty_matrix)
    score_indexes = np.where(penalty_matrix == best_score)

    stimuli_penalty_index = score_indexes[0]
    behaviour_penalty_index = score_indexes[1]
    interaction_penalty_index = score_indexes[2]

    stimuli_penalty_value = penalty_possible_values[stimuli_penalty_index][0]
    behaviour_penalty_value = penalty_possible_values[behaviour_penalty_index][0]
    interaction_penalty_value = penalty_possible_values[interaction_penalty_index][0]

    return stimuli_penalty_value, behaviour_penalty_value, interaction_penalty_value



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




"""
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

"""


def fit_full_model(mvar_directory_root, session, model_type):

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = Model_Fitting_Utils.load_design_matrix(session, mvar_directory_root, model_type)

    # Get Ridge Penalties
    stimuli_penalty, behaviour_penalty, interaction_penalty = get_best_ridge_penalties(os.path.join(mvar_directory_root, session, "Ridge_Penalty_Search", model_type))

    # Create Model
    model = Ridge_Model_Class.ridge_model(Nvar, Nstim, Nt, Nbehav, Ntrials, model_type, [stimuli_penalty, behaviour_penalty, interaction_penalty])

    # Fit Model
    mean_error, mean_weights = Model_Fitting_Utils.n_fold_cv(design_matrix, delta_f_matrix, model, n_folds=5)

    # Create Output Folder
    save_directory = os.path.join(mvar_directory_root, session, "Full_Model")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Save Outputs
    model_regression_dictionary = { "MVAR_Parameters": mean_weights,
                                    "Nvar": Nvar,
                                    "Nbehav": Nbehav,
                                    "Nt": Nt,
                                    "N_stim": Nstim,
                                    "Ntrials": Ntrials,
                                    "stimuli_penalty":stimuli_penalty,
                                    "behaviour_penalty":behaviour_penalty,
                                    "interaction_penalty":interaction_penalty,
                                    "Mean_Error": mean_error,
                                    }

    np.save(os.path.join(save_directory, model_type + "_Model_Dict.npy"), model_regression_dictionary)


