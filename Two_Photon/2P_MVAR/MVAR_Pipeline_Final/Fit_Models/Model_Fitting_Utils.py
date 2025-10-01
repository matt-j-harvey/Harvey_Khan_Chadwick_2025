import numpy as np
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold


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




def n_fold_cv(design_matrix, delta_f_matrix, model, n_folds=5, return_parameters=True):

    # Transpose DF Matrix
    delta_f_matrix = np.transpose(delta_f_matrix)

    # Create K Fold Object
    k_fold_object = KFold(n_splits=n_folds, shuffle=True)

    # Test and Train each Fold
    error_list = []
    weights_list = []

    for i, (train_index, test_index) in enumerate(k_fold_object.split(design_matrix)):

        # Split Data Into Train and Test
        x_train = design_matrix[train_index]
        y_train = delta_f_matrix[train_index]

        x_test = design_matrix[test_index]
        y_test = delta_f_matrix[test_index]

        # Fit Model
        model.fit(x_train, y_train)

        # Predict Test Data
        y_pred = model.predict(x_test)

        # Get Score
        score = r2_score(y_true=y_test, y_pred=y_pred)
        error_list.append(score)

        # Save Parameters
        if return_parameters == True:
            model_parameters = model.MVAR_parameters
            weights_list.append(model_parameters)

    # Get Mean Error Across All Folds
    mean_error = np.mean(error_list)

    if return_parameters == True:
        mean_weights = np.mean(np.array(weights_list), axis=0)
    else:
        mean_weights = None

    return mean_error, mean_weights
