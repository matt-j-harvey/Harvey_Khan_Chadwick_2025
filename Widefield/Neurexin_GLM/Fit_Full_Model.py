import os
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Ridge, LinearRegression




def load_best_parameters(glm_output_directory, session, design_matrix, max_mousecam_components):

    # Load Score Matrix
    score_matrix = np.load(os.path.join(glm_output_directory, session, "Parameter_Search", "Ridge_Penalty_Search_Results.npy"))

    # Get Max Indicies
    max_score = np.max(score_matrix)

    max_indicies = np.where(score_matrix == max_score)
    ridge_index = max_indicies[0][0]
    mousecam_index = max_indicies[1][0]

    # Load Possible Values
    ridge_penalty_possible_values = np.load(os.path.join(glm_output_directory, session, "Parameter_Search", "ridge_penalty_possible_values.npy"))
    mousecam_component_possible_values = np.load(os.path.join(glm_output_directory, session, "Parameter_Search", "Mousecam_component_possible_values.npy"))

    # Get Selected Values
    best_ridge_penalty = ridge_penalty_possible_values[ridge_index]
    best_mousecam_n = mousecam_component_possible_values[mousecam_index]

    # Get Design Matrix With Selected Number of Mousecam Component
    n_regressors = np.shape(design_matrix)[1]
    mousecam_start = n_regressors - max_mousecam_components
    mousecam_stop = mousecam_start + best_mousecam_n

    design_matrix = design_matrix[:, 0:mousecam_stop]#

    return best_ridge_penalty, design_matrix



def fit_full_model(session, glm_output_directory, design_matrix, delta_f_matrix, max_mousecam_components):

    # Load Best Parameters
    ridge_penalty, design_matrix = load_best_parameters(glm_output_directory, session, design_matrix, max_mousecam_components)

    # Run Regression
    model = Ridge(alpha=ridge_penalty, fit_intercept=True)
    model.fit(X=design_matrix, y=delta_f_matrix)

    # Extract Coefs
    model_coefs = model.coef_
    model_intercept = model.intercept_
    print("Model Coefs", np.shape(model_coefs))

    #sanity_check_coefs(model_coefs, start_window, stop_window)

    # Save Coefs
    save_directory = os.path.join(glm_output_directory, session, "Model_Output")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Model_Coefs.npy"), model_coefs)
    np.save(os.path.join(save_directory, "model_intercept.npy"), model_intercept)
