import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold



def n_fold_cv(design_matrix, delta_f_matrix, model, n_folds=5):

    delta_f_matrix = np.nan_to_num(delta_f_matrix)
    design_matrix = np.nan_to_num(design_matrix)

    # Create K Fold Object
    k_fold_object = KFold(n_splits=n_folds, shuffle=True)

    # Test and Train each Fold
    error_list = []

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

    # Get Mean Error Across All Folds
    mean_error = np.mean(error_list)

    return mean_error



def plot_score_matrix(save_directory, score_matrix, penalty_possible_values, mousecam_possible_values):

    figure_1 = plt.figure(figsize=(10,10))
    score_min = np.min(score_matrix)
    score_max = np.max(score_matrix)

    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.set_title("Parameter Search")
    im = axis_1.imshow(score_matrix, cmap='viridis', vmin=score_min, vmax=score_max)

    n_ridge_penalties = len(penalty_possible_values)
    n_mousecam_components = len(mousecam_possible_values)

    axis_1.set_xticks(list(range(0, n_mousecam_components)))
    axis_1.set_yticks(list(range(0, n_ridge_penalties)))
    axis_1.set_xticklabels(mousecam_possible_values)
    axis_1.set_yticklabels(penalty_possible_values)


    axis_1.set_xlabel("N Mousecam Components")
    axis_1.set_ylabel("Ridge Penalty")


    # Loop over data dimensions and create text annotations.
    for i in range(n_ridge_penalties):
        for j in range(n_mousecam_components):
            text = axis_1.text(j, i, np.around(score_matrix[i, j] * 100, 2), ha="center", va="center", color="white")


    plt.colorbar(im, fraction=0.05, location='left')
    plt.tight_layout()

    plt.savefig(os.path.join(save_directory,"Parameter_Sweep.png"))
    plt.close()





def parameter_search(output_root, session, design_matrix, delta_f_matrix, max_mousecam_components):

    # Create Save Directory
    save_directory = os.path.join(output_root, session, "Parameter_Search")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Get Selection Of Potential Ridge Penalties
    penalty_start = -4
    penalty_stop = 2
    number_of_ridge_penalties = (penalty_stop - penalty_start) + 1
    ridge_penalty_possible_values = np.logspace(start=penalty_start, stop=penalty_stop, base=10, num=number_of_ridge_penalties)
    np.save(os.path.join(save_directory, "ridge_penalty_possible_values.npy"), ridge_penalty_possible_values)

    # Get Mousecam Component Structure
    n_predictors = np.shape(design_matrix)[1]
    mousecam_component_possible_values = list(range(0, max_mousecam_components+1, 50))
    np.save(os.path.join(save_directory, "Mousecam_component_possible_values.npy"), mousecam_component_possible_values)

    n_mousecam_components = len(mousecam_component_possible_values)

    # Create Empty Matrix To Hold Errors
    error_matrix = np.zeros((number_of_ridge_penalties, n_mousecam_components))

    # Iterate Through Each Penalty
    for ridge_penalty_index in range(number_of_ridge_penalties):
        for mousecam_component_index in range(n_mousecam_components):

                # Select Ridge Penalty
                ridge_penalty = ridge_penalty_possible_values[ridge_penalty_index]
                mousecam_component_stop = (n_predictors - max_mousecam_components) + mousecam_component_possible_values[mousecam_component_index]

                # Create Model
                model = Ridge(alpha=ridge_penalty)

                # Create Design Matrix With Selected Number Of Mousecam Components
                truncated_design_matrix = design_matrix[:, 0:mousecam_component_stop]

                # Perform 5-Fold Cross Validation
                mean_error = n_fold_cv(truncated_design_matrix, delta_f_matrix, model)

                # Add This To The Error Matrix
                error_matrix[ridge_penalty_index, mousecam_component_index] = mean_error

    # Draw Matrix
    plot_score_matrix(save_directory, error_matrix, ridge_penalty_possible_values, mousecam_component_possible_values)

    # Save This Matrix
    np.save(os.path.join(save_directory, "Ridge_Penalty_Search_Results.npy"), error_matrix)

