import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import r2_score

# Custom Modules
import Ridge_Model_Class






def plot_error_matrix(save_directory, context, error_matrix, number_of_penalties, penalty_possible_values):

    figure_1 = plt.figure(figsize=(35,8))
    error_min = np.min(error_matrix)
    error_max = np.max(error_matrix)

    axis_list = []
    for stimuli_penalty_index in range(number_of_penalties):

        axis_1 = figure_1.add_subplot(1, number_of_penalties + 1, stimuli_penalty_index + 1)
        axis_list.append(axis_1)
        axis_1.set_title("Stimuli Penalty: " + str(penalty_possible_values[stimuli_penalty_index]))
        im = axis_1.imshow(error_matrix[stimuli_penalty_index], cmap='viridis', vmin=error_min, vmax=error_max)


        axis_1.set_xticks(list(range(0, number_of_penalties)))
        axis_1.set_yticks(list(range(0, number_of_penalties)))
        axis_1.set_xticklabels(penalty_possible_values)
        axis_1.set_yticklabels(penalty_possible_values)

        axis_1.set_xlabel("Interaction_Penalty")

        # Loop over data dimensions and create text annotations.
        for i in range(number_of_penalties):
            for j in range(number_of_penalties):
                text = axis_1.text(j, i, np.around(error_matrix[stimuli_penalty_index, i, j] * 100, 2), ha="center", va="center", color="w")

    axis_list[0].set_ylabel("Behaviour Penalty")

    colourbar_axis = figure_1.add_subplot(1, number_of_penalties + 1, number_of_penalties + 1)
    colourbar_axis.axis('off')
    plt.colorbar(im, ax=colourbar_axis, fraction=0.05, location='left')
    plt.tight_layout()

    #plt.show()
    plt.savefig(os.path.join(save_directory, context + "_Parameter_Sweep.png"))
    plt.close()



def n_fold_cv(n_trials_list, trial_length, design_matrix, delta_f_matrix, model, n_folds=5):

    n_trials = np.sum(n_trials_list)
    trials_per_fold = int(n_trials / n_folds)
    fold_size = trials_per_fold * trial_length


    delta_f_matrix = np.transpose(delta_f_matrix)
    n_pixels = np.shape(delta_f_matrix)[1]

    error_list = []
    for fold_index in range(n_folds):
        fold_start = fold_index * fold_size
        fold_stop = fold_start + fold_size

        x_test = design_matrix[fold_start:fold_stop]
        y_test = delta_f_matrix[fold_start:fold_stop]

        x_train = np.delete(design_matrix, list(range(fold_start, fold_stop)), axis=0)
        y_train = np.delete(delta_f_matrix, list(range(fold_start, fold_stop)), axis=0)

        # Fit Model
        model.fit(x_train, y_train)

        # Predict Test Data
        y_pred = model.predict(x_test)

        # Get Score
        score = r2_score(y_true=y_test, y_pred=y_pred)
        error_list.append(score)

        """
        parameters = model.MVAR_parameters
        weight_matrix = parameters[:, 0:n_pixels]
        weight_magnitude = np.percentile(np.abs(weight_matrix), 99)
        plt.imshow(weight_matrix, vmin=-weight_magnitude, vmax=weight_magnitude, cmap="bwr")
        plt.show()
        """

        print("Score", score)

    # Get Mean Error Across All Folds
    mean_error = np.mean(error_list)

    return mean_error



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


def get_cv_ridge_penalties(session, mvar_output_directory, context):

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = load_regression_matrix(session, mvar_output_directory, context)
    print("Nvar", Nvar)
    print("Nbehav", Nbehav)
    print("Nt", Nt)
    print("Nstim", Nstim)
    print("Ntrials", Ntrials)

    # Get Selection Of Potential Ridge Penalties
    penalty_start = -1
    penalty_stop = 5
    number_of_penalties = (penalty_stop - penalty_start) + 1
    penalty_possible_values = np.logspace(start=penalty_start, stop=penalty_stop, base=10, num=number_of_penalties)
    print("Penalty Possible Values", penalty_possible_values)

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, session, "Ridge_Penalty_Search")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    # Create Empty Matrix To Hold Errors
    error_matrix = np.zeros((number_of_penalties, number_of_penalties, number_of_penalties))

    # Iterate Through Each Penalty
    for stimuli_penalty_index in range(number_of_penalties):
        for behaviour_penalty_index in range(number_of_penalties):
            for interaction_penalty_index in tqdm(range(number_of_penalties)):

                # Select Penalties
                stimuli_penalty = penalty_possible_values[stimuli_penalty_index]
                behaviour_penalty = penalty_possible_values[behaviour_penalty_index]
                interaction_penalty = penalty_possible_values[interaction_penalty_index]

                # Create Model
                print("pre class",)
                print("Nvar",Nvar)
                print("Nstim",Nstim)
                print("Nt",Nt)
                print("Nbehav",Nbehav)
                print("Ntrials",Ntrials)

                model = Ridge_Model_Class.ridge_model(Nvar, Nstim, Nt, Nbehav, Ntrials, interaction_penalty, stimuli_penalty, behaviour_penalty)

                # Perform 5 Fold Cross Validation
                mean_error = n_fold_cv(Ntrials, Nt, design_matrix, delta_f_matrix, model)

                # Add This To The Error Matrix
                error_matrix[stimuli_penalty_index, behaviour_penalty_index, interaction_penalty_index] = mean_error

                # Save The Weights
                model_name = "Stimuli_Penalty_" + str(stimuli_penalty_index).zfill(3) + "Interaction_Penalty_" + str(interaction_penalty_index).zfill(3) + "_Behaviour_Penalty_" + str(behaviour_penalty_index).zfill(3)

                # Draw Matrix
                plot_error_matrix(save_directory, context, error_matrix, number_of_penalties, penalty_possible_values)

                print("Stimuli Penalty: ", stimuli_penalty, "Behaviour Penalty", behaviour_penalty, "Interaction", interaction_penalty, "Score", mean_error, "at ", datetime.now())

    # Save This Matrix
    np.save(os.path.join(save_directory, context + "_Ridge_Penalty_Search_Results.npy"), error_matrix)
