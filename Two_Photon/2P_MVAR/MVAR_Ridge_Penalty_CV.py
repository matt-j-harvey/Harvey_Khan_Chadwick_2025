import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

# Custom Modules
import MVAR_Utils_2P
import Ridge_Model_Class







def get_best_ridge_penalties(output_directory, context):

    # Get Selection Of Potential Ridge Penalties
    penalty_possible_values = np.load(os.path.join(output_directory, context + "_penalty_possible_values.npy"))

    # Load Visual Penalty Matrix
    penalty_matrix = np.load(os.path.join(output_directory, context + "_Ridge_Penalty_Search_Results.npy"))
    best_score = np.max(penalty_matrix)
    score_indexes = np.where(penalty_matrix == best_score)

    stimuli_penalty_index = score_indexes[0]
    behaviour_penalty_index = score_indexes[1]
    interaction_penalty_index = score_indexes[2]

    stimuli_penalty_value = penalty_possible_values[stimuli_penalty_index][0]
    behaviour_penalty_value = penalty_possible_values[behaviour_penalty_index][0]
    interaction_penalty_value = penalty_possible_values[interaction_penalty_index][0]

    return stimuli_penalty_value, behaviour_penalty_value, interaction_penalty_value


def create_ridge_penalty_dictionary(save_directory_root, context):

    stimuli_penalty, behaviour_penalty, interaction_penalty = get_best_ridge_penalties(save_directory_root, context)
    print("stimuli_penalty", stimuli_penalty)
    print("behaviour_penalty", behaviour_penalty)
    print("interaction_penalty", interaction_penalty)

    ridge_penalty_dict = {
        "stimuli_penalty": stimuli_penalty,
        "behaviour_penalty": behaviour_penalty,
        "interaction_penalty": interaction_penalty,
    }

    np.save(os.path.join(save_directory_root, context + "_ridge_penalty_dict.npy"), ridge_penalty_dict)



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



def n_fold_cv(design_matrix, delta_f_matrix, model, n_folds=5):

    # Transpose DF Matrix
    delta_f_matrix = np.transpose(delta_f_matrix)

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

        """
        parameters = model.MVAR_parameters
        weight_matrix = parameters[:, 0:n_pixels]
        weight_magnitude = np.percentile(np.abs(weight_matrix), 99)
        plt.imshow(weight_matrix, vmin=-weight_magnitude, vmax=weight_magnitude, cmap="bwr")
        plt.show()
        """

    # Get Mean Error Across All Folds
    mean_error = np.mean(error_list)

    return mean_error




def get_cv_ridge_penalties(session, mvar_output_directory, context):

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, session, "Ridge_Penalty_Search")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, mvar_output_directory, context)

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
    np.save(os.path.join(save_directory, context + "_penalty_possible_values.npy"), penalty_possible_values)
    print("Penalty Possible Values", penalty_possible_values)

    # Create Empty Matrix To Hold Errors
    error_matrix = np.zeros((number_of_penalties, number_of_penalties, number_of_penalties))

    # Iterate Through Each Penalty
    for stimuli_penalty_index in tqdm(range(number_of_penalties)):
        for behaviour_penalty_index in range(number_of_penalties):
            for interaction_penalty_index in (range(number_of_penalties)):

                # Select Penalties
                stimuli_penalty = penalty_possible_values[stimuli_penalty_index]
                behaviour_penalty = penalty_possible_values[behaviour_penalty_index]
                interaction_penalty = penalty_possible_values[interaction_penalty_index]

                # Create Model
                """
                print("pre class",)
                print("Nvar",Nvar)
                print("Nstim",Nstim)
                print("Nt",Nt)
                print("Nbehav",Nbehav)
                print("Ntrials",Ntrials)
                """

                model = Ridge_Model_Class.ridge_model(Nvar, Nstim, Nt, Nbehav, Ntrials, interaction_penalty, stimuli_penalty, behaviour_penalty)

                # Perform 5 Fold Cross Validation
                mean_error = n_fold_cv(design_matrix, delta_f_matrix, model)

                # Add This To The Error Matrix
                error_matrix[stimuli_penalty_index, behaviour_penalty_index, interaction_penalty_index] = mean_error

        # Draw Matrix
        plot_error_matrix(save_directory, context, error_matrix, number_of_penalties, penalty_possible_values)
        print("Stimuli Penalty: ", stimuli_penalty, "Behaviour Penalty", behaviour_penalty, "Interaction", interaction_penalty, "Score", mean_error, "at ", datetime.now())

    # Save This Matrix
    np.save(os.path.join(save_directory, context + "_Ridge_Penalty_Search_Results.npy"), error_matrix)

    # Create
    create_ridge_penalty_dictionary(save_directory, context)