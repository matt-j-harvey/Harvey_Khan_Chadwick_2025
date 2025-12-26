import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from datetime import datetime

# Custom Modules
import Ridge_Model_Class
import Model_Fitting_Utils


def plot_error_matrix(save_directory, model_type, error_matrix, number_of_penalties, penalty_possible_values):

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

    plt.savefig(os.path.join(save_directory, model_type + "_Parameter_Sweep.png"))
    plt.show()
    plt.close()







def get_cv_ridge_penalties(session, mvar_output_directory, model_type):

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, session, "Ridge_Penalty_Search", model_type)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = Model_Fitting_Utils.load_design_matrix(session, mvar_output_directory, model_type)
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
    np.save(os.path.join(save_directory, "penalty_possible_values.npy"), penalty_possible_values)

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
                model = Ridge_Model_Class.ridge_model(Nvar, Nstim, Nt, Nbehav, Ntrials, model_type,[stimuli_penalty, behaviour_penalty, interaction_penalty])

                # Perform 5-Fold Cross Validation
                mean_error, mean_weights = Model_Fitting_Utils.n_fold_cv(design_matrix, delta_f_matrix, model)

                # Add This To The Error Matrix
                error_matrix[stimuli_penalty_index, behaviour_penalty_index, interaction_penalty_index] = mean_error
                #print("Stimuli Penalty: ", stimuli_penalty, "Behaviour Penalty", behaviour_penalty, "Interaction", interaction_penalty, "Score", mean_error, "at ", datetime.now())

    # Draw Matrix
    plot_error_matrix(save_directory, model_type, error_matrix, number_of_penalties, penalty_possible_values)

    # Save This Matrix
    np.save(os.path.join(save_directory, "Ridge_Penalty_Search_Results.npy"), error_matrix)

