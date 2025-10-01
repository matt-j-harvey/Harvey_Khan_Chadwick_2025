import os
import numpy as np
from tqdm import tqdm
import Model_Fitting_Utils
from sklearn.linear_model import Ridge

def fit_model_per_neuron(design_matrix, delta_f_matrix, Nvar, penalty):

    # Split Design Matric
    common_design_matrix = design_matrix[:, Nvar:]
    df_negshift = design_matrix[:, 0:Nvar]

    r2_list = []
    for neuron_index in range(Nvar):

        # Create Neuron Specific Design Matrix
        neuron_df_negshift = np.expand_dims(df_negshift[:, neuron_index],1 )
        neuron_design_matrix = np.hstack([neuron_df_negshift, common_design_matrix])

        # Get Neuron DF
        neuron_delta_f_matrix = delta_f_matrix[neuron_index]

        # Create Model
        model = Ridge(alpha=penalty)

        # Fit Model
        neuron_r2, neuron_weights = Model_Fitting_Utils.n_fold_cv(neuron_design_matrix, neuron_delta_f_matrix, model, n_folds=5, return_parameters=False)
        r2_list.append(neuron_r2)

    mean_r2 = np.mean(r2_list)
    print("mean r2", mean_r2)
    return mean_r2



def fit_diagonal_with_stimuli(mvar_output_directory, session):

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, session, "Ridge_Penalty_Search", "Diagonal_with_Stim")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Get Selection Of Potential Ridge Penalties
    penalty_start = -1
    penalty_stop = 5
    number_of_penalties = (penalty_stop - penalty_start) + 1
    penalty_possible_values = np.logspace(start=penalty_start, stop=penalty_stop, base=10, num=number_of_penalties)
    np.save(os.path.join(save_directory, "penalty_possible_values.npy"), penalty_possible_values)

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = Model_Fitting_Utils.load_design_matrix(session, mvar_output_directory, "Standard")
    print("Nvar", Nvar)
    print("design_matrix", np.shape(design_matrix))
    print("delta_f_matrix", np.shape(delta_f_matrix))

    # Iterate Through Each Penalty
    r2_list = []
    for stimuli_penalty_index in tqdm(range(number_of_penalties)):
        penalty = penalty_possible_values[stimuli_penalty_index]
        penalty_r2 = fit_model_per_neuron(design_matrix, delta_f_matrix, Nvar, penalty)
        r2_list.append(penalty_r2)

    np.save(os.path.join(save_directory, "penalty_possible_values.npy"), penalty_possible_values)
    np.save(os.path.join(save_directory, "Ridge_Penalty_Search_Results.npy"), r2_list)