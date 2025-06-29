import numpy as np
import os
import matplotlib.pyplot as plt


def baseline_correct_tensor(tensor, baseline_window):

    baseline_corrected_tensor = []

    n_trials = np.shape(tensor)[0]
    for trial_index in range(n_trials):
        trial_data = tensor[trial_index]
        trial_baseline = trial_data[baseline_window[0]:baseline_window[1]]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial_data = np.subtract(trial_data, trial_baseline)
        baseline_corrected_tensor.append(trial_data)

    baseline_corrected_tensor = np.array(baseline_corrected_tensor)
    return baseline_corrected_tensor


def get_contribution(design_matrix, model_params, Nstim, Ntrials, Nt, Nvar, baseline_correction, baseline):

    # Get Prediction
    prediction = np.matmul(design_matrix, np.transpose(model_params))
    print("prediction", np.shape(prediction))

    # Split Prediction by trial type
    prediction_list = []

    stim_start = 0
    for stim_index in range(Nstim):
        stim_trials = Ntrials[stim_index]
        stim_size = stim_trials * Nt
        stim_stop = stim_start + stim_size

        stim_prediction = prediction[stim_start:stim_stop]
        stim_prediction = np.reshape(stim_prediction, (stim_trials, Nt, Nvar))

        if baseline_correction == True:
            stim_prediction = baseline_correct_tensor(stim_prediction, baseline)

        stim_prediction = np.mean(stim_prediction, axis=0)

        prediction_list.append(stim_prediction)

        stim_start = stim_stop

    return prediction_list


def get_lick_cd_projection(tensor, lick_cd):
    projection_tensor = []
    for trial in tensor:
        trial_projection = np.dot(trial, lick_cd)
        projection_tensor.append(trial_projection)
    projection_tensor = np.array(projection_tensor)
    projection_tensor = np.squeeze(projection_tensor)
    return projection_tensor



def partition_model_contributions(data_directory, session, design_matrix, Nvar, Nt, Nstim, Ntrials, model_dict, output_directory, baseline_correction=True, baseline=[0,5]):

    # Load Model Params
    model_params = model_dict['MVAR_Parameters']
    print("model_params", np.shape(model_params))

    # Create Partial Design Matricies
    recurrent_only_params = np.zeros(np.shape(model_params))
    stim_only_params = np.zeros(np.shape(model_params))

    recurrent_weights = model_params[:, 0:Nvar]
    np.fill_diagonal(recurrent_weights, 0)
    recurrent_only_params[:, 0:Nvar] = model_params[:, 0:Nvar]
    stim_only_params[:, Nvar:Nvar + (Nstim * Nt)] = model_params[:, Nvar:Nvar + (Nstim * Nt)]

    # Get Partial Contributions
    recurrent_contribution = get_contribution(design_matrix, recurrent_only_params, Nstim, Ntrials, Nt, Nvar, baseline_correction, baseline)
    stimulus_contribution = get_contribution(design_matrix, stim_only_params, Nstim, Ntrials, Nt, Nvar, baseline_correction, baseline)
    print("recurrent contribution", np.shape(recurrent_contribution))

    # Load Lick CD
    lick_cd = np.load(os.path.join(data_directory, session, "Coding_Dimensions", "Lick_CD.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)
    print("Lick CD", np.shape(lick_cd))

    # Get Partial Contribution Lick CDs
    recurrent_contribution_lick_cd = get_lick_cd_projection(recurrent_contribution, lick_cd)
    stimulus_contribution_lick_cd = get_lick_cd_projection(stimulus_contribution, lick_cd)
    print("recurrent_contribution_lick_cd", np.shape(recurrent_contribution_lick_cd))
    print("stimulus_contribution_lick_cd", np.shape(stimulus_contribution_lick_cd))

    # Save These
    save_directory = os.path.join(output_directory, "Partitioned_Contributions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    np.save(os.path.join(save_directory, "recurrent_contribution.npy"), recurrent_contribution)
    np.save(os.path.join(save_directory, "stimulus_contribution.npy"), stimulus_contribution)
    np.save(os.path.join(save_directory, "recurrent_contribution_lick_cd.npy"), recurrent_contribution_lick_cd)
    np.save(os.path.join(save_directory, "stimulus_contribution_lick_cd.npy"), stimulus_contribution_lick_cd)