import os
import numpy as np
import matplotlib.pyplot as plt


import Visualise_Results_Utils

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



def extract_model_predictions(data_directory, session, design_matrix, delta_f_matrix, Nvar, Nt, Nstim, Ntrials, model_dict, stimulus_list, save_directory, basline_correction=True, baseline=[0,5]):

    # Load Model Params
    model_params = model_dict['MVAR_Parameters']
    print("model_params", np.shape(model_params))

    # Get Prediction
    prediction = np.matmul(design_matrix, np.transpose(model_params))
    print("prediction", np.shape(prediction))

    # Split Prediction by trial type
    mean_prediction_list = []
    mean_real_list = []

    delta_f_matrix = np.transpose(delta_f_matrix)

    stim_start = 0
    for stim_index in range(Nstim):
        stim_trials = Ntrials[stim_index]
        stim_size = stim_trials * Nt
        stim_stop = stim_start + stim_size

        stim_prediction = prediction[stim_start:stim_stop]
        stim_prediction = np.reshape(stim_prediction, (stim_trials, Nt, Nvar))

        stim_real = delta_f_matrix[stim_start:stim_stop]
        stim_real = np.reshape(stim_real, (stim_trials, Nt, Nvar))

        if basline_correction == True:
            stim_prediction = baseline_correct_tensor(stim_prediction, baseline)
            stim_real = baseline_correct_tensor(stim_real, baseline)

        stim_prediction = np.mean(stim_prediction, axis=0)
        stim_real = np.mean(stim_real, axis=0)

        mean_prediction_list.append(stim_prediction)
        mean_real_list.append(stim_real)

        stim_start = stim_stop

    # Get Stimuli Lick CD Projections

    # Load Lick CD
    lick_cd = np.load(os.path.join(data_directory, session, "Coding_Dimensions", "Lick_CD.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)
    print("Lick CD", np.shape(lick_cd))

    # Project Each Stimulus
    real_lick_cd_list = []
    predicted_lick_cd_list = []

    for stim_index in range(Nstim):

        print("activity")
        real_lick_cd_projection = np.dot(mean_real_list[stim_index], lick_cd)
        predicted_lick_cd_projection = np.dot(mean_prediction_list[stim_index], lick_cd)

        real_lick_cd_list.append(real_lick_cd_projection)
        predicted_lick_cd_list.append(predicted_lick_cd_projection)


    # Save These
    mean_prediction_list = np.array(mean_prediction_list)
    mean_real_list = np.array(mean_real_list)
    real_lick_cd_list = np.array(real_lick_cd_list)
    predicted_lick_cd_list = np.array(predicted_lick_cd_list)
    print("mean predictions", np.shape(mean_prediction_list))

    np.save(os.path.join(save_directory, "Full_Prediction.npy"), prediction)
    np.save(os.path.join(save_directory, "Stim_Predictions.npy"), mean_prediction_list)
    np.save(os.path.join(save_directory, "Stim_Actual.npy"), mean_real_list)
    np.save(os.path.join(save_directory, "real_lick_cd_list.npy"), real_lick_cd_list)
    np.save(os.path.join(save_directory, "predicted_lick_cd_list.npy"), predicted_lick_cd_list)

