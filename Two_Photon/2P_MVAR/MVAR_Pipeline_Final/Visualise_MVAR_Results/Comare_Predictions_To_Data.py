import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


"""
View Raw data which we will compare the MVAR results to do:

This predominantly invovles:
    Trial average histograms
    Trial average lick CD Projections

"""

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def plot_comparison(stim_data, stim_weights, stim_name):

    figure_1 = plt.figure()
    data_axis = figure_1.add_subplot(1,2,1)
    weight_axis = figure_1.add_subplot(1,2,2)

    # Get Magnitude
    combined_data = np.concatenate([stim_data, stim_weights])
    data_magnitude = np.percentile(np.abs(combined_data), q=99)
    vmin = -data_magnitude
    vmax = data_magnitude

    # Plot Data
    data_axis.imshow(stim_data, vmin=vmin, vmax=vmax, cmap='bwr')
    weight_axis.imshow(stim_weights,  vmin=vmin, vmax=vmax, cmap='bwr')

    # Set Titles
    data_axis.set_title("Data")
    weight_axis.set_title("Predictions")

    # Force Aspect
    forceAspect(data_axis)
    forceAspect(weight_axis)

    figure_1.suptitle(stim_name)

    plt.show()



def get_mean_trial_data(data, N_stim, N_trials, Nt, N_neurons):
    print("data",  np.shape(data))
    print("N_stim", N_stim)
    print("N_trials", N_trials)
    print("Nt", Nt)
    print("N_neurons", N_neurons)

    stim_mean_list = []
    stim_start = 0
    for stim_index in range(N_stim):
        stim_stop = stim_start + (N_trials[stim_index] * Nt)
        stim_data = data[stim_start:stim_stop]

        stim_data = np.reshape(stim_data, (N_trials[stim_index], Nt, N_neurons))

        stim_mean = np.mean(stim_data, axis=0)
        stim_mean_list.append(stim_mean)
        stim_start = stim_stop

    return stim_mean_list



def load_raw_data(mvar_root, session):

    # Load Regression Matrix
    regression_matrix = np.load(os.path.join(mvar_root, session, "Design_Matricies", "Combined_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]
    dFtot = np.transpose(regression_matrix["dFtot"])
    Nt = regression_matrix['Nt']
    N_stim = regression_matrix['N_stim']
    N_trials = regression_matrix['N_trials']
    N_neurons = np.shape(dFtot)[1]
    print("dFtot", np.shape(dFtot))
    print("N_trials", N_trials)


    stim_mean_list = get_mean_trial_data(dFtot, N_stim, N_trials, Nt, N_neurons)

    return stim_mean_list



def load_predictions(mvar_root, session):

    # Load Model Parameters
    model_dict = np.load(os.path.join(mvar_root, session, "Full_Model","Combined_Model_Dict.npy"), allow_pickle=True)[()]
    model_parameters = model_dict['MVAR_Parameters']
    print("model_parameters", np.shape(model_parameters))

    # Load Design Matrix
    design_matrix_dict = np.load(os.path.join(mvar_root, session, "Design_Matricies", "Combined_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]
    design_matrix = np.transpose(design_matrix_dict["DesignMatrix"])
    print("design_matrix_dict", design_matrix_dict.keys())
    Nt = design_matrix_dict['Nt']
    N_stim = design_matrix_dict['N_stim']
    N_trials = design_matrix_dict['N_trials']
    N_neurons = design_matrix_dict['Nvar']
    print("design_matrix", np.shape(design_matrix))
    print("Nt", Nt)
    print("N_stim", N_stim)
    print("N_trials", N_trials)
    print("N_neurons", N_neurons)

    # Get Prediction
    prediction = np.dot(model_parameters, design_matrix)
    prediction = np.transpose(prediction)
    print("prediction", np.shape(prediction))

    # Get Predicted Stimuli Means
    stimuli_prediction_list = get_mean_trial_data(prediction,  N_stim, N_trials, Nt, N_neurons)

    return stimuli_prediction_list



def compare_predictions_to_data(session_list, mvar_root):

    stimulus_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
        "odour_1",
        "odour_2",
        ]

    for session in session_list:

        # Load Raw Data and Predictions
        stimuli_mean_list = load_raw_data(mvar_root, session)
        stimuli_prediction_list = load_predictions(mvar_root, session)

        # Iterate Through Each Stimulus
        n_stimuli = len(stimulus_list)
        for stimulus_index in range(n_stimuli):

            # Get Trial Type Data
            trial_mean_data = np.transpose(stimuli_mean_list[stimulus_index])
            trial_mean_pred = np.transpose(stimuli_prediction_list[stimulus_index])

            # Plot Comparison
            plot_comparison(trial_mean_data, trial_mean_pred, stimulus_list[stimulus_index])




# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Final"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

compare_predictions_to_data(control_session_list, mvar_output_root)