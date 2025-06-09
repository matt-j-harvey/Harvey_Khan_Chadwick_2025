import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib.pyplot import GridSpec

#from Two_Photon.ALM_2P_Analysis.View_Lick_Mega_Raster import save_directory


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def extract_stim_weights(model_dict):

    Nt = model_dict["Nt"]
    model_params = model_dict["MVAR_Parameters"]
    n_neurons = np.shape(model_params)[0]

    stim_weights_list = []
    for x in range(6):
        regressor_start = n_neurons + (x * Nt)
        regressor_stop = regressor_start + Nt
        stim_weights = model_params[:, regressor_start:regressor_stop]
        stim_weights_list.append(stim_weights)

    return stim_weights_list




def plot_psth(data, axis, x_values, title=None, onset=None):

    # get Data Magnitde
    data_magnitude = np.percentile(np.abs(data), q=95)

    n_neurons = np.shape(data)[0]
    extent = [x_values[0], x_values[-1], 0, n_neurons]
    handle_1 = axis.imshow(data, cmap="bwr", vmin=-data_magnitude, vmax=data_magnitude, extent=extent)
    plt.colorbar(handle_1)

    # Set Title
    axis.set_title(title)

    # Force Aspect
    forceAspect(axis)

    if onset != None:
        axis.axvline(0, c='k', linestyle='dashed')



def plot_stim_weights(model_dict, frame_rate, save_directory):

    # Extract Stim Weights
    stim_weight_list = extract_stim_weights(model_dict)
    visual_context_vis_1 = stim_weight_list[0]
    visual_context_vis_2 = stim_weight_list[1]
    odour_context_vis_1 = stim_weight_list[2]
    odour_context_vis_2 = stim_weight_list[3]
    odour_1 = stim_weight_list[4]
    odour_2 = stim_weight_list[5]

    # Save Stim Weights
    weight_save_directory = os.path.join(save_directory, "Stimuli_Weights")
    if not os.path.exists(weight_save_directory):
        os.makedirs(weight_save_directory)

    np.save(os.path.join(weight_save_directory, "visual_context_vis_1_weights.npy"), visual_context_vis_1)
    np.save(os.path.join(weight_save_directory, "visual_context_vis_2_weights.npy"), visual_context_vis_2)
    np.save(os.path.join(weight_save_directory, "odour_context_vis_1_weights.npy"), odour_context_vis_1)
    np.save(os.path.join(weight_save_directory, "odour_context_vis_2_weights.npy"), odour_context_vis_2)
    np.save(os.path.join(weight_save_directory, "odour_1_weights.npy"), odour_1)
    np.save(os.path.join(weight_save_directory, "odour_2_weights.npy"), odour_2)

    # Create Figure
    figure_1 = plt.figure(figsize=(15, 5))
    gridspec_1 = GridSpec(nrows=2, ncols=4)

    # Create Axes
    visual_context_vis_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
    visual_context_vis_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
    odour_context_vis_1_axis =  figure_1.add_subplot(gridspec_1[1, 0])
    odour_context_vis_2_axis =  figure_1.add_subplot(gridspec_1[1, 1])
    odour_1_axis =  figure_1.add_subplot(gridspec_1[1, 2])
    odour_2_axis =  figure_1.add_subplot(gridspec_1[1, 3])

    # Plot Onset
    Nt = model_dict["Nt"]
    onset = int(Nt/2)
    x_values = list(range(-onset, onset))
    frame_period = 1.0 / frame_rate
    x_values = np.multiply(x_values, frame_period)

    # Plot Data
    plot_psth(visual_context_vis_1, visual_context_vis_1_axis, x_values, title="visual_context_vis_1", onset=onset)
    plot_psth(visual_context_vis_2, visual_context_vis_2_axis, x_values, title="visual_context_vis_2", onset=onset)
    plot_psth(odour_context_vis_1, odour_context_vis_1_axis, x_values, title="odour_context_vis_1", onset=onset)
    plot_psth(odour_context_vis_2, odour_context_vis_2_axis, x_values, title="odour_context_vis_2", onset=onset)
    plot_psth(odour_1, odour_1_axis, x_values, title="odour_1", onset=onset)
    plot_psth(odour_2, odour_2_axis, x_values, title="odour_2", onset=onset)

    plt.savefig(os.path.join(save_directory, "Mean_Stim_Weights.png"))
    plt.close()



def plot_lick_cd_projections(data_directory, session, mvar_directory, frame_rate):

    # Load Lick CD
    lick_cd = np.load(os.path.join(data_directory, session, "Coding_Dimensions", "Lick_CD.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)
    print("Lick CD", np.shape(lick_cd))

    # Iterate Through Each Stimulus
    stimulus_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
        "odour_1",
        "odour_2"]

    # Create Plot Save Directory
    plot_save_directory = os.path.join(mvar_directory, session,  "MVAR_Results", "Lick_CD_Projections")
    if not os.path.exists(plot_save_directory):
        os.makedirs(plot_save_directory)

    for stimulus in stimulus_list:

        # Load Stimulus Weights
        stim_weights = np.load(os.path.join(mvar_directory, session, "MVAR_Results", stimulus + "_weights.npy"))
        print("Stim weights", np.shape(stim_weights))

        # Project
        lick_cd_projection = np.dot(np.transpose(stim_weights), lick_cd)

        # Save
        np.save(os.path.join(plot_save_directory, stimulus + "_lick_cd_projection.npy"), lick_cd_projection)

        # Plot
        plt.title(stimulus)
        plt.plot(lick_cd_projection)

        # Save Fig
        plt.savefig(os.path.join(plot_save_directory, stimulus + ".png"))
        plt.close()

    #save_directory = os.path.join(save_directory)


def plot_recurrent_weights(model_dict, save_directory):

    model_params = model_dict["MVAR_Parameters"]
    n_neurons = np.shape(model_params)[0]
    recurrent_weights = model_params[:, 0:n_neurons]

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    weight_magnitude = np.percentile(np.abs(recurrent_weights), q=95)
    axis_1.imshow(recurrent_weights, cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.savefig(os.path.join(save_directory, "Recurrent_Weights.png"))
    plt.close()


def plot_mean_prediction(regression_matrix, model_dict):
    print("regression_matrix", regression_matrix.keys())
    print("model_dict", model_dict.keys())

    design_matrix = regression_matrix['DesignMatrix']
    print("design_matrix", np.shape(design_matrix))

    model_params = model_dict['MVAR_Parameters']
    print("model_params", np.shape(model_params))

    # Get Prediction
    prediction = np.matmul(design_matrix, np.transpose(model_params))
    print("prediction", np.shape(prediction))

    # Get Real
    real_data = regression_matrix['dFtot']
    real_magnitude = np.percentile(np.abs(real_data), q=95)
    print("real data", np.shape(real_data))

    figure_1 = plt.figure()
    real_axis = figure_1.add_subplot(1,2,1)
    prediction_axis = figure_1.add_subplot(1,2,2)

    prediction_axis.imshow(np.transpose(prediction), cmap='bwr', vmin=-real_magnitude, vmax=real_magnitude)
    real_axis.imshow(real_data, cmap='bwr', vmin=-real_magnitude, vmax=real_magnitude)

    forceAspect(real_axis)
    forceAspect(prediction_axis)
    plt.show()

    # split Prediction by trial type
    n_stim = regression_matrix['N_stim']
    n_trials = regression_matrix['N_trials']
    n_t = regression_matrix['Nt']
    n_neurons = np.shape(model_params)[0]
    n_regressors = np.shape(design_matrix)[1]
    print("n_stim", n_stim, "n_trials", n_trials, "n_t", n_t)

    mean_prediction_list = []
    stim_start = 0
    for stim_index in range(n_stim):

        stim_trials = n_trials[stim_index]
        stim_size = stim_trials * n_t
        stim_stop = stim_start + stim_size

        stim_data = prediction[stim_start:stim_stop]
        print("stim data", np.shape(stim_data))
        stim_data = np.reshape(stim_data, (stim_trials, n_t, n_neurons))
        print("stim data", np.shape(stim_data))

        mean_prediction = np.mean(stim_data, axis=0)
        mean_prediction_list.append(mean_prediction)

        stim_start = stim_stop

    mean_prediction_list = np.array(mean_prediction_list)
    print("mean predictions", np.shape(mean_prediction_list))




def visualise_mvar_results(data_root_directory, mvar_directory, session_list):

    for session in tqdm(session_list, position=0, desc="Session:"):

        # Load Frame Rate
        frame_rate = np.load(os.path.join(data_root_directory, session, "Frame_Rate.npy"))

        # Load Model Dict
        model_dict = np.load(os.path.join(mvar_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
        print("model_dict", model_dict.keys())

        # Unpack Dict
        Nt = model_dict["Nt"]
        model_params = model_dict["MVAR_Parameters"]
        n_neurons = np.shape(model_params)[0]

        # Load Regression Matrix
        regression_matrix = np.load(os.path.join(mvar_directory, session, "Design_Matricies", "Combined_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]
        N_trials = regression_matrix['N_trials']
        print("N_trials", N_trials)
        print("regression_matrix", regression_matrix.keys())

        # Save These
        save_directory = os.path.join(mvar_directory, session, "MVAR_Results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Plot Stim Weights
        plot_stim_weights(model_dict, frame_rate, save_directory)

        # Plot Lick CD Projections
        plot_lick_cd_projections(data_root_directory, session, mvar_directory, frame_rate)

        # Plot Recurrent Weights
        #plot_recurrent_weights(model_dict, save_directory)

        # Plot Prediction
        #plot_mean_prediction(regression_matrix, model_dict)


    # Visualise Group Results?
    stimulus_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
        "odour_1",
        "odour_2"]

    # Plot Average Lick CDs

    # Create Mean Lick CD Save Directory
    group_lick_cd_save_directory = os.path.join(mvar_directory, "Group_Results", "Stimulus_Weights_Lick_CD_Projections")
    if not os.path.exists(group_lick_cd_save_directory):
        os.makedirs(group_lick_cd_save_directory)

    for stimulus in stimulus_list:
        stimulus_projection_list = []

        for session in tqdm(session_list, position=0, desc="Session:"):
            session_lick_cd_projection = np.load(os.path.join(mvar_directory, session, "MVAR_Results", "Lick_CD_Projections", stimulus + "_lick_cd_projection.npy"))
            stimulus_projection_list.append(session_lick_cd_projection)

        stimulus_projection_list = np.array(stimulus_projection_list)
        mean_projection = np.mean(stimulus_projection_list, axis=0)
        plt.plot(mean_projection)
        plt.savefig(os.path.join(group_lick_cd_save_directory, stimulus + "Lick_CD_Projection.png"))
        plt.close()

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

visualise_mvar_results(data_root, mvar_output_root, control_session_list)