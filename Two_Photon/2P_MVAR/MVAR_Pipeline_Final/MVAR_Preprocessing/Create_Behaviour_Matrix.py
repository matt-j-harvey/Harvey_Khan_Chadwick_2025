import os
import numpy as np
from scipy import stats
import tables
from tqdm import tqdm
import matplotlib.pyplot as plt

import MVAR_Preprocessing_Utils


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def create_lagged_matrix(lick_onsets, n_timepoints, n_lags):

    # Create Empty Regressor
    lagged_regressor = np.zeros((n_timepoints, n_lags+1))
    print("lagged regressor", np.shape(lagged_regressor))

    # Populate With Lags
    for onset in lick_onsets:

        # Add Following Regressor
        for lag_index in range(n_lags+1):
            timepoint = onset + lag_index
            if timepoint <= n_timepoints:
                lagged_regressor[timepoint, lag_index] = 1

    """
    # Visualise
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(np.transpose(lagged_regressor[0:100]))

    for onset in lick_onsets[0:2]:
        axis_1.axvline(onset)

    forceAspect(plt.gca())
    plt.show()
    """

    return lagged_regressor


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def plot_design_matrix(design_matrix, regressor_names, save_directory):

    number_of_regressors = np.shape(design_matrix)[1]

    figure_1 = plt.figure(figsize=(20,20))
    axis_1 = figure_1.add_subplot(1,1,1)

    design_matrix_magnitude = np.max(np.abs(design_matrix))
    axis_1.imshow(np.transpose(design_matrix[3000:4000]), cmap="seismic", vmin=-design_matrix_magnitude, vmax=design_matrix_magnitude)

    axis_1.set_yticks(list(range(0, number_of_regressors)))
    axis_1.set_yticklabels(regressor_names)

    figure_1.suptitle("Design Matrix Sample")

    axis_1.set_aspect('equal')
    axis_1.set_title("Design matrix")
    forceAspect(plt.gca())
    plt.savefig(os.path.join(save_directory, "Design_Matrix_Sample.svg"))
    plt.close()


def scale_continous_regressors(regressor_matrix):

    # Subtract Mean
    regressor_mean = np.mean(regressor_matrix, axis=0)
    regressor_sd = np.std(regressor_matrix, axis=0)

    # Devide By 2x SD
    regressor_matrix = np.subtract(regressor_matrix, regressor_mean)
    regressor_matrix = np.divide(regressor_matrix, 2 * regressor_sd)

    #plt.hist(np.ndarray.flatten(regressor_matrix), bins=100)
    #plt.show()
    return regressor_matrix


def zero_running_trace(running_trace, threshold=1260):
    running_trace = np.subtract(running_trace, threshold)
    running_trace = np.clip(running_trace, a_min=0, a_max=None)
    return running_trace


def create_behaviour_matrix(data_root_directory, base_directory, mvar_output_directory, lick_lag=1.0):

    """
    Design Matrix Structure

    Lagged Binarised Lick Trace Upto 1000ms following lick

    Running Trace

    Design Matrix Is Z Scored

    """

    # Load Downsampled AI
    downsampled_ai_file = os.path.join(mvar_output_directory, base_directory, "Behaviour", "Downsampled_AI_Matrix_Framewise.npy")
    downsampled_ai_matrix = np.load(downsampled_ai_file)
    n_timepoints = np.shape(downsampled_ai_matrix)[1]

    # Create Stimuli Dictionary
    stimuli_dictionary = MVAR_Preprocessing_Utils.load_rig_1_channel_dict()

    # Extract Running Trace
    running_trace = downsampled_ai_matrix[stimuli_dictionary["Running"]]

    # Zero Running Trace
    running_trace = zero_running_trace(running_trace)

    # Scale Running Trace
    running_trace = scale_continous_regressors(running_trace)
    running_trace = np.expand_dims(running_trace, axis=1)

    # Get Lagged Lick Regressor
    frame_rate = np.load(os.path.join(data_root_directory, base_directory, "Frame_Rate.npy"))
    n_lags = frame_rate * lick_lag
    n_lags = int(np.around(n_lags, 0))
    print("n lags", n_lags)

    lick_onsets = np.load(os.path.join(mvar_output_directory, base_directory, "Behaviour", "Lick_Onset_Frames.npy"))
    lick_regressors = create_lagged_matrix(lick_onsets, n_timepoints, n_lags=n_lags)

    # Create Design Matrix
    design_matrix = [
        running_trace,
        lick_regressors,
    ]

    design_matrix = np.hstack(design_matrix)

    # Save These
    save_directory = os.path.join(mvar_output_directory, base_directory, "Behaviour")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Create Dictionary Key
    print("Behaviour Matrix before saving", np.shape(design_matrix))
    np.save(os.path.join(save_directory, "Behaviour_Matrix.npy"), design_matrix)





