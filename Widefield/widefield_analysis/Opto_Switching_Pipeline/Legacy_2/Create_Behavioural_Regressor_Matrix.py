import os
import numpy as np
from scipy import stats
import tables
from tqdm import tqdm
import matplotlib.pyplot as plt

import Opto_GLM_Utils


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def get_step_onsets(trace, threshold=1, window=10):
    state = 0
    number_of_timepoints = len(trace)
    onset_times = []
    time_below_threshold = 0

    onset_line = []

    for timepoint in range(number_of_timepoints):
        if state == 0:
            if trace[timepoint] > threshold:
                state = 1
                onset_times.append(timepoint)
                time_below_threshold = 0
            else:
                pass
        elif state == 1:
            if trace[timepoint] > threshold:
                time_below_threshold = 0
            else:
                time_below_threshold += 1
                if time_below_threshold > window:
                    state = 0
                    time_below_threshold = 0
        onset_line.append(state)

    return onset_times


def create_lagged_matrix(lick_trace, lick_threshold, n_lags):

    # Get Lick Onsets
    lick_onsets  = get_step_onsets(lick_trace, lick_threshold)

    # Create Empty Regressor
    n_timepoints = len(lick_trace)
    lagged_regressor = np.zeros((n_timepoints, (n_lags*2)+1))

    # Populate With Lags
    for onset in lick_onsets:

        # Add Preceeding Regressor
        for lag_index in range(n_lags):
            timepoint = onset - (n_lags - lag_index)
            if timepoint >= 0:
                lagged_regressor[timepoint, lag_index] = 1

        # Add Following Regressor
        for lag_index in range(n_lags+1):
            timepoint = onset + lag_index
            if timepoint < n_timepoints:
                lagged_regressor[timepoint, n_lags + lag_index] = 1

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



def visualise_lick_threshold(lick_trace, lick_threshold):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(lick_trace)
    axis_1.axhline(lick_threshold, c='k', linestyle='dashed')
    plt.show()

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


def zero_running_trace(running_trace, threshold=0.42):
    running_trace = np.subtract(running_trace, threshold)
    running_trace = np.clip(running_trace, a_min=0, a_max=None)
    return running_trace


def trim_design_matrix_if_needed(design_matrix_list):

    length_list = []
    for item in design_matrix_list:
        length_list.append(np.shape(item)[0])
    min_length = np.min(length_list)

    trimmed_design_matrix_list = []
    for item in design_matrix_list:
        item = item[0:min_length]
        trimmed_design_matrix_list.append(item)

    return trimmed_design_matrix_list



def create_behaviour_matrix(data_root_directory, base_directory, mvar_output_directory, lick_lag=1.0):

    """
    Design Matrix Structure

    Lagged Binarised Lick Trace Upto 1000 prior to 1000 following

    Running Trace

    Design Matrix Is Z Scored

    """

    # Load Downsampled AI
    downsampled_ai_file = os.path.join(data_root_directory, base_directory, "Downsampled_AI_Matrix_Framewise.npy")
    downsampled_ai_matrix = np.load(downsampled_ai_file)
    print("downsampled_ai_matrix", np.shape(downsampled_ai_matrix))
    # Create Stimuli Dictionary
    stimuli_dictionary = Opto_GLM_Utils.create_stimuli_dictionary()

    # Extract Lick and Running Traces
    lick_trace = downsampled_ai_matrix[stimuli_dictionary["Lick"]]
    running_trace = downsampled_ai_matrix[stimuli_dictionary["Running"]]

    # Zero Running Trace
    running_trace = zero_running_trace(running_trace)

    # Scale Running Trace
    running_trace = scale_continous_regressors(running_trace)
    running_trace = np.expand_dims(running_trace, axis=1)

    # Get Lagged Lick Regressor
    n_lags = int(np.around(1500/36, 0))
    lick_threshold = np.load(os.path.join(data_root_directory, base_directory, "Lick_Threshold.npy"))
    lick_regressors = create_lagged_matrix(lick_trace, lick_threshold, n_lags=n_lags)

    # Load Mousecam Components
    face_motion_svd = np.load(os.path.join(data_root_directory, base_directory, "Mousecam_Analysis", "Face_Motion_SVD.npy"))
    #face_motion_svd = np.random.uniform(low=-1, high=1, size=np.shape(face_motion_svd))
    print("face_motion_svd", np.shape(face_motion_svd))
    print("running_trace", np.shape(running_trace))
    print("lick_regressors", np.shape(lick_regressors))

    #print("Running Trace", np.shape(running_trace))
    # Create Design Matrix
    design_matrix = [
        running_trace,
        lick_regressors,
        face_motion_svd
    ]

    # Trim If Needed
    design_matrix = trim_design_matrix_if_needed(design_matrix)

    design_matrix = np.hstack(design_matrix)
    print("design_matrix", np.shape(design_matrix))

    # Save These
    save_directory = os.path.join(mvar_output_directory, base_directory, "Behaviour")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create Dictionary Key
    #print("Behaviour Matrix before saving", np.shape(design_matrix))
    np.save(os.path.join(save_directory, "Behavioural_Regressor_Matrix.npy"), design_matrix)





