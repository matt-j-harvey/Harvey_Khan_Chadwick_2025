import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from scipy import stats

import Get_Data_Tensor

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def load_df_matrix(base_directory):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)

    # Z Score
    df_matrix = stats.zscore(df_matrix, axis=0)

    # Remove NaN
    df_matrix = np.nan_to_num(df_matrix)

    # Return
    return df_matrix



def get_mean_lick_activity(lick_trace, onset_list, start_window, stop_window):

    n_timepoints = len(lick_trace)

    lick_trace_list = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start > 0 and trial_stop < n_timepoints:
            trial_data = lick_trace[trial_start:trial_stop]
            lick_trace_list.append(trial_data)

    lick_trace_list = np.array(lick_trace_list)
    mean_lick_trace = np.mean(lick_trace_list, axis=0)

    lick_std = np.std(lick_trace_list, axis=0)
    lick_upper_bound = np.add(mean_lick_trace, lick_std)
    lick_lower_bound = np.subtract(mean_lick_trace, lick_std)

    return mean_lick_trace, lick_lower_bound, lick_upper_bound


def test_cosine_simmilarity(tensor_1, tensor_2, mean_period, save_directory):

    #Get Means
    group_1_vectors = np.mean(tensor_1[:, mean_period[0]:mean_period[1]], axis=1)
    group_2_vectors = np.mean(tensor_2[:, mean_period[0]:mean_period[1]], axis=1)

    # Get Average Cosine Similarity
    group_1_mean_vector = np.mean(group_1_vectors, axis=0)
    group_2_mean_vector = np.mean(group_2_vectors, axis=0)
    real_distance = np.dot(group_1_mean_vector,group_2_mean_vector)/(norm(group_1_mean_vector)*norm(group_2_mean_vector))

    # Create Combined Dataset
    group_1_labels = np.zeros(len(group_1_vectors))
    group_2_labels = np.ones(len(group_2_vectors))
    combined_data = np.concatenate([group_1_vectors, group_2_vectors])
    combined_labels = np.concatenate([group_1_labels, group_2_labels])

    n_iterations = 100000

    shuffled_distribution = []
    for iteration in tqdm(range(n_iterations)):
        shuffled_labels = np.copy(combined_labels)
        np.random.shuffle(shuffled_labels)

        shuffled_group_1_data = combined_data[np.argwhere(shuffled_labels==0)]
        shuffled_group_2_data = combined_data[np.argwhere(shuffled_labels==1)]

        shuffled_group_1_mean_vector = np.squeeze(np.mean(shuffled_group_1_data, axis=0))
        shuffled_group_2_mean_vector = np.squeeze(np.mean(shuffled_group_2_data, axis=0))
        shuffle_distance = np.dot(shuffled_group_1_mean_vector, shuffled_group_2_mean_vector) / (norm(shuffled_group_1_mean_vector) * norm(shuffled_group_2_mean_vector))
        shuffled_distribution.append(shuffle_distance)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.hist(shuffled_distribution, bins=60)
    axis_1.axvline(real_distance, c='k', linestyle='dashed')
    axis_1.set_xlim([0.4, 1])
    axis_1.set_xlabel("Cosine Simmilarity")
    axis_1.set_ylabel("Frequency")
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.savefig(os.path.join(save_directory, "Coding_Dimension_Cosine_Simmilarity.png"))
    plt.show()

    # Return Values
    mean_shuffled_distance = np.mean(shuffled_distribution)

    return real_distance, mean_shuffled_distance



def plot_group_cosine_simmilarity(output_directory, session_list):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    real_means_list = []
    shuffled_means_list = []

    for session in session_list:

        # Load Data
        session_real_value = np.load(os.path.join(output_directory, session, "Real_Costine_Simmilarity.npy"))
        session_shuffled_value = np.load(os.path.join(output_directory, session, "Shuffled_Costine_Simmilarity.npy"))

        real_means_list.append(session_real_value)
        shuffled_means_list.append(session_shuffled_value)


        axis_1.plot([0, 1], [session_real_value, session_shuffled_value])
        axis_1.scatter([0, 1], [session_real_value, session_shuffled_value])

    t_stat, p_value = stats.ttest_rel(real_means_list, shuffled_means_list)
    print("pvalue", p_value)
    axis_1.set_ylim([0, 1])
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_xticks([0, 1], ["Actual", "Shuffled"], rotation='horizontal')
    plt.show()



def pad_tensor_with_nans(tensor, full_duration):

    n_trials = len(tensor)
    n_neurons = np.shape(tensor[0])[1]
    print("N neurons", n_neurons)

    # Create Tensor of NaNs
    padded_tensor = np.empty((n_trials, full_duration, n_neurons))
    padded_tensor[:] = np.nan

    # Fill It With Each Trial
    for trial_index in range(n_trials):
        trial_data = tensor[trial_index]
        print("trial_data", np.shape(trial_data))

        padded_tensor[trial_index, 0:len(trial_data)] = trial_data

    return padded_tensor


def plot_psth(vis_1_mean, vis_2_mean, odour_1_mean, odour_2_mean):

    # Sort Matricies
    vis_1_response = np.mean(vis_1_mean[6:12], axis=0)
    sorted_indicies = np.argsort(vis_1_response)
    sorted_indicies = np.flip(sorted_indicies)
    vis_1_mean = vis_1_mean[:, sorted_indicies]
    vis_2_mean = vis_2_mean[:, sorted_indicies]
    odour_1_mean = odour_1_mean[:, sorted_indicies]
    odour_2_mean = odour_2_mean[:, sorted_indicies]

    # Get Differences
    vis_diff = np.subtract(vis_1_mean, vis_2_mean)
    odour_diff = np.subtract(odour_1_mean, odour_2_mean)

    rewarded_diff = np.subtract(vis_1_mean, odour_1_mean)
    unrewarded_diff = np.subtract(vis_2_mean, odour_2_mean)




    # Create Figure
    figure_1 = plt.figure()

    vis_1_axis = figure_1.add_subplot(3,3,1)
    vis_2_axis = figure_1.add_subplot(3,3,2)
    vis_diff_axis = figure_1.add_subplot(3,3,3)

    odr_1_axis = figure_1.add_subplot(3,3,4)
    odr_2_axis = figure_1.add_subplot(3,3,5)
    odr_diff_axis = figure_1.add_subplot(3,3,6)

    rewarded_diff_axis = figure_1.add_subplot(3,3,7)
    unrewarded_diff_axis = figure_1.add_subplot(3,3,8)



    magnitude = 0.6
    vis_1_axis.imshow(np.transpose(vis_1_mean), cmap="bwr", vmin=-magnitude, vmax=magnitude)
    vis_2_axis.imshow(np.transpose(vis_2_mean), cmap="bwr", vmin=-magnitude, vmax=magnitude)
    vis_diff_axis.imshow(np.transpose(vis_diff), cmap="bwr",   vmin=-magnitude, vmax=magnitude)

    odr_1_axis.imshow(np.transpose(odour_1_mean), cmap="bwr", vmin=-magnitude, vmax=magnitude)
    odr_2_axis.imshow(np.transpose(odour_2_mean), cmap="bwr", vmin=-magnitude, vmax=magnitude)
    odr_diff_axis.imshow(np.transpose(odour_diff), cmap="bwr",   vmin=-magnitude, vmax=magnitude)

    rewarded_diff_axis.imshow(np.transpose(rewarded_diff), cmap="bwr",   vmin=-magnitude, vmax=magnitude)
    unrewarded_diff_axis.imshow(np.transpose(unrewarded_diff), cmap="bwr", vmin=-magnitude, vmax=magnitude)


    # Force Aspect
    forceAspect(vis_1_axis)
    forceAspect(vis_2_axis)
    forceAspect(vis_diff_axis)
    forceAspect(odr_1_axis)
    forceAspect(odr_2_axis)
    forceAspect(odr_diff_axis)
    forceAspect(rewarded_diff_axis)
    forceAspect(unrewarded_diff_axis)

    plt.show()


def compare_choice_vectors(data_root, session, output_root, start_window, stop_window):

    # Load Delta F Matrix
    df_matrix = load_df_matrix(os.path.join(data_root, session))
    print("df_matrix", np.shape(df_matrix))

    # Load Onsets
    vis_1_stimuli_onsets = np.load(os.path.join(output_root, session,  "Trial_Timings",  "vis_1_stimuli_onsets.npy"))
    vis_1_response_onsets = np.load(os.path.join(output_root, session,  "Trial_Timings",  "vis_1_response_onsets.npy"))
    odr_1_stimuli_onsets = np.load(os.path.join(output_root, session,  "Trial_Timings",  "odr_1_stimuli_onsets.npy"))
    odr_1_response_onsets = np.load(os.path.join(output_root, session,  "Trial_Timings",  "odr_1_response_onsets.npy"))
    vis_2_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))
    odour_2_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Odour_2_onset_frames.npy"))

    # Get Data Tensors
    vis_1_tensor = Get_Data_Tensor.get_data_tensor_seperate_starts_stops(df_matrix, vis_1_stimuli_onsets, vis_1_response_onsets, start_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    vis_1_tensor = pad_tensor_with_nans(vis_1_tensor, stop_window-start_window)
    vis_2_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

    odr_1_tensor = Get_Data_Tensor.get_data_tensor_seperate_starts_stops(df_matrix, odr_1_stimuli_onsets, odr_1_response_onsets, start_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    odr_1_tensor = pad_tensor_with_nans(odr_1_tensor, stop_window-start_window)
    odour_2_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, odour_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

    # Get Mean Responses
    vis_1_mean = np.nanmean(vis_1_tensor, axis=0)
    vis_2_mean = np.mean(vis_2_tensor, axis=0)
    odour_1_mean = np.nanmean(odr_1_tensor, axis=0)
    odour_2_mean = np.mean(odour_2_tensor, axis=0)

    # Plot PSTH
    plot_psth(vis_1_mean, vis_2_mean, odour_1_mean, odour_2_mean)


    print("vis_1_tensor", np.shape(vis_1_tensor))
    print("odr_1_tensor", np.shape(odr_1_tensor))
    print("vis_2_tensor", np.shape(vis_2_tensor))
    print("odour_2_tensor", np.shape(odour_2_tensor))



data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Response_Dimension_Comparison"

# - 2.5 Seconds
# 3 Seconds
start_window = -6
stop_window = 18


session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
    ]

for session in session_list:
    compare_choice_vectors(data_root, session, output_root, start_window, stop_window)