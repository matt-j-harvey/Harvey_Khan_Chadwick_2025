import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

import Get_Data_Tensor
import Perform_CV_Decoding
import sklearn.linear_model

from sklearn.linear_model import LogisticRegression

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def get_means_and_bounds(data_list):

    data_list = np.array(data_list)
    data_mean = np.mean(data_list, axis=0)

    data_sem = stats.sem(data_list, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound


def compare_decoding_weights(base_directory, context_start_window, context_stop_window):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    context_x_values = list(range(start_window, stop_window))
    context_x_values = np.multiply(context_x_values, period)

    lick_start_window = -int(2.5 * frame_rate)
    lick_stop_window = int(1 * frame_rate)
    lick_x_values = list(range(lick_start_window, lick_stop_window))
    lick_x_values = np.multiply(lick_x_values, period)

    # Load Lick Means
    licking_activity = np.load(os.path.join(base_directory, "Cell Significance Testing", "lick_mean_activity.npy"))

    # Load Context CD
    context_weights = np.load(os.path.join(base_directory, "Context_Decoding", "Decoding_Coefs.npy"))
    context_weights = np.squeeze(context_weights)

    # Sort Rasters
    lick_mean = np.mean(licking_activity, axis=0)
    licking_indicies = lick_mean.argsort()
    licking_indicies = np.flip(licking_indicies)

    context_mean = np.mean(context_weights, axis=0)
    context_indicies = context_mean.argsort()
    context_indicies = np.flip(context_indicies)


    licking_magnitude = np.percentile(np.abs(lick_mean), q=99)
    context_magnitude = np.percentile(np.abs(context_mean), q=99)
    print("Licking activity", np.shape(licking_activity))
    print("context_weights", np.shape(context_weights))


    licking_activity_lick_sort = licking_activity[:, licking_indicies]
    context_weights_lick_sort = context_weights[:, licking_indicies]

    licking_activity_context_sort = licking_activity[:, context_indicies]
    context_weights_context_sort = context_weights[:, context_indicies]

    figure_1 = plt.figure(figsize=(10,7))
    licking_lick_axis = figure_1.add_subplot(2,2,1)
    context_lick_axis = figure_1.add_subplot(2,2,2)
    licking_context_axis = figure_1.add_subplot(2,2,3)
    context_context_axis = figure_1.add_subplot(2,2,4)

    n_neurons = np.shape(licking_activity_lick_sort)[1]

    licking_lick_axis.imshow(np.transpose(licking_activity_lick_sort), cmap="bwr", vmax=licking_magnitude, vmin=-licking_magnitude,     extent = [lick_x_values[0], lick_x_values[-1], 0, n_neurons])
    context_lick_axis.imshow(np.transpose(context_weights_lick_sort), cmap="bwr", vmax=context_magnitude, vmin=-context_magnitude,     extent = [context_x_values[0], context_x_values[-1], 0, n_neurons])
    licking_context_axis.imshow(np.transpose(licking_activity_context_sort), cmap="bwr", vmax=licking_magnitude, vmin=-licking_magnitude,     extent = [lick_x_values[0], lick_x_values[-1], 0, n_neurons])
    context_context_axis.imshow(np.transpose(context_weights_context_sort), cmap="bwr", vmax=context_magnitude, vmin=-context_magnitude,     extent = [context_x_values[0], context_x_values[-1], 0, n_neurons])

    forceAspect(licking_lick_axis)
    forceAspect(context_lick_axis)
    forceAspect(licking_context_axis)
    forceAspect(context_context_axis)

    # Set X Labels

    licking_context_axis.set_xlabel("Time (S)")
    context_context_axis.set_xlabel("Time (S)")

    # Set Y Labels
    licking_lick_axis.set_ylabel("Neurons")
    context_lick_axis.set_ylabel("Neurons")
    licking_context_axis.set_ylabel("Neurons")
    context_context_axis.set_ylabel("Neurons")

    # Set Titles
    licking_lick_axis.set_title("Licking Activity - Sorted")
    context_lick_axis.set_title("Context Decoding Weights - Sorted by Lick")
    licking_context_axis.set_title("Licking Activity - Sorted by Context Weights")
    context_context_axis.set_title("Context Decoding Weights - Sorted")



    plt.show()







def create_combined_dataset(condition_1_tensor, condition_2_tensor):

    # Assumes Tensors Are Of Shape (n_trials x n_timepoints x n_neurons)
    n_condition_1_trials = np.shape(condition_1_tensor)[0]
    n_condition_2_trials = np.shape(condition_2_tensor)[0]
    condition_1_labels = np.zeros(n_condition_1_trials)
    condition_2_labels = np.ones(n_condition_2_trials)
    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])
    combined_data = np.vstack([condition_1_tensor, condition_2_tensor])
    return combined_data, combined_labels


def decode_context_session(base_directory, start_window, stop_window, baseline_correction=False):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)
    #df_matrix = stats.zscore(df_matrix, axis=0)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    # Load Onsets
    visual_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets" + ".npy"))
    odour_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_stable_vis_2_onsets" + ".npy"))

    # Get Tensors
    visual_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, visual_onsets, start_window, stop_window, baseline_correction=baseline_correction, baseline_start=0, baseline_stop=5)
    odour_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, odour_onsets, start_window, stop_window, baseline_correction=baseline_correction, baseline_start=0, baseline_stop=5)
    print("visual_tensor", np.shape(visual_tensor))
    print("odour_tensor", np.shape(odour_tensor))

    # Create Combined Data
    combined_data, combined_labels = create_combined_dataset(odour_tensor, visual_tensor)
    print("combined_data", np.shape(combined_data))
    print("combined_labels", np.shape(combined_labels))

    # Iterate Through Timepoints
    score_list = []
    coef_list = []
    n_timepoints = np.shape(visual_tensor)[1]
    for timepoint_index in range(n_timepoints):

        model = LogisticRegression(max_iter=500)

        # Perform CV Decoding
        average_score, average_coefs = Perform_CV_Decoding.perform_cv(model, combined_data[:, timepoint_index], combined_labels, n_balance_iterations=20, n_folds=5)
        print("average_score", average_score)
        score_list.append(average_score)
        coef_list.append(average_coefs)


    # Perform shuffled Decoding as Null Model
    shuffled_score_list = []
    for timepoint_index in range(n_timepoints):
        model = LogisticRegression(max_iter=500)
        average_score = Perform_CV_Decoding.perform_shuffled_decoding(model, combined_data[:, timepoint_index], combined_labels, n_balance_iterations=20, n_folds=5)
        print("shuffled average_score", average_score)
        shuffled_score_list.append(average_score)

    # Save Results
    save_directory = os.path.join(base_directory, "Context_Decoding")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    np.save(os.path.join(save_directory, "Decoding_Scores.npy"), score_list)
    np.save(os.path.join(save_directory, "Decoding_Coefs.npy"), coef_list)
    np.save(os.path.join(save_directory, "Decoding_Scores_Shuffled.npy", shuffled_score_list))

    return score_list


data_root_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

start_window = -18
stop_window = 18


for session in tqdm(session_list):
    decode_context_session(os.path.join(data_root_directory, session), start_window, stop_window)
    #compare_decoding_weights(os.path.join(data_root_directory, session),  start_window, stop_window)


score_list = []
for session in tqdm(session_list):
    session_scores = np.load(os.path.join(data_root_directory, session, "Context_Decoding", "Decoding_Scores.npy"))
    score_list.append(session_scores)

data_mean, lower_bound, upper_bound = get_means_and_bounds(np.array(score_list))

frame_rate = np.load(os.path.join(data_root_directory, session, "Frame_Rate.npy"))
period = 1.0 / frame_rate

figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)
x_values = list(range(start_window, stop_window))
x_values = np.multiply(x_values, period)
axis_1.plot(x_values, data_mean)
axis_1.fill_between(x_values, y1=lower_bound, y2=upper_bound, alpha=0.5)
axis_1.set_ylim([0.4, 1])
axis_1.set_ylabel("Decoding accuracy")
axis_1.set_xlabel("Time (S)")
axis_1.axhline(0.5, color='k', linestyle='dashed')
axis_1.spines[['right', 'top']].set_visible(False)

plt.show()