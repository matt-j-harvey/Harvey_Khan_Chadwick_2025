import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from scipy import stats

import Two_Photon_Utils
import View_PSTH
import Get_Data_Tensor


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





def get_prelick_onsets(base_directory, frame_rate, min_rt=0.5, max_rt=2.5, lick_threshold=600):

    # Load Lick Trace
    lick_trace = get_downsampled_lick_trace(base_directory)

    min_rt_frames = min_rt * frame_rate
    max_rt_frames = max_rt * frame_rate

    # Load Visual Hit Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))
    print("Hit Onsets", vis_1_onsets)

    # Load Odour Hit Onsets
    odour_1_cued_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "odour_1_cued_onsets.npy"))
    odour_1_non_cued_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "odour_1_not_cued_onsets.npy"))
    odour_onsets = list(odour_1_non_cued_onsets)  + list(odour_1_cued_onsets)

    visual_lick_onsets = []
    for onset in vis_1_onsets:
        trial_lick_onset = get_next_onset(lick_trace, onset, lick_threshold, max_rt_frames)
        print(trial_lick_onset)
        if trial_lick_onset - onset > min_rt_frames:
            visual_lick_onsets.append(trial_lick_onset)

    odour_lick_onsets = []
    for onset in odour_onsets:
        trial_lick_onset = get_next_onset(lick_trace, onset, lick_threshold, max_rt_frames)
        if trial_lick_onset - onset > min_rt_frames:
            odour_lick_onsets.append(trial_lick_onset)

    return visual_lick_onsets, odour_lick_onsets



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




def compare_lick_vectors(data_root, session, start_window, stop_window, output_root):

    # Get Lick CD
    lick_cd_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"
    lick_cd = np.load(os.path.join(lick_cd_directory, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

    # Load Context CD
    context_cd = np.load(os.path.join(data_root, session, "Context_Decoding", "Decoding_Coefs.npy"))
    context_cd = np.mean(context_cd[0:18], axis=0)
    context_cd = np.squeeze(context_cd)

    # Test Session Cosine Simmilarity
    real_distance, mean_shuffled_distance = test_cosine_simmilarity(visual_tensor, odour_tensor, [-12,-6], save_directory)
    np.save(os.path.join(save_directory, "Real_Costine_Simmilarity.npy"), real_distance)
    np.save(os.path.join(save_directory, "Shuffled_Costine_Simmilarity.npy"), mean_shuffled_distance)

    return visual_tensor, odour_tensor, period





data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Lick_Coding_Dimension_Comparison"

start_window=-16
stop_window=6


session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
    ]

plot_group_cosine_simmilarity(output_root, session_list)


visual_tensor_list = []
odour_tensor_list = []
for session in session_list:
    print(output_root)
    visual_tensor, odour_tensor, frame_rate = compare_lick_vectors(data_root, session, start_window, stop_window, output_root)
    mean_visual = np.mean(visual_tensor, axis=0)
    mean_odour = np.mean(odour_tensor, axis=0)

    visual_tensor_list.append(mean_visual)
    odour_tensor_list.append(mean_odour)

visual_tensor_list = np.hstack(visual_tensor_list)
odour_tensor_list = np.hstack(odour_tensor_list)

print("visual_tensor_list", np.shape(visual_tensor_list))
print("odour_tensor_list", np.shape(odour_tensor_list))

base_directory= None

# mean_1, mean_2, start_window, stop_window, frame_rate, sorting_window_start, sorting_window_stop,

View_PSTH.view_two_mean_psth(visual_tensor_list,
                             odour_tensor_list,
                             start_window,
                             stop_window,
                             frame_rate,
                             -12,
                             -6,
                             plot_titles=["Visual", "Odour"])