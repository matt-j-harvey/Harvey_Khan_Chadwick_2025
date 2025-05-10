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



def downsample_ai_trace(ai_trace, stack_onsets):

    # Get Average Stack Duration
    stack_duration_list = np.diff(stack_onsets)
    mean_stack_duration = int(np.mean(stack_duration_list))

    downsampled_trace = []
    n_stacks = len(stack_onsets)
    for stack_index in range(n_stacks-1):
        stack_start = stack_onsets[stack_index]
        stack_stop = stack_onsets[stack_index + 1]
        stack_data = ai_trace[stack_start:stack_stop]
        stack_data = np.mean(stack_data)
        downsampled_trace.append(stack_data)

    # Add Last
    final_data = ai_trace[stack_onsets[-1]:stack_onsets[-1] + mean_stack_duration]
    final_data = np.mean(final_data)
    downsampled_trace.append(final_data)

    return downsampled_trace


def get_downsampled_lick_trace(base_directory):

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Behaviour", "AI_Matrix.npy"))
    print("ai_data", np.shape(ai_data))

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))
    print("stack_onsets", len(stack_onsets))

    # Load Stimuli Dict
    stimuli_dict = Two_Photon_Utils.load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[:, stimuli_dict["Lick"]]

    # Downsample Lick Trace
    downsampled_lick_trace = downsample_ai_trace(lick_trace, stack_onsets)


    return downsampled_lick_trace



def get_next_onset(trace, onset, threshold, max_rt):

    index = 0
    above = False
    while above == False:
        instantaneous_value = trace[onset + index]
        if instantaneous_value > threshold:
            return onset + index

        else:
            index += 1
            if index > max_rt:
                return False



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

    base_directory = os.path.join(data_root, session)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1)/frame_rate

    # Load Onsets
    visual_onsets, odour_onsets = get_prelick_onsets(base_directory, frame_rate, min_rt=0.5, max_rt=2.5, lick_threshold=600)

    # Load DF Matrix
    df_matrix = load_df_matrix(base_directory)

    # Get Tensors
    visual_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, visual_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    odour_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, odour_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

    # Visualise Mean Behaviour Traces
    downsampled_lick_trace = get_downsampled_lick_trace(base_directory)
    mean_visual_lick, visual_lick_lower, visual_lick_upper = get_mean_lick_activity(downsampled_lick_trace, visual_onsets, start_window, stop_window)
    mean_odour_lick, odour_lick_lower, odour_lick_upper = get_mean_lick_activity(downsampled_lick_trace, odour_onsets, start_window, stop_window)

    x_values = list(range(start_window, stop_window))
    print("x values", x_values)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, mean_visual_lick, c='b')
    axis_1.fill_between(x=x_values, y1=visual_lick_lower, y2=visual_lick_upper, alpha=0.5, color='b')
    axis_1.plot(x_values, mean_odour_lick, c='g')
    axis_1.fill_between(x=x_values, y1=odour_lick_lower, y2=odour_lick_upper, alpha=0.5, color='g')
    plt.show()

    print("Visual Tensor", np.shape(visual_tensor))
    print("Odour Tensor", np.shape(odour_tensor))

    save_directory = os.path.join(output_root, session)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    print("save_directory", save_directory)

    View_PSTH.view_two_psth(visual_tensor,
                          odour_tensor,
                          start_window,
                          stop_window,
                          period,
                          save_directory,
                          "Lick Conditions",
                          -12,
                          -6,
                            plot_titles=["Visual", "Odour"])

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