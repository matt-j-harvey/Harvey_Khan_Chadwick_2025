import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import alpha
from scipy import stats
import Two_Photon_Utils


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


def get_rt_time(onset, downsampled_lick_trace, period, lick_threshold):

    n_timepoints = len(downsampled_lick_trace)
    licked = False
    count = 0
    while licked == False:
        if downsampled_lick_trace[onset + count] > lick_threshold:
            return count * period

        else:
            count += 1
            if count == n_timepoints:
                return None





def split_odour_1_by_preceeding_vis_stim(base_directory):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # If Vis 1, if Hit, if Preceeded by dsiaul stimulus
    odour_1_preceeded_by_vis_1 = []
    odour_1_preceeded_by_vis_2 = []
    odour_1_no_irrel = []

    n_trials = np.shape(behaviour_matrix)[0]

    for trial_index in range(n_trials):
        trial_data = behaviour_matrix[trial_index]
        trial_type = trial_data[1]
        correct = trial_data[3]
        preceeded_by_irrel = trial_data[5]
        irrel_type = trial_data[6]
        ignore_irrel = trial_data[7]
        stimuli_onset_frame = trial_data[18]

        if trial_type == 3:
            if correct == 1:
                if preceeded_by_irrel == 0:
                    odour_1_no_irrel.append(stimuli_onset_frame)

                elif preceeded_by_irrel == 1:
                    if ignore_irrel == 1:

                        if irrel_type == 1:
                            odour_1_preceeded_by_vis_1.append(stimuli_onset_frame)
                        elif irrel_type == 2:
                            odour_1_preceeded_by_vis_2.append(stimuli_onset_frame)

    return [odour_1_preceeded_by_vis_1, odour_1_preceeded_by_vis_2, odour_1_no_irrel]



def get_rt_time(onset, downsampled_lick_trace, period, lick_threshold):

    n_timepoints = len(downsampled_lick_trace)
    licked = False
    count = 0
    while licked == False:
        if downsampled_lick_trace[onset + count] > lick_threshold:
            return count * period

        else:
            count += 1
            if count == n_timepoints:
                return None


def get_rt_time_list(onset_list, downsampled_lick_trace, period, lick_threshold):

    rt_time_list = []
    for onset in onset_list:
        trial_rt = get_rt_time(onset, downsampled_lick_trace, period, lick_threshold)
        if trial_rt != None:
            rt_time_list.append(trial_rt)
    return rt_time_list


data_root = r"C:\Users\harveym\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"

session_list = [
    os.path.join(data_root, r"65.2a\2024_08_05_Switching"),
    os.path.join(data_root, r"65.2b\2024_07_31_Switching"),
    os.path.join(data_root, r"67.3b\2024_08_09_Switching"),
    os.path.join(data_root, r"67.3C\2024_08_20_Switching"),
    os.path.join(data_root, r"69.2a\2024_08_12_Switching"),
    os.path.join(data_root, r"72.3C\2024_09_10_Switching"),
    ]


lick_threshold=600

mean_vis_1_list = []
mean_vis_2_list = []
mean_none_list = []

for base_directory in session_list:

    # Get Odour 1 Onsets
    [odour_1_preceeded_by_vis_1,
     odour_1_preceeded_by_vis_2,
     odour_1_no_irrel] = split_odour_1_by_preceeding_vis_stim(base_directory)

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))
    print("stack_onsets", len(stack_onsets))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1)/frame_rate
    period = np.multiply(period, 1000)

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Behaviour", "AI_Matrix.npy"))

    # Load Stimuli Dict
    stimuli_dict = Two_Photon_Utils.load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[:, stimuli_dict["Lick"]]

    # Downsample Lick Trace
    downsampled_lick_trace = downsample_ai_trace(lick_trace, stack_onsets)

    # Get RT Dists
    vis_1_pre_rts = get_rt_time_list(odour_1_preceeded_by_vis_1, downsampled_lick_trace, period, lick_threshold)
    vis_2_pre_rts = get_rt_time_list(odour_1_preceeded_by_vis_2, downsampled_lick_trace, period, lick_threshold)
    no_pre_rts = get_rt_time_list(odour_1_no_irrel, downsampled_lick_trace, period, lick_threshold)

    mean_vis_1_list.append(np.mean(vis_1_pre_rts))
    mean_vis_2_list.append(np.mean(vis_2_pre_rts))
    mean_none_list.append(np.mean(no_pre_rts))

    """
    plt.hist(vis_1_pre_rts, color='b', alpha=0.5, density=True)
    plt.hist(vis_2_pre_rts, color='r', alpha=0.5, density=True)
    plt.hist(no_pre_rts, color='g', alpha=0.5, density=True)
    plt.show()
    """

n_mice = len(session_list)
for mouse_index in range(n_mice):
    plt.plot([mean_vis_1_list[mouse_index], mean_vis_2_list[mouse_index], mean_none_list[mouse_index]])

plt.show()


group_vis_1_mean = np.mean(mean_vis_1_list)
group_vis_2_mean = np.mean(mean_vis_2_list)
group_none_mean = np.mean(mean_none_list)

t_stat, p_value = stats.ttest_rel(mean_vis_1_list, mean_vis_2_list, axis=0)
print("t_stat", t_stat)
print("p_value", p_value)
print("group_vis_1_mean", group_vis_1_mean, "group_vis_2_mean", group_vis_2_mean, "group_none_mean", group_none_mean)