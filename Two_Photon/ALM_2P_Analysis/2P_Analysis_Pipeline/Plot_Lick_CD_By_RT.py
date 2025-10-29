import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import stats

import ALM_Analysis_Utils

# Split Licks By RT
# 500 - 750
# 750 - 1000
# 1000 - 1250
# 1250 - 1500


def get_hits_by_rt(behaviour_matrix, rt_window_start, rt_window_stop):

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        reaction_time = trial[23]
        onset_frame = trial[18]

        if trial_type == 1:
            if correct == 1:
                if reaction_time > rt_window_start and reaction_time <= rt_window_stop:
                    onset_list.append(onset_frame)

    return onset_list





def get_selected_onsets(onset_rt_matrix, rt_window):

    # Get Window Times
    window_start = rt_window[0]
    window_stop = rt_window[1]

    # Load Onsets
    onsets = onset_rt_matrix[:, 0]
    rt_times = onset_rt_matrix[:, 1]

    n_trials = len(onsets)
    selected_onsets = []
    for trial_index in range(n_trials):
        trial_onset = int(onsets[trial_index])
        trial_rt = rt_times[trial_index]

        if trial_rt >= window_start and trial_rt < window_stop:
            selected_onsets.append(trial_onset)


    return selected_onsets



def plot_graph(window_means, rt_windows_list, start_window, stop_window, frame_rate, save_directory):

    # Get Means and SEMs
    n_windows = len(rt_windows_list)
    group_condition_means = []
    group_condition_lower_bounds = []
    group_condition_higher_bounds = []

    for window_index in range(n_windows):
        condition_means = np.array(window_means[window_index])
        group_mean, lower_bound, upper_bound = ALM_Analysis_Utils.get_sem_and_bounds(condition_means)

        group_condition_means.append(group_mean)
        group_condition_lower_bounds.append(lower_bound)
        group_condition_higher_bounds.append(upper_bound)

    x_values = list(range(start_window, stop_window))
    period = 1.0 / frame_rate
    x_values = np.multiply(x_values, period)
    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot()

    n_conditions = len(group_condition_means)
    cmap = plt.get_cmap('plasma')
    for condition_index in range(n_conditions):
        colour = cmap(float(condition_index) / n_conditions)
        condition_mean = group_condition_means[condition_index]
        condition_lower_bound = group_condition_lower_bounds[condition_index]
        condition_higher_bound = group_condition_higher_bounds[condition_index]
        axis_1.plot(x_values, condition_mean, c=colour, label=str(rt_windows_list[condition_index][0]) + "_" + str(rt_windows_list[condition_index][1]))
        axis_1.fill_between(x=x_values, y1=condition_lower_bound, y2=condition_higher_bound, color=colour, alpha=0.5)

    axis_1.axvline(0, c='k', linestyle='dashed')

    for window in rt_windows_list:
        axis_1.axvline(float(window[0]) / 1000, c='b', linestyle='dashed', alpha=0.2)
    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylabel("Lick CD Projection")

    axis_1.spines[['right', 'top']].set_visible(False)



    plt.legend()
    plt.savefig(os.path.join(save_directory, "Lick_CD_By_RT.png"))
    plt.close()



def plot_lick_cd_by_rt(data_root, session_list, output_root):


    start_window = -12
    stop_window = 18
    rt_windows_list = [[500, 750],
                  [750, 1000],
                  [1000, 1250],
                  [1250, 1500]]
    n_rt_windows = len(rt_windows_list)

    # Iterate Through RT Windows
    window_means = []
    for rt_window_index in range(n_rt_windows):

        rt_window = rt_windows_list[rt_window_index]

        condition_means = []
        for session in session_list:

            # Load DF Matrix
            df_matrix = np.load(os.path.join(output_root, session, "df_over_f_matrix.npy"))

            # Load Frame Rate
            frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))

            # Load Lick CD
            lick_cd = np.load(os.path.join(output_root, session, "Lick_Coding", "Lick_Coding_Dimension.npy"))

            # Load RT Onsets
            onset_rt_matrix = np.load(os.path.join(output_root, session, "Behaviour", "Hits_By_RT.npy"))

            selected_onsets = get_selected_onsets(onset_rt_matrix, rt_window)
            if len(selected_onsets) > 0:

                # Get Tensors
                tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix, selected_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

                # Get Mean
                mean_activity = np.mean(tensor, axis=0)

                # Get Projection
                session_projection = np.dot(mean_activity, lick_cd)

                # Add To List
                condition_means.append(session_projection)

        window_means.append(condition_means)


    save_directory = os.path.join(output_root, "Group_Plots")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plot_graph(window_means, rt_windows_list, start_window, stop_window, frame_rate, save_directory)

