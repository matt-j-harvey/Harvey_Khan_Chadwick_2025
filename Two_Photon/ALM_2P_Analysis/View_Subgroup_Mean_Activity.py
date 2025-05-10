import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import Get_Data_Tensor
import Get_Mean_SEM_Bounds


def load_df_matrix(base_directory):
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)
    df_matrix = stats.zscore(df_matrix, axis=0)
    df_matrix = np.nan_to_num(df_matrix)
    return df_matrix



def view_subgroup_mean_activity(session_list, cell_subgroup, condition_onsets_list, start_window, stop_window, colour_list=["r, b"]):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(session_list[0], "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    n_conditions = len(condition_onsets_list)
    group_condition_means = []
    group_condition_lower_bounds = []
    group_condition_higher_bounds = []


    for condition_index in range(n_conditions):

        # Get Onsets File
        onsets_file = condition_onsets_list[condition_index]
        condition_means = []

        for base_directory in session_list:

           # Load DF Matrix
            df_matrix = load_df_matrix(base_directory)
            print("df matrix", np.shape(df_matrix))

            # Load Onsets
            onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file + ".npy"))

            # Load Selected Neurons
            selected_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", cell_subgroup + ".npy"))
            df_matrix = df_matrix[:, selected_neurons]
            print("df matrix", np.shape(df_matrix))

            # Get Tensors
            tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

            print("Tensor", np.shape(tensor))

            # Get Mean
            mean_activity = np.mean(tensor, axis=0)
            mean_activity = np.mean(mean_activity, axis=1)

           # Add To List
            condition_means.append(mean_activity)
            #plt.plot(mean_activity)
            #plt.show()

        condition_means = np.array(condition_means)
        group_mean, lower_bound, upper_bound = Get_Mean_SEM_Bounds.get_sem_and_bounds(condition_means)

        group_condition_means.append(group_mean)
        group_condition_lower_bounds.append(lower_bound)
        group_condition_higher_bounds.append(upper_bound)

    x_values = list(range(start_window, stop_window))
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot()

    n_conditions = len(group_condition_means)
    cmap=plt.get_cmap('jet')
    for condition_index in range(n_conditions):
        colour = cmap(float(condition_index) / n_conditions)
        condition_mean = group_condition_means[condition_index]
        condition_lower_bound = group_condition_lower_bounds[condition_index]
        condition_higher_bound = group_condition_higher_bounds[condition_index]
        axis_1.plot(x_values, condition_mean, c=colour, label=condition_onsets_list[condition_index])
        axis_1.fill_between(x=x_values, y1=condition_lower_bound, y2=condition_higher_bound, color=colour, alpha=0.5)

    plt.legend()
    plt.show()



def compare_subgroup_mean_acitivity(session_list,
                                    condition_onsets_file,
                                    start_window,
                                    stop_window,
                                    colour_list=["r", "b"],
                                    subgroup_list=["positive_cell_indexes", "negative_cell_indexes"]):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(session_list[0], "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    group_condition_means = []
    group_condition_lower_bounds = []
    group_condition_higher_bounds = []

    for cell_subgroup in subgroup_list:
        subgroup_mean_list = []

        for base_directory in session_list:

            # Load DF Matrix
            df_matrix = load_df_matrix(base_directory)

            # Load Onsets
            onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", condition_onsets_file + ".npy"))

            # Load Selected Neurons
            selected_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", cell_subgroup + ".npy"))
            df_matrix = df_matrix[:, selected_neurons]

            # Get Tensors
            tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onsets, start_window, stop_window,
                                                     baseline_correction=True, baseline_start=0, baseline_stop=5)

            print("Tensor", np.shape(tensor))

            # Get Mean
            mean_activity = np.mean(tensor, axis=0)
            mean_activity = np.mean(mean_activity, axis=1)

            # Add To List
            subgroup_mean_list.append(mean_activity)


        subgroup_mean_list = np.array(subgroup_mean_list)
        group_mean, lower_bound, upper_bound = Get_Mean_SEM_Bounds.get_sem_and_bounds(subgroup_mean_list)

        group_condition_means.append(group_mean)
        group_condition_lower_bounds.append(lower_bound)
        group_condition_higher_bounds.append(upper_bound)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, period)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot()

    n_subgroups = len(subgroup_list)

    for subgroup_index in range(n_subgroups):
        colour = colour_list[subgroup_index]
        condition_mean = group_condition_means[subgroup_index]
        condition_lower_bound = group_condition_lower_bounds[subgroup_index]
        condition_higher_bound = group_condition_higher_bounds[subgroup_index]
        axis_1.plot(x_values, condition_mean, c=colour, label=subgroup_list[subgroup_index])
        axis_1.fill_between(x=x_values, y1=condition_lower_bound, y2=condition_higher_bound, color=colour,
                            alpha=0.5)

    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylabel("Z Score DF/F")
    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_ylim([-0.4, 0.5])
    axis_1.spines['top'].set_visible(False)
    axis_1.spines['right'].set_visible(False)
    #plt.legend()
    plt.show()


start_window = -18
stop_window = 18


#cell_subgroup = "negative_cell_indexes"
#onsets_file = "odour_2_cued_onsets"
#onsets_file = "odour_context_stable_vis_2_onsets"


cell_subgroup = "positive_cell_indexes"
#cell_subgroup = "negative_cell_indexes"
#condition_onsets_list = ["visual_context_stable_vis_2_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "visual_context_false_alarms_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_2_onsets", "odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]
#condition_onsets_list = ["visual_context_stable_vis_1_onsets", "visual_context_stable_vis_2_onsets", "odour_context_stable_vis_1_onsets", "odour_context_stable_vis_2_onsets"]



root_directory = r"C:\Users\harveym\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"

session_list = [
    os.path.join(root_directory, r"65.2a\2024_08_05_Switching"),
    os.path.join(root_directory, r"65.2b\2024_07_31_Switching"),
    os.path.join(root_directory, r"67.3b\2024_08_09_Switching"),
    os.path.join(root_directory, r"67.3C\2024_08_20_Switching"),
    os.path.join(root_directory, r"69.2a\2024_08_12_Switching"),
    os.path.join(root_directory, r"72.3C\2024_09_10_Switching"),
]

#view_subgroup_mean_activity(session_list, cell_subgroup, condition_onsets_list, start_window, stop_window)
compare_subgroup_mean_acitivity(session_list,
                            "visual_context_stable_vis_2_onsets",
                                    start_window,
                                    stop_window)
