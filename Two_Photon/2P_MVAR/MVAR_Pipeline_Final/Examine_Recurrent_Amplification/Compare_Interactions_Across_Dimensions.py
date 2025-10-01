import matplotlib.pyplot as plt
from scipy import stats
import sigfig
import numpy as np
import os


def get_mean_and_bounds(group_results):

    group_mean = np.mean(group_results, axis=0)
    group_sem = stats.sem(group_results, axis=0)

    # Get Bounds
    group_upper_bound = np.add(group_mean, group_sem)
    group_lower_bound = np.subtract(group_mean, group_sem)

    return group_mean, group_upper_bound, group_lower_bound


def load_group_results(output_directory, session_list, dimension, matrix_type):

    group_list = []
    for session in session_list:

        # Load Data
        session_data = np.load(os.path.join(output_directory, session, "Recurrent Amplification", "Stimuli_Weight_Interactions_Comparisons", dimension + "_" + matrix_type + "_Interaction.npy"))

        # Add To Group List
        group_list.append(session_data)

    return group_list


def get_integrated_interaction(stimulus_vector, recurrent_weights, duration=9):
    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_state = np.zeros(n_neurons)
    for x in range(duration):
        trial_vector.append(current_state)
        current_state = np.matmul(recurrent_weights, current_state)
        current_state = np.add(current_state, stimulus_vector)

    trial_vector = np.array(trial_vector)
    return trial_vector



def plot_group_interactions_sum(mvar_output_directory, session_list, dimension_list, plot_save_directory, selected_dimension, colour, title):

    dimension_sum_list_diagonal = []
    dimension_sum_list_recurrent = []

    for dimension in dimension_list:

        # Load Results
        group_diagonal = load_group_results(mvar_output_directory, session_list, dimension, "diagonal_weights")
        group_recurrent = load_group_results(mvar_output_directory, session_list, dimension, "recurrent_weights")
        print("group_diagonal", np.shape(group_diagonal))
        print("group_recurrent", np.shape(group_recurrent))

        # Squeeze!
        group_diagonal = np.squeeze(group_diagonal)
        group_recurrent = np.squeeze(group_recurrent)
        print("group_diagonal", np.shape(group_diagonal))
        print("group_recurrent", np.shape(group_recurrent))

        # Average Across Time
        group_diagonal = np.sum(group_diagonal, axis=2)
        group_recurrent = np.sum(group_recurrent, axis=2)
        print("group_diagonal", np.shape(group_diagonal))
        print("group_recurrent", np.shape(group_recurrent))

        dimension_sum_list_diagonal.append(group_diagonal)
        dimension_sum_list_recurrent.append(group_recurrent)


    dimension_sum_list_diagonal = np.array(dimension_sum_list_diagonal)
    dimension_sum_list_recurrent = np.array(dimension_sum_list_recurrent)

    print("dimension_sum_list_diagonal", np.shape(dimension_sum_list_diagonal))
    print("dimension_sum_list_recurrent", np.shape(dimension_sum_list_recurrent))


    # Plot Only Vis 1 Loading
    selected_diagonal = dimension_sum_list_diagonal[:, :, selected_dimension]
    selected_recurrent = dimension_sum_list_recurrent[:, :, selected_dimension]
    print("selected_diagonal", np.shape(selected_diagonal))
    print("selected_recurrent", np.shape(selected_recurrent))

    figure_1 = plt.figure(figsize=(15, 5))
    axis_1 = figure_1.add_subplot(1,1,1)

    n_dimensions = len(dimension_list)
    x_values = list(range(n_dimensions))

    for dimension_index in range(n_dimensions):

        # Get Mean
        diagonal_mean = np.mean(selected_diagonal[dimension_index])
        recurrent_mean = np.mean(selected_recurrent[dimension_index])

        # Get X Values
        diagonal_x  = x_values[dimension_index]
        recurrent_x = diagonal_x + 0.2
        axis_1.bar(diagonal_x, diagonal_mean, width=0.1, color='Grey', alpha=0.5)
        axis_1.bar(recurrent_x, recurrent_mean, width=0.1, color=colour, alpha=0.5)

        # Plot Each Mouse
        n_mice = len(session_list)
        for mouse_index in range(n_mice):
            axis_1.scatter([diagonal_x, recurrent_x], [selected_diagonal[dimension_index][mouse_index], selected_recurrent[dimension_index][mouse_index]],  c='cornflowerblue', alpha=0.3)
            axis_1.plot([diagonal_x, recurrent_x], [selected_diagonal[dimension_index][mouse_index], selected_recurrent[dimension_index][mouse_index]],  c='cornflowerblue', alpha=0.3)

        # Test Signficance
        t_stat, p_value = stats.ttest_rel(selected_diagonal[dimension_index], selected_recurrent[dimension_index])
        p_value = sigfig.round(p_value, 2)
        print("Dimension: ", dimension_list[dimension_index], "P Value", p_value)


    # Remove Splines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_ylabel("Dimension Projection")

    axis_1.set_xticks(x_values,dimension_list)
    axis_1.set_ylim([-20, 32])

    figure_1.suptitle(title)

    plt.savefig(os.path.join(plot_save_directory, title + "_Sum_Comparison.png"))
    plt.show()




def plot_differences(mvar_output_directory, session_list, dimension_list, plot_save_directory):

    dimension_sum_list_diagonal = []
    dimension_sum_list_recurrent = []

    for dimension in dimension_list:

        # Load Results
        group_diagonal = load_group_results(mvar_output_directory, session_list, dimension, "diagonal_weights")
        group_recurrent = load_group_results(mvar_output_directory, session_list, dimension, "recurrent_weights")
        print("group_diagonal", np.shape(group_diagonal))
        print("group_recurrent", np.shape(group_recurrent))

        # Squeeze!
        group_diagonal = np.squeeze(group_diagonal)
        group_recurrent = np.squeeze(group_recurrent)
        print("group_diagonal", np.shape(group_diagonal))
        print("group_recurrent", np.shape(group_recurrent))

        # Average Across Time
        group_diagonal = np.sum(group_diagonal, axis=2)
        group_recurrent = np.sum(group_recurrent, axis=2)
        print("group_diagonal", np.shape(group_diagonal))
        print("group_recurrent", np.shape(group_recurrent))

        dimension_sum_list_diagonal.append(group_diagonal)
        dimension_sum_list_recurrent.append(group_recurrent)

    dimension_sum_list_diagonal = np.array(dimension_sum_list_diagonal)
    dimension_sum_list_recurrent = np.array(dimension_sum_list_recurrent)

    print("dimension_sum_list_diagonal", np.shape(dimension_sum_list_diagonal))
    print("dimension_sum_list_recurrent", np.shape(dimension_sum_list_recurrent))

    # Get Rewarded rel v Irrel
    visual_context_diagonal = dimension_sum_list_diagonal[:, :, 0]
    visual_context_recurrent = dimension_sum_list_recurrent[:, :, 0]
    visual_context_diff = np.subtract(visual_context_recurrent, visual_context_diagonal)
    print("visual context diff", np.shape(visual_context_diff))

    odour_context_diagonal = dimension_sum_list_diagonal[:, :, 2]
    odour_context_recurrent = dimension_sum_list_recurrent[:, :, 2]
    odour_context_diff = np.subtract(odour_context_recurrent, odour_context_diagonal)
    print("Odour context diff", np.shape(odour_context_diff))

    figure_1 = plt.figure(figsize=(15, 5))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    n_dimensions = len(dimension_list)
    x_values = list(range(n_dimensions))

    for dimension_index in range(n_dimensions):

        # Get Means
        visual_context_mean = np.mean(visual_context_diff[dimension_index])
        odour_context_mean = np.mean(odour_context_diff[dimension_index])

        # Get X Values
        odour_x = x_values[dimension_index]
        visual_x = odour_x + 0.2

        axis_1.bar(odour_x, odour_context_mean, width=0.1, color='Green', alpha=0.5)
        axis_1.bar(visual_x, visual_context_mean, width=0.1, color='Blue', alpha=0.5)


        # Plot Each Mouse
        n_mice = len(session_list)
        for mouse_index in range(n_mice):
            axis_1.scatter([odour_x, visual_x], [odour_context_diff[dimension_index][mouse_index], visual_context_diff[dimension_index][mouse_index]], c='cornflowerblue', alpha=0.3)
            axis_1.plot([odour_x, visual_x], [odour_context_diff[dimension_index][mouse_index], visual_context_diff[dimension_index][mouse_index]], c='cornflowerblue', alpha=0.3)


        # Test Signficance
        t_stat, p_value = stats.ttest_rel(odour_context_diff[dimension_index], visual_context_diff[dimension_index])
        p_value = sigfig.round(p_value, 2)
        print("Dimension: ", dimension_list[dimension_index], "P Value", p_value)


    # Remove Splines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_ylabel("Dimension Projection")

    axis_1.set_xticks(x_values, dimension_list)
    #axis_1.set_ylim([-10, 25])

    figure_1.suptitle("Differences")

    #plt.savefig(os.path.join(plot_save_directory, title + "_Sum_Comparison.png"))
    plt.show()



def plot_group_interactions(mvar_output_directory, session_list, dimension, save_directory, colour_list=["b", "r", "g", "m"]):

    # Load Results
    group_diagonal = load_group_results(mvar_output_directory, session_list, dimension, "diagonal_weights")
    group_recurrent = load_group_results(mvar_output_directory, session_list, dimension, "recurrent_weights")
    print("group_diagonal", np.shape(group_diagonal))

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)

    # Get Mean and Bounds
    diagonal_mean, diagonal_upper_bound, diagonal_lower_bound = get_mean_and_bounds(group_diagonal)
    recurrent_mean, recurrent_upper_bound, recurrent_lower_bound = get_mean_and_bounds(group_recurrent)

    # Create Figure
    figure_1 = plt.figure(figsize=(5,10))
    diagonal_axis = figure_1.add_subplot(2,1,1)
    recurrent_axis = figure_1.add_subplot(2,1,2)

    # Get X Values
    n_timepoints = np.shape(group_diagonal)[2]
    x_values = list(range(0, n_timepoints))

    # Get Magnitude
    max_value = np.max(np.concatenate([diagonal_upper_bound, recurrent_upper_bound])) * 1.2
    min_value = np.min(np.concatenate([diagonal_lower_bound, recurrent_lower_bound])) * 1.2
    max_value = 5
    min_value = -5


    # Plot Data
    n_stimuli = 4
    for stimulus_index in range(n_stimuli):

        diagonal_axis.plot(x_values, diagonal_mean[stimulus_index], c=colour_list[stimulus_index])
        recurrent_axis.plot(x_values, recurrent_mean[stimulus_index], c=colour_list[stimulus_index])

        diagonal_axis.fill_between(x=x_values, y1=diagonal_lower_bound[stimulus_index], y2=diagonal_upper_bound[stimulus_index], alpha=0.2, color=colour_list[stimulus_index])
        recurrent_axis.fill_between(x=x_values, y1=recurrent_lower_bound[stimulus_index], y2=recurrent_upper_bound[stimulus_index], alpha=0.2, color=colour_list[stimulus_index])

    diagonal_axis.set_ylim([min_value, max_value])
    recurrent_axis.set_ylim([min_value, max_value])

    diagonal_axis.set_title("Diagonal weights")
    recurrent_axis.set_title("With Recurrent weights")

    diagonal_axis.set_xlabel("Time")
    recurrent_axis.set_xlabel("Time")

    diagonal_axis.set_ylabel("Lick CD Projection")
    recurrent_axis.set_ylabel("Lick CD Projection")


    # Remove Splines
    diagonal_axis.spines[['right', 'top']].set_visible(False)
    recurrent_axis.spines[['right', 'top']].set_visible(False)

    # Save Figure
    figure_1.suptitle(dimension)
    plt.savefig(os.path.join(save_directory, dimension + "_Weight_Amplification_Comparisons.png"))
    plt.close()



def get_stimuli_recurrent_interactions(mvar_output_directory, dimensions_output_directory, output_vector_file, weight_matrix_file):


    stimuli_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
    ]

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, "Recurrent Amplification", "Stimuli_Weight_Interactions_Comparisons")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Load Lick CD
    lick_cd = np.load(os.path.join(dimensions_output_directory, "Coding_Dimensions", output_vector_file + "_dimension.npy"))
    lick_cd = np.expand_dims(lick_cd, axis=1)

    # Load Weight Matrix
    weight_matrix = np.load(os.path.join(mvar_output_directory, "Recurrent Amplification", "Weight_Matricies", weight_matrix_file + ".npy"))

    interaction_list = []
    for stimulus in stimuli_list:

        # Load Vector
        stimulus_vector = np.load(os.path.join(mvar_output_directory,"Recurrent Amplification", "Stimuli Vectors", stimulus + ".npy"))

        # Get Interaction
        stimuli_weight_interaction = get_integrated_interaction(stimulus_vector, weight_matrix)

        # Get Lick CD
        stimuli_weight_interaction_lick_cd = np.dot(stimuli_weight_interaction, lick_cd)

        # Add To List
        interaction_list.append(stimuli_weight_interaction_lick_cd)

    # Save List
    np.save(os.path.join(save_directory, output_vector_file + "_" + weight_matrix_file + "_Interaction.npy"), interaction_list)




output_dimension_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Output_Vector_Analysis"
#mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results_Positive_Only"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

dimension_list = [
    "odour_lick",
    "odour_context_choice",
    "combined_lick",
    "visual_context_choice",
    "visual_lick",
]

# Get Recurrent Interactions For Each Dimension
for session in control_session_list:
    mvar_output_directory = os.path.join(mvar_output_root, session)
    dimension_output_directory = os.path.join(output_dimension_root, session)

    for dimension in dimension_list:

        get_stimuli_recurrent_interactions(mvar_output_directory, dimension_output_directory, dimension, "diagonal_weights")
        get_stimuli_recurrent_interactions(mvar_output_directory, dimension_output_directory, dimension, "recurrent_weights")



# Plot Group Results
group_plot_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results_Positive_Only\Stimuli Weight Interactions\Dimension_Comparisons"

plot_differences(mvar_output_root, control_session_list, dimension_list, group_plot_directory)

plot_group_interactions_sum(mvar_output_root, control_session_list, dimension_list, group_plot_directory, selected_dimension=0, colour="slateblue", title="Rewarded Relevant")
plot_group_interactions_sum(mvar_output_root, control_session_list, dimension_list, group_plot_directory, selected_dimension=1, colour='red', title="Unrewarded Relevant")
plot_group_interactions_sum(mvar_output_root, control_session_list, dimension_list, group_plot_directory, selected_dimension=2, colour='green', title="Rewarded Irrelevant")


plot_group_interactions(mvar_output_root, control_session_list, dimension_list[0], group_plot_directory, colour_list=["b", "r", "g", "m"])
plot_group_interactions(mvar_output_root, control_session_list, dimension_list[1], group_plot_directory, colour_list=["b", "r", "g", "m"])
plot_group_interactions(mvar_output_root, control_session_list, dimension_list[2], group_plot_directory, colour_list=["b", "r", "g", "m"])
plot_group_interactions(mvar_output_root, control_session_list, dimension_list[3], group_plot_directory, colour_list=["b", "r", "g", "m"])
plot_group_interactions(mvar_output_root, control_session_list, dimension_list[4], group_plot_directory, colour_list=["b", "r", "g", "m"])