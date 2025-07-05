import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
from scipy import stats


def load_group_results(output_directory, session_list, matrix_type):

    group_list = []
    for session in session_list:

        # Load Data
        session_data = np.load(os.path.join(output_directory, session, "Recurrent Amplification", "Stimuli_Weight_Interactions", matrix_type + "_Interaction.npy"))

        # Add To Group List
        group_list.append(session_data)

    return group_list



def get_mean_and_bounds(group_results):

    group_mean = np.mean(group_results, axis=0)
    group_sem = stats.sem(group_results, axis=0)

    # Get Bounds
    group_upper_bound = np.add(group_mean, group_sem)
    group_lower_bound = np.subtract(group_mean, group_sem)

    return group_mean, group_upper_bound, group_lower_bound



def plot_scatters(mvar_output_directory, session_list):

    stimuli_list = [
        "Rewarded - Relevant",
        "Unrewarded - Relevant",
        "Rewarded - Irrelevant",
        "Unrewarded - Irrelevant"
    ]


    # Load Results
    group_diagonal = load_group_results(mvar_output_directory, session_list, "diagonal_weights")
    group_recurrent = load_group_results(mvar_output_directory, session_list, "recurrent_weights")
    group_shuffled = load_group_results(mvar_output_directory, session_list, "shuffled_recurrent_weights")
    print("group_diagonal", np.shape(group_diagonal))

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)
    group_shuffled = np.squeeze(group_shuffled)

    # Sum Over Time
    group_diagonal = np.sum(group_diagonal, axis=2)
    group_recurrent = np.sum(group_recurrent, axis=2)
    group_shuffled = np.sum(group_shuffled, axis=2)

    # Create X Bins
    jitter_size = 0.05
    x_positions = np.array([0, 1, 2])

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, "Stimuli Weight Interactions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Plot Data
    n_stimuli = 4
    n_mice = np.shape(group_diagonal)[0]
    colourmap = cm.get_cmap("winter")
    for stimulus_index in range(n_stimuli):

        # Create Figure
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1, 1, 1)

        for mouse_index in range(n_mice):

            # Get Mouse Colour
            mouse_colour = colourmap(float(mouse_index)/n_mice)

            mouse_jitter = np.random.uniform(low=-jitter_size, high=jitter_size, size=3)
            x_values = np.add(mouse_jitter, x_positions)

            y_values = [group_diagonal[mouse_index, stimulus_index],
                        group_recurrent[mouse_index, stimulus_index],
                        group_shuffled[mouse_index, stimulus_index]]

            axis_1.plot(x_values, y_values, alpha=0.4, c=mouse_colour)
            axis_1.scatter(x_values, y_values, c=mouse_colour)


        t_stat, p_value = stats.ttest_rel(group_diagonal[:, stimulus_index],group_recurrent[:, stimulus_index], axis=0)
        print("recurrent v diagonal", "t_stat", t_stat, "p_value", p_value)

        t_stat, p_value = stats.ttest_rel(group_shuffled[:, stimulus_index],group_recurrent[:, stimulus_index], axis=0)
        print("recurrent v shuffle", "t_stat", t_stat, "p_value", p_value)

        # Remove Splines
        axis_1.spines[['right', 'top']].set_visible(False)

        # Set X Ticks
        axis_1.set_xticks([0, 1, 2], labels=["Diagonal Only", "Recurrent", "Recurrent Shuffled"])

        # Set X Axis Extent
        axis_1.set_xlim([-0.5, 2.5])

        # Set Y Label
        axis_1.set_ylabel("Total lick CD projection")

        # Set Title
        axis_1.set_title(stimuli_list[stimulus_index])

        plt.savefig(os.path.join(save_directory, stimuli_list[stimulus_index] + ".png"))
        plt.close()


def plot_session_interactions(output_directory):

    # Load Data
    diagonal_data = np.load(os.path.join(output_directory,"Stimuli_Weight_Interactions", "diagonal_weights_Interaction.npy"))
    recurrent_data = np.load(os.path.join(output_directory,"Stimuli_Weight_Interactions", "recurrent_weights_Interaction.npy"))
    shuffled_data = np.load(os.path.join(output_directory, "Stimuli_Weight_Interactions", "shuffled_recurrent_weights_Interaction.npy"))

    # Create Figure
    figure_1 = plt.figure(figsize=(20, 5))
    diagonal_axis = figure_1.add_subplot(1, 3, 1)
    recurrent_axis = figure_1.add_subplot(1, 3, 2)
    shuffled_axis = figure_1.add_subplot(1, 3, 3)

    # Plot Data
    n_stimuli = 4
    for stimulus_index in range(n_stimuli):
        diagonal_axis.plot(diagonal_data[stimulus_index])
        recurrent_axis.plot(recurrent_data[stimulus_index])
        shuffled_axis.plot(shuffled_data[stimulus_index])

    plt.savefig(os.path.join(output_directory, "Stimuli_Weight_Interactions.png"))
    plt.close()
    print("diagonal data", np.shape(diagonal_data))


def plot_group_interactions(mvar_output_directory, session_list, colour_list=["b", "r", "g", "m"]):

    # Load Results
    group_diagonal = load_group_results(mvar_output_directory, session_list, "diagonal_weights")
    group_recurrent = load_group_results(mvar_output_directory, session_list, "recurrent_weights")
    group_shuffled = load_group_results(mvar_output_directory, session_list, "shuffled_recurrent_weights")
    print("group_diagonal", np.shape(group_diagonal))

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)
    group_shuffled = np.squeeze(group_shuffled)

    # Get Mean and Bounds
    diagonal_mean, diagonal_upper_bound, diagonal_lower_bound = get_mean_and_bounds(group_diagonal)
    recurrent_mean, recurrent_upper_bound, recurrent_lower_bound = get_mean_and_bounds(group_recurrent)
    shuffled_mean, shuffled_upper_bound, shuffled_lower_bound = get_mean_and_bounds(group_shuffled)

    # Create Figure
    figure_1 = plt.figure(figsize=(20,5))
    diagonal_axis = figure_1.add_subplot(1,3,1)
    recurrent_axis = figure_1.add_subplot(1,3,2)
    shuffled_axis = figure_1.add_subplot(1,3,3)

    # Get X Values
    n_timepoints = np.shape(group_diagonal)[2]
    x_values = list(range(0, n_timepoints))

    # Get Magnitude
    max_value = np.max(np.concatenate([diagonal_upper_bound, recurrent_upper_bound, shuffled_upper_bound])) * 1.2
    min_value = np.min(np.concatenate([diagonal_lower_bound, recurrent_lower_bound, shuffled_lower_bound])) * 1.2

    # Plot Data
    n_stimuli = 4
    for stimulus_index in range(n_stimuli):

        diagonal_axis.plot(x_values, diagonal_mean[stimulus_index], c=colour_list[stimulus_index])
        recurrent_axis.plot(x_values, recurrent_mean[stimulus_index], c=colour_list[stimulus_index])
        shuffled_axis.plot(x_values, shuffled_mean[stimulus_index], c=colour_list[stimulus_index])

        diagonal_axis.fill_between(x=x_values, y1=diagonal_lower_bound[stimulus_index], y2=diagonal_upper_bound[stimulus_index], alpha=0.2, color=colour_list[stimulus_index])
        recurrent_axis.fill_between(x=x_values, y1=recurrent_lower_bound[stimulus_index], y2=recurrent_upper_bound[stimulus_index], alpha=0.2, color=colour_list[stimulus_index])
        shuffled_axis.fill_between(x=x_values, y1=shuffled_lower_bound[stimulus_index], y2=shuffled_upper_bound[stimulus_index], alpha=0.2, color=colour_list[stimulus_index])


    diagonal_axis.set_ylim([min_value, max_value])
    recurrent_axis.set_ylim([min_value, max_value])
    shuffled_axis.set_ylim([min_value, max_value])

    diagonal_axis.set_title("Diagonal weights")
    recurrent_axis.set_title("With Recurrent weights")
    shuffled_axis.set_title("Shuffled Recurrent weights")

    diagonal_axis.set_xlabel("Time")
    recurrent_axis.set_xlabel("Time")
    shuffled_axis.set_xlabel("Time")

    diagonal_axis.set_ylabel("Lick CD Projection")
    recurrent_axis.set_ylabel("Lick CD Projection")
    shuffled_axis.set_ylabel("Lick CD Projection")

    # Remove Splines
    diagonal_axis.spines[['right', 'top']].set_visible(False)
    recurrent_axis.spines[['right', 'top']].set_visible(False)
    shuffled_axis.spines[['right', 'top']].set_visible(False)

    # Save Figure
    save_directory = os.path.join(mvar_output_directory, "Stimuli Weight Interactions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Weight_Amplification_Comparisons.png"))
    plt.close()


