import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import os
from scipy import stats
import matplotlib as mpl



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

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)

    # Sum Over Time
    group_diagonal = np.sum(group_diagonal, axis=2)
    group_recurrent = np.sum(group_recurrent, axis=2)

    # Create X Bins
    jitter_size = 0.05
    x_positions = np.array([0, 1])

    # Create Save Directory
    save_directory = os.path.join(mvar_output_directory, "Group_Results", "Stimuli Weight Interactions")
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

            mouse_jitter = np.random.uniform(low=-jitter_size, high=jitter_size, size=2)
            x_values = np.add(mouse_jitter, x_positions)

            y_values = [group_diagonal[mouse_index, stimulus_index],
                        group_recurrent[mouse_index, stimulus_index]]

            axis_1.plot(x_values, y_values, alpha=0.4, c=mouse_colour)
            axis_1.scatter(x_values, y_values, c=mouse_colour)


        #t_stat, p_value = stats.ttest_rel(group_diagonal[:, stimulus_index],group_recurrent[:, stimulus_index], axis=0)
        #print("recurrent v diagonal", "t_stat", t_stat, "p_value", p_value)

        # Remove Splines
        axis_1.spines[['right', 'top']].set_visible(False)

        # Set X Ticks
        axis_1.set_xticks([0, 1], labels=["Diagonal Only", "Recurrent"])

        # Set X Axis Extent
        axis_1.set_xlim([-0.5, 1.5])

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

    # Create Figure
    figure_1 = plt.figure(figsize=(20, 5))
    diagonal_axis = figure_1.add_subplot(1, 2, 1)
    recurrent_axis = figure_1.add_subplot(1, 2, 2)

    # Plot Data
    n_stimuli = 4
    for stimulus_index in range(n_stimuli):
        diagonal_axis.plot(diagonal_data[stimulus_index])
        recurrent_axis.plot(recurrent_data[stimulus_index])

    plt.savefig(os.path.join(output_directory, "Stimuli_Weight_Interactions.png"))
    plt.close()


def plot_group_interactions(mvar_output_directory, session_list, colour_list=["b", "r", "g", "m"]):

    # Load Results
    group_diagonal = load_group_results(mvar_output_directory, session_list, "diagonal_weights")
    group_recurrent = load_group_results(mvar_output_directory, session_list, "recurrent_weights")

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)

    # Get Mean and Bounds
    diagonal_mean, diagonal_upper_bound, diagonal_lower_bound = get_mean_and_bounds(group_diagonal)
    recurrent_mean, recurrent_upper_bound, recurrent_lower_bound = get_mean_and_bounds(group_recurrent)

    # Create Figure
    figure_1 = plt.figure(figsize=(20,5))
    diagonal_axis = figure_1.add_subplot(1,2,1)
    recurrent_axis = figure_1.add_subplot(1,2,2)

    # Get X Values
    n_timepoints = np.shape(group_diagonal)[2]
    x_values = list(range(0, n_timepoints))

    # Get Magnitude
    max_value = np.max(np.concatenate([diagonal_upper_bound, recurrent_upper_bound])) * 1.2
    min_value = np.min(np.concatenate([diagonal_lower_bound, recurrent_lower_bound])) * 1.2

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
    save_directory = os.path.join(mvar_output_directory, "Group_Results", "Stimuli Weight Interactions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Weight_Amplification_Comparisons.png"))
    plt.close()





def plot_total_interactions(mvar_output_directory, session_list):

    # Load Results
    group_diagonal = load_group_results(mvar_output_directory, session_list, "diagonal_weights")
    group_recurrent = load_group_results(mvar_output_directory, session_list, "recurrent_weights")

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)

    # Sum Over Time
    group_diagonal = np.sum(group_diagonal, axis=2)
    group_recurrent = np.sum(group_recurrent, axis=2)

    # Get Mean Across Group
    mean_group_diagonal = np.mean(group_diagonal, axis=0)
    mean_group_recurrent = np.mean(group_recurrent, axis=0)

    # Plot These
    figure_1 = plt.figure(figsize=(7, 7))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    bar_width = 0.2
    bar_jitter = bar_width / 2
    spacing = 0.5

    # Rewarded Relevant
    axis_1.bar((1 * spacing) - bar_jitter, mean_group_diagonal[0], width=bar_width, edgecolor='Blue', facecolor='white', alpha=0.5, linewidth=2)
    axis_1.bar((1 * spacing) + bar_jitter, mean_group_recurrent[0], width=bar_width, edgecolor='Blue',  facecolor='Blue', alpha=0.5, linewidth=2)

    # Unrewarded Relevant
    axis_1.bar((2 * spacing) - bar_jitter, mean_group_diagonal[2],  width=bar_width, edgecolor='Green', facecolor='white', alpha=0.5, linewidth=2)
    axis_1.bar((2 * spacing) + bar_jitter, mean_group_recurrent[2], width=bar_width, edgecolor='Green',  facecolor='Green', alpha=0.5, linewidth=2)

    # Rewarded Irrelevant
    axis_1.bar((3 * spacing) - bar_jitter, mean_group_diagonal[1],  width=bar_width, edgecolor='Red', facecolor='white', alpha=0.5, linewidth=2)
    axis_1.bar((3 * spacing) + bar_jitter, mean_group_recurrent[1], width=bar_width, edgecolor='Red',  facecolor='Red', alpha=0.5, linewidth=2)

    # Unrewarded Irrelevant
    axis_1.bar((4 * spacing) - bar_jitter, mean_group_diagonal[3],  width=bar_width, edgecolor='Purple', facecolor='white', alpha=0.5, linewidth=2)
    axis_1.bar((4 * spacing) + bar_jitter, mean_group_recurrent[3], width=bar_width, edgecolor='Purple',  facecolor='Purple', alpha=0.5, linewidth=2)


    # Add Scatterplots
    n_mice = len(group_diagonal)
    for mouse_index in range(n_mice):

        axis_1.scatter([(1 * spacing) - bar_jitter, (1 * spacing) + bar_jitter], [group_diagonal[mouse_index][0], group_recurrent[mouse_index][0]], c="Blue", alpha=0.4)
        axis_1.plot([(1 * spacing) - bar_jitter, (1 * spacing) + bar_jitter], [group_diagonal[mouse_index][0], group_recurrent[mouse_index][0]], c="Blue", alpha=0.4)

        axis_1.scatter([(2 * spacing) - bar_jitter, (2 * spacing) + bar_jitter], [group_diagonal[mouse_index][2], group_recurrent[mouse_index][2]], c="Green", alpha=0.4)
        axis_1.plot([(2 * spacing) - bar_jitter, (2 * spacing) + bar_jitter], [group_diagonal[mouse_index][2], group_recurrent[mouse_index][2]], c="Green", alpha=0.4)

        axis_1.scatter([(3 * spacing) - bar_jitter, (3 * spacing) + bar_jitter], [group_diagonal[mouse_index][1], group_recurrent[mouse_index][1]], c="Red", alpha=0.4)
        axis_1.plot([(3 * spacing) - bar_jitter, (3 * spacing) + bar_jitter], [group_diagonal[mouse_index][1], group_recurrent[mouse_index][1]], c="Red", alpha=0.4)

        axis_1.scatter([(4 * spacing) - bar_jitter, (4 * spacing) + bar_jitter], [group_diagonal[mouse_index][3], group_recurrent[mouse_index][3]], c="Purple", alpha=0.4)
        axis_1.plot([(4 * spacing) - bar_jitter, (4 * spacing) + bar_jitter], [group_diagonal[mouse_index][3], group_recurrent[mouse_index][3]], c="Purple", alpha=0.4)


    #axis_1.set_xlabel("Stimuli")
    axis_1.set_ylabel("Total Lick CD Projection")

    # Remove Splines
    axis_1.spines[['right', 'top']].set_visible(False)

    #axis_1.axhline(0, c='Grey', alpha=0.4)

    # Set Ticks
    x_values = list(range(1, 5))
    x_values = np.multiply(x_values, spacing)
    axis_1.set_xticks(x_values, labels=["Rewarded \nRelevant", "Rewarded \nIrrelevant", "Unrewarded \nRelevant", "Unrewarded \nIrrelevant"])


    # Perform T Tests
    stim_list = ["Rewarded Relevant", "Unrewarded Relevant", "Rewarded Irrelevant", "Unrewarded Irrelevant"]
    print("Total Interaction T Stats")
    for stimuli_index in range(4):
        t_stat, p_value = stats.ttest_rel(group_diagonal[:, stimuli_index], group_recurrent[:, stimuli_index])
        t_stat = np.around(t_stat, 3)
        p_value = np.around(p_value, 3)
        print(stim_list[stimuli_index], "T stat: ", t_stat, "P value ", p_value)

    # Save Figure
    save_directory = os.path.join(mvar_output_directory, "Group_Results", "Stimuli Weight Interactions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Summed_Interaction_Plot.png"))
    plt.close()



def compare_modulation_interaction(mvar_output_directory, session_list):

    # Load Results
    group_diagonal = load_group_results(mvar_output_directory, session_list, "diagonal_weights")
    group_recurrent = load_group_results(mvar_output_directory, session_list, "recurrent_weights")

    # Squeeze!
    group_diagonal = np.squeeze(group_diagonal)
    group_recurrent = np.squeeze(group_recurrent)

    # Sum Over Time
    group_diagonal = np.sum(group_diagonal, axis=2)
    group_recurrent = np.sum(group_recurrent, axis=2)

    # Get Amplification
    amplification = np.subtract(group_recurrent, group_diagonal)

    # Extract Vis 1
    rel_amplification = amplification[:, 0]
    irrel_amplification = amplification[:, 2]

    # Get Mean
    mean_rel_amplification = np.mean(rel_amplification, axis=0)
    mean_irrel_amplification = np.mean(irrel_amplification, axis=0)

    # Test Significance
    t_stat, p_value = stats.ttest_rel(rel_amplification, irrel_amplification)
    print("Modulation t stat", t_stat, "p value", p_value)

    # Plot These
    figure_1 = plt.figure(figsize=(4, 7))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    bar_width = 0.2


    # Rewarded Relevant
    axis_1.bar(0.9, mean_irrel_amplification, width=bar_width, edgecolor='Green', facecolor="White", alpha=0.5, linewidth=2, hatch="/")
    axis_1.bar(1.1, mean_rel_amplification, width=bar_width, edgecolor='Blue', facecolor="White", alpha=0.5, linewidth=2, hatch="/")
    axis_1.set_xlim([0.7, 1.3])

    mpl.rcParams['hatch.linewidth'] = 8.0

    # Add Scatterplots
    n_mice = len(rel_amplification)
    for mouse_index in range(n_mice):
        axis_1.scatter([0.9, 1.1], [irrel_amplification[mouse_index], rel_amplification[mouse_index]], c="Black", alpha=0.8)
        axis_1.plot([0.9, 1.1], [irrel_amplification[mouse_index], rel_amplification[mouse_index]], c="Black", alpha=0.8)

    # axis_1.set_xlabel("Stimuli")
    axis_1.set_ylabel("Total Lick CD Projection")

    # Remove Splines
    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_xticks([0.9, 1.1], labels=["Irrelevant", "Relevant"])

    # Save Figure
    save_directory = os.path.join(mvar_output_directory, "Group_Results", "Stimuli Weight Interactions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Modulation_Interaction.png"))
    plt.close()