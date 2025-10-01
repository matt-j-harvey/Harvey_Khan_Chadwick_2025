import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt



def simmilarity_scatterplot(group_1_values, group_2_values, save_directory, title, labels):

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    # Plot Data
    n_mice = len(group_1_values)
    for mouse_index in range(n_mice):
        axis_1.plot([1, 2], [group_1_values[mouse_index], group_2_values[mouse_index]], c='tab:purple')
        axis_1.scatter([1, 2], [group_1_values[mouse_index], group_2_values[mouse_index]], c='tab:purple')

    # Set X Ticks
    axis_1.set_xticks([1, 2], labels=labels)

    # Set Title
    axis_1.set_title(title)
    axis_1.set_ylabel("Cosine Simmilarity")

    # Set Axis Limits
    axis_1.set_ylim([0, 1])
    axis_1.set_xlim([0.8, 2.2])
    #axis_1.set_yticks(list(range(0, 1, 10)))

    # Remove Borders
    axis_1.spines[['right', 'top']].set_visible(False)

    # Save Figure
    plt.savefig(os.path.join(save_directory, title + ".png"))
    plt.close()



def load_group_data(output_directory, session_list, comaprison_name):

    real_list = []
    shuffled_list = []

    for session in session_list:
        real_distance = np.load(os.path.join(output_directory, session, "Cosine_Simmilarity", comaprison_name, "Real_Distance.npy"))
        shuffled_distance = np.load(os.path.join(output_directory, session, "Cosine_Simmilarity", comaprison_name, "Shuffled_Distance.npy"))

        real_list.append(real_distance)
        shuffled_list.append(shuffled_distance)

    return real_list, shuffled_list




def plot_group_simmilarities(output_directory, session_list, save_directory):

    # Get Lick Comaprison Data
    real_list, shuffled_list = load_group_data(output_directory, session_list, "Lick")
    t_stat, p_values = stats.ttest_rel(real_list, shuffled_list)
    print("Lick Comparison", "t_stat", t_stat, "p_values", p_values)
    simmilarity_scatterplot(real_list, shuffled_list, save_directory, "Lick Dimensions", ["Real Distance", "Shuffled Distance"])


    # Get Choice Dimension Comparison
    real_list, shuffled_list = load_group_data(output_directory, session_list, "Choice_Dimensions")
    t_stat, p_values = stats.ttest_rel(real_list, shuffled_list)
    print("Choice Comparison", "t_stat", t_stat, "p_values", p_values)
    simmilarity_scatterplot(real_list, shuffled_list, save_directory, "Choice_Dimensions", ["Real Distance", "Shuffled Distance"])
