import matplotlib.pyplot as plt
import numpy as np
import os
import Behaviour_Analysis_Functions
import Session_List


def combined_behaviour_plot_pipeline(data_root, nested_session_list):

    # Create Empty List
    group_visual_list = []
    group_odour_list = []
    group_irrel_list = []

    # Iterate Through Mice
    for mouse in nested_session_list:
        mouse_visual_list = []
        mouse_odour_list = []
        mouse_irrel_list = []

        # Iterate Through Sessions
        for session in mouse:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
            #print("behaviour_matrix.npy", np.shape(behaviour_matrix))

            # Get Performance
            session_visual_performance = Behaviour_Analysis_Functions.get_session_visual_performance(behaviour_matrix)
            session_odour_performance = Behaviour_Analysis_Functions.get_session_odour_performance(behaviour_matrix)
            session_irrel_performance = Behaviour_Analysis_Functions.get_session_irrel_performance(behaviour_matrix)

            mouse_visual_list.append(session_visual_performance)
            mouse_odour_list.append(session_odour_performance)
            mouse_irrel_list.append(session_irrel_performance)

            print("session", session, "visual performance", session_visual_performance, "irrel performance", session_irrel_performance)

        # Get Mouse Average
        mouse_visual = np.mean(mouse_visual_list)
        mouse_odour = np.mean(mouse_odour_list)
        mouse_irrel = np.mean(mouse_irrel_list)

        print("Mouse", session, "visual performance", mouse_visual, "irrel performance", mouse_irrel)

        group_visual_list.append(mouse_visual)
        group_odour_list.append(mouse_odour)
        group_irrel_list.append(mouse_irrel)


    # Plot These
    print("group_visual_list", group_visual_list)
    print("group_odour_list", group_odour_list)
    print("group_irrel_list", group_irrel_list)


    mean_visual = np.mean(group_visual_list)
    mean_odour = np.mean(group_odour_list)
    mean_irrel = np.mean(group_irrel_list)

    visual_sd = np.std(group_visual_list)
    odour_sd = np.std(group_odour_list)
    irrel_sd = np.std(group_irrel_list)

    print("mean visual", np.mean(group_visual_list))
    print("mean odour", np.mean(group_odour_list))
    print("mean irrel", np.mean(group_irrel_list))

    print("visual_sd", visual_sd)
    print("irrel_sd", irrel_sd)


    figure_1 = plt.figure(figsize=(5, 10))
    axis_1 = figure_1.add_subplot(1,1,1)

    n_mice = len(group_visual_list)



    # Plot Individual Mice
    noise_magnitude = 0.02
    for mouse_index in range(n_mice):

        mouse_visual_x = np.random.uniform(low=-noise_magnitude, high=noise_magnitude, size=1)
        mouse_odour_x = 1 + np.random.uniform(low=-noise_magnitude, high=noise_magnitude, size=1)

        axis_1.plot([mouse_visual_x, mouse_odour_x], [group_visual_list[mouse_index], group_irrel_list[mouse_index]], alpha=0.5, c='cornflowerblue')
        axis_1.scatter([mouse_visual_x, mouse_odour_x], [group_visual_list[mouse_index], group_irrel_list[mouse_index]], alpha=0.5, c='cornflowerblue')
        axis_1.scatter([mouse_odour_x], [group_odour_list[mouse_index]], alpha=0.5, c='cornflowerblue')

    # Plot Means
    axis_1.plot([0, 1], [mean_visual, mean_irrel], c='k', linewidth=5, alpha=0.8)
    axis_1.scatter([0, 1], [mean_visual, mean_irrel], c='k', alpha=0.8)
    axis_1.scatter([1], [mean_odour], c='k', alpha=0.8)

    # Add Error Bars
    axis_1.errorbar(0, mean_visual, yerr=visual_sd, fmt='o', c='k', linewidth=4, markersize=10, alpha=0.8)
    axis_1.errorbar(1, mean_irrel, yerr=irrel_sd, fmt='o', c='k', linewidth=4,  markersize=10, alpha=0.8)
    axis_1.errorbar(1, mean_odour, yerr=odour_sd, fmt='o', c='green', linewidth=4,  markersize=10, alpha=0.8)


    # Remove Borders
    axis_1.spines['top'].set_visible(False)
    axis_1.spines['right'].set_visible(False)

    axis_1.set_xticks([0,1], ["Visual\nRelevant", "Visual\nIrrelevant"])

    axis_1.set_ylim([0, 4.2])

    plt.show()


session_list = Session_List.nested_session_list
data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Widefield_Opto"
combined_behaviour_plot_pipeline(data_root, session_list)