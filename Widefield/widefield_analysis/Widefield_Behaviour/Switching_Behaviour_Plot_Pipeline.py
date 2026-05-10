import os
import numpy as np
import matplotlib.pyplot as plt

import Session_List
import Behaviour_Analysis_Functions
import Plot_Mouse_Performance



def get_blockwise_performance(behaviour_matrix):

    # Find first visual block
    first_trial_type = behaviour_matrix[0, 1]
    if first_trial_type == 3 or first_trial_type == 4:
        first_block = 1
    else:
        first_block = 0

    # Get Performance
    visual_d_prime_list = []
    odour_d_prime_list = []
    irrel_d_prime_list = []

    for x in range(0, 4, 2):
        visual_d_prime = Behaviour_Analysis_Functions.get_visual_block_performance(behaviour_matrix, first_block + x)
        visual_d_prime_list.append(visual_d_prime)

    for x in range(1, 5, 2):
        odour_d_prime = Behaviour_Analysis_Functions.get_odour_block_performance(behaviour_matrix, first_block + x)
        irrel_d_prime = Behaviour_Analysis_Functions.get_odour_block_irrel_performance(behaviour_matrix, first_block + x)
        odour_d_prime_list.append(odour_d_prime)
        irrel_d_prime_list.append(irrel_d_prime)

    return visual_d_prime_list, odour_d_prime_list, irrel_d_prime_list




def switching_behaviour_plot_pipeline(nested_session_list, data_root):

    # Create Empty List
    group_visual_list = []
    group_odour_list = []

    # Iterate Through Mice
    for mouse in nested_session_list:
        mouse_visual_list = []
        mouse_odour_list = []

        # Iterate Through Sessions
        for session in mouse:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)
            print("behaviour_matrix.npy", np.shape(behaviour_matrix))

            # Get Performance
            session_visual_d_prime_list, session_odour_d_prime_list, session_irrel_d_prime_list = get_blockwise_performance(behaviour_matrix)
            mouse_visual_list.append([session_visual_d_prime_list[0], session_irrel_d_prime_list[0], session_visual_d_prime_list[1], session_irrel_d_prime_list[1]])
            mouse_odour_list.append(session_odour_d_prime_list)


            print("session", session)
            print("session_irrel_d_prime_list", session_irrel_d_prime_list)

        # Get Mouse Average
        mean_visual_list = np.mean(mouse_visual_list, axis=0)
        mean_odour_list = np.mean(mouse_odour_list, axis=0)

        group_visual_list.append(mean_visual_list)
        group_odour_list.append(mean_odour_list)

    # Convert To Arrays
    group_visual_list = np.array(group_visual_list)
    group_odour_list = np.array(group_odour_list)

    # Plot These
    print("group_visual_list", np.shape(group_visual_list))
    print("group_odour_list", np.shape(group_odour_list))
    Plot_Mouse_Performance.plot_performance(group_visual_list, group_odour_list)



session_list = Session_List.nested_session_list
data_root= r"/media/matthew/Expansion1/Cortex_Wide_Opto/Experimental_Mice"
switching_behaviour_plot_pipeline(session_list, data_root)