import numpy as np
import os
import matplotlib.pyplot as plt

def scatter_plot(visual_r2_list, odour_r2_list):


    figure_1 = plt.figure(figsize=(4, 6))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    n_mice = len(visual_r2_list)
    for mouse_index in range(n_mice):
        axis_1.scatter([0, 1], [visual_r2_list[mouse_index], odour_r2_list[mouse_index]], c=['b', 'g'], zorder=2)

    axis_1.set_xticks(ticks=[0,1], labels=["Visual", "Odour"])
    axis_1.set_xlim(-0.2, 1.2)
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_xlabel("Context")
    axis_1.set_ylabel("Cross Validated R2 (%)")

    plt.show()


def view_group_r2(mvar_output_directory, session_list):

    best_visual_r2 = []
    best_odour_r2 = []

    for session in session_list:

        # Load r2 matrix
        visual_r2_matrix = np.load(os.path.join(mvar_output_directory, session, "Ridge_Penalty_Search", "visual_Ridge_Penalty_Search_Results.npy"))
        odour_r2_matrix = np.load(os.path.join(mvar_output_directory, session, "Ridge_Penalty_Search", "odour_Ridge_Penalty_Search_Results.npy"))

        max_visual_r2 = np.max(visual_r2_matrix) * 100
        max_odour_r2 = np.max(odour_r2_matrix) * 100

        best_visual_r2.append(max_visual_r2)
        best_odour_r2.append(max_odour_r2)

    mean_visual_r2 = np.mean(best_visual_r2)
    mean_odour_r2 = np.mean(best_odour_r2)

    scatter_plot(best_visual_r2, best_odour_r2)
    print("best_visual_r2", best_visual_r2)
    print("best_odour_r2", best_odour_r2)

    print("mean best visual", mean_visual_r2)
    print("Mean best ofour", mean_odour_r2)



mvar_output_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

view_group_r2(mvar_output_directory, control_session_list)

