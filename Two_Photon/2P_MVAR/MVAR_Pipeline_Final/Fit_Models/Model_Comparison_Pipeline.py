import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


import Get_Previous_Step_R2
import Fit_Diagonal_with_Stimuli


def plot_model_results(response_only, diagonal, standard, seperate_contexts):

    # Create Score Matrix
    response_only = np.expand_dims(response_only, 1)
    diagonal = np.expand_dims(diagonal, 1)
    standard = np.expand_dims(standard, 1)
    seperate_contexts = np.expand_dims(seperate_contexts, 1)
    score_matrix = np.hstack([response_only, diagonal, standard, seperate_contexts])
    print("Score Matrix", np.shape(score_matrix))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)    
    n_mice = len(response_only)
    
    x_values = list(range(0, 4))
    
    for mouse_index in range(n_mice):
        axis_1.plot(x_values, score_matrix[mouse_index], c='cornflowerblue', alpha=0.5)
        axis_1.scatter(x_values, score_matrix[mouse_index], c='cornflowerblue', alpha=0.5)

    # Get Mean Scores
    mean_scores = np.mean(score_matrix, axis=0)
    print("mean scores", mean_scores)

    axis_1.bar(x=x_values, height=mean_scores,  color="slateblue", alpha=0.5)

    # Remove Borders
    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_xticks(x_values, labels=["Response \nBehaviour",
                                        "Diagonal \nWeights",
                                        "Standard \nModel",
                                        "Seperate \nContexts"])

    axis_1.set_ylabel('CV r2')

    # Print Signficance
    t_stat, p_value = stats.ttest_rel(score_matrix[:, 1], score_matrix[:, 0], axis=0)
    print("Response V Diag: t_stat", t_stat, "p_value", p_value)

    t_stat, p_value = stats.ttest_rel(score_matrix[:, 2], score_matrix[:, 1], axis=0)
    print("Recurrent v Only Diag: t_stat", t_stat, "p_value", p_value)

    t_stat, p_value = stats.ttest_rel(score_matrix[:, 3], score_matrix[:, 2], axis=0)
    print("Seperate Contexts", t_stat, "p_value", p_value)
    plt.show()



def model_comparison_pipeline(mvar_output_root, session_list):

    response_only_list = []
    diag_stim_list = []
    standard_list = []
    seperate_contexts_list = []

    for session in session_list:

        # Get Diagonal With Stim R2
        #Fit_Diagonal_with_Stimuli.fit_diagonal_with_stimuli(mvar_output_root, session)

        # Get Previous Step R2
        #previous_step_2 = Get_Previous_Step_R2.get_previous_step_r2(mvar_output_root, session)

        # Get Response + Behaviour
        response_behaviour_scores = np.load(os.path.join(mvar_output_root, session, "Ridge_Penalty_Search", "No_Recurrent", "Ridge_Penalty_Search_Results.npy"))
        response_behaviour = np.max(response_behaviour_scores)
        response_only_list.append(response_behaviour)

        # Get Top Standard Model R2
        standard_scores = np.load(os.path.join(mvar_output_root, session, "Ridge_Penalty_Search", "Standard", "Ridge_Penalty_Search_Results.npy"))
        standard_r2 = np.max(standard_scores)
        standard_list.append(standard_r2)

        # Get Seperate Context R2
        seperate_context_scores = np.load(os.path.join(mvar_output_root, session, "Ridge_Penalty_Search", "Seperate_Contexts", "Ridge_Penalty_Search_Results.npy"))
        seperate_r2 = np.max(seperate_context_scores)
        seperate_contexts_list.append(seperate_r2)

        # Get No Recurrent R2
        no_recurrent_scores = np.load(os.path.join(mvar_output_root, session, "Ridge_Penalty_Search", "Diagonal_with_Stim", "Ridge_Penalty_Search_Results.npy"))
        no_reurrent_r2 = np.max(no_recurrent_scores)
        diag_stim_list.append(no_reurrent_r2)
        #print("Session", session, "Previous_Step_R2", previous_step_2, "no recurrent r2", no_reurrent_r2, "Standard Model", standard_r2, "Seperate Context R2", seperate_r2)

    plot_model_results(response_only_list,
                       diag_stim_list,
                       standard_list,
                       seperate_contexts_list)

    t_stat, p_value = stats.ttest_rel(standard_list, diag_stim_list)
    print("t_stat", t_stat, "p_value", p_value)




"""
Different model types:
"Previous Timestep"
"No_Recurrent + Previous Timestep"
"Standard"
"Seperate_Contexts"
"""

# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Final_No_Z"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results_MW1"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results"



control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

# Model Info
start_window = -17
stop_window = 12


# Control Switching
model_comparison_pipeline(mvar_output_root, control_session_list)

