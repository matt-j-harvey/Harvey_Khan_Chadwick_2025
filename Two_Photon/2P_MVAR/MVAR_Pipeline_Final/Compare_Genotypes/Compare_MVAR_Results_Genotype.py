import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import stats


def load_group_projections(root_directory, session_list):

    diagonal_list = []
    recurrent_list=  []

    for session in session_list:
        session_diagonal = np.load(os.path.join(root_directory, session, "Recurrent Amplification", "Stimuli_Weight_Interactions", "diagonal_weights_Interaction.npy"))
        session_recurrent=  np.load(os.path.join(root_directory, session, "Recurrent Amplification", "Stimuli_Weight_Interactions", "recurrent_weights_Interaction.npy"))

        diagonal_list.append(session_diagonal)
        recurrent_list.append(session_recurrent)

    diagonal_list = np.array(diagonal_list)
    recurrent_list = np.array(recurrent_list)

    return diagonal_list, recurrent_list


def compare_genotypes(control_directory_root, hom_directory_root, control_session_list, hom_session_list, output_directory):

    # Compare Projections

    # Load projections
    control_diagonal, control_recurrent = load_group_projections(control_directory_root, control_session_list)
    hom_diagonal, hom_recurrent = load_group_projections(hom_directory_root, hom_session_list)
    print("control_diagonal", np.shape(control_diagonal))

    # Select Rewarded Vis 1
    control_diagonal = control_diagonal[:, 0]
    control_recurrent = control_recurrent[:, 0]
    hom_diagonal = hom_diagonal[:, 0]
    hom_recurrent = hom_recurrent[:, 0]

    # Squeeze!
    control_diagonal = np.squeeze(control_diagonal)
    control_recurrent = np.squeeze(control_recurrent)
    hom_diagonal = np.squeeze(hom_diagonal)
    hom_recurrent = np.squeeze(hom_recurrent)

    # Get Means
    control_diagonal_mean = np.mean(control_diagonal, axis=0)
    control_recurrent_mean = np.mean(control_recurrent, axis=0)
    hom_diagonal_mean = np.mean(hom_diagonal, axis=0)
    hom_recurrent_mean = np.mean(hom_recurrent, axis=0)
    print("control_diagonal_mean", np.shape(control_diagonal_mean))

    # Get SEMs
    control_diagonal_sem = stats.sem(control_diagonal, axis=0)
    control_recurrent_sem = stats.sem(control_recurrent, axis=0)
    hom_diagonal_sem = stats.sem(hom_diagonal, axis=0)
    hom_recurrent_sem = stats.sem(hom_recurrent, axis=0)
    print("control_diagonal_sem", np.shape(control_diagonal_sem))

    # Get Upper and Lower Bounds
    control_diagonal_upper_bound = np.add(control_diagonal_mean, control_diagonal_sem)
    control_diagonal_lower_bound = np.subtract(control_diagonal_mean, control_diagonal_sem)
    control_recurrent_upper_bound = np.add(control_recurrent_mean, control_recurrent_sem)
    control_recurrent_lower_bound = np.subtract(control_recurrent_mean, control_recurrent_sem)
    hom_diagonal_upper_bound = np.add(hom_diagonal_mean, hom_diagonal_sem)
    hom_diagonal_lower_bound = np.subtract(hom_diagonal_mean, hom_diagonal_sem)
    hom_recurrent_upper_bound = np.add(hom_recurrent_mean, hom_recurrent_sem)
    hom_recurrent_lower_bound = np.subtract(hom_recurrent_mean, hom_recurrent_sem)

    print("control_diagonal", np.shape(control_diagonal))
    print("hom_diagonal", np.shape(hom_diagonal))

    x_values = list(range(0,9))
    x_values = np.multiply(x_values, 1.0/6.37)

    figure_1 = plt.figure(figsize=(15, 5))
    diagonal_axis = figure_1.add_subplot(1,2,1)
    recurrent_axis = figure_1.add_subplot(1,2,2)

    diagonal_axis.plot(x_values, control_diagonal_mean, c='b')
    diagonal_axis.fill_between(x_values, control_diagonal_lower_bound, control_diagonal_upper_bound, alpha=0.5, color='b')

    diagonal_axis.plot(x_values, hom_diagonal_mean, c='m')
    diagonal_axis.fill_between(x_values, hom_diagonal_lower_bound, hom_diagonal_upper_bound, alpha=0.5, color='m')

    recurrent_axis.plot(x_values, control_recurrent_mean, c='b')
    recurrent_axis.fill_between(x_values, control_recurrent_lower_bound, control_recurrent_upper_bound, alpha=0.5, color='b')

    recurrent_axis.plot(x_values, hom_recurrent_mean, c='m')
    recurrent_axis.fill_between(x_values, hom_recurrent_lower_bound, hom_recurrent_upper_bound, alpha=0.5, color='m')

    t_stats, p_value = stats.ttest_ind(control_recurrent[:, 0], hom_recurrent[:, 0])
    print("Recurrent p_value", p_value)

    t_stats, p_value = stats.ttest_ind(control_diagonal[:, 0], hom_diagonal[:, 0])
    print("Diagonal p_value", p_value)

    control_sum = np.sum(control_recurrent, axis=1)
    hom_sum = np.sum(hom_recurrent, axis=1)

    t_stats, p_value = stats.ttest_ind(control_sum,hom_sum)
    print("sum p_value", p_value)

    diagonal_axis.set_xlabel("Time (S)")
    diagonal_axis.set_ylabel("Lick CD Projection")

    recurrent_axis.set_xlabel("Time (S)")
    recurrent_axis.set_ylabel("Lick CD Projection")

    diagonal_axis.spines[['right', 'top']].set_visible(False)
    recurrent_axis.spines[['right', 'top']].set_visible(False)


    plt.show()






# Output directory where you want the data to be saved to
control_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR\Controls"
hom_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR\Homs"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

hom_session_list = [
    r"64.1B\2024_09_09_Switching",
    r"70.1A\2024_09_19_Switching",
    r"70.1B\2024_09_12_Switching",
    r"72.1E\2024_08_23_Switching",
]

output_directory = None

compare_genotypes(control_mvar_root, hom_mvar_root, control_session_list, hom_session_list, output_directory)
