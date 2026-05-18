import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import Plotting_Functions.Data_Loading_Functions as Data_Loading_Functions



def get_mean_and_bounds(data):
    print("data", np.shape(data))
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)
    return data_mean, lower_bound, upper_bound



def plot_eigenspectrums(session_list, output_root, group_name):

    # Load Eigenspectrum List
    eigenspectrum_list = Data_Loading_Functions.load_eigenspectrums(session_list, output_root)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    x_values = list(range(1, len(eigenspectrum_list[0])+1))

    for spectrum in eigenspectrum_list:
        axis_1.plot(x_values, spectrum, alpha=0.5)
        axis_1.scatter(x_values, spectrum, alpha=0.5)

    axis_1.set_ylim([0, 1.05])
    #axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_title("Eigenspectrum " + group_name)

    plt.show()


def plot_observability_eigenspectrums(session_list, output_root, group_name):

    # Load Eigenspectrum List
    eigenspectrum_list = Data_Loading_Functions.load_observability_eigenspectrums(session_list, output_root)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    x_values = list(range(1, len(eigenspectrum_list[0]) + 1))

    for spectrum in eigenspectrum_list:
        axis_1.plot(x_values, spectrum, alpha=0.5)
        axis_1.scatter(x_values, spectrum, alpha=0.5)

    #axis_1.set_ylim([0, 1.05])
    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_title("Observability Eigenspectrum " + group_name)

    plt.show()







def plot_right_alignment(session_list, output_root, group_name):

    # Load Eigenspectrum List
    alignment_list = Data_Loading_Functions.load_data(session_list, output_root, "Right_Eigenvectors_Lick_Alignment.npy", truncation=30)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    x_values = list(range(1, len(alignment_list[0])+1))

    for spectrum in alignment_list:
        axis_1.plot(x_values, spectrum, alpha=0.5)
        axis_1.scatter(x_values, spectrum, alpha=0.5)

    axis_1.set_ylim([-1, 1])
    #axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_title("Right Alignment " + group_name)

    plt.show()


def plot_left_alignment(session_list, output_root, group_name):

    # Load alignment List
    vis_1_alignment_list = Data_Loading_Functions.load_data(session_list, output_root, "Left_Eigenvectors_Vis_1_Alignment.npy", truncation=30)
    vis_2_alignment_list = Data_Loading_Functions.load_data(session_list, output_root, "Left_Eigenvectors_Vis_2_Alignment.npy", truncation=30)

    x_values = list(range(1, len(vis_1_alignment_list[0]) + 1))

    figure_1 = plt.figure()
    vis_1_axis = figure_1.add_subplot(1, 2, 1)
    vis_2_axis = figure_1.add_subplot(1,2,2)

    for spectrum in vis_1_alignment_list:
        vis_1_axis.plot(x_values, spectrum, alpha=0.5)
        vis_1_axis.scatter(x_values, spectrum, alpha=0.5)

    for spectrum in vis_2_alignment_list:
        vis_2_axis.plot(x_values, spectrum, alpha=0.5)
        vis_2_axis.scatter(x_values, spectrum, alpha=0.5)

    vis_1_axis.set_ylim([-1, 1])
    vis_2_axis.set_ylim([-1, 1])

    vis_1_axis.spines[['right', 'top']].set_visible(False)
    vis_2_axis.spines[['right', 'top']].set_visible(False)

    vis_1_axis.set_title("Vis 1 left alignment " + group_name)
    vis_2_axis.set_title("Vis 2 left alignment " + group_name)
    plt.show()






def plot_left_alignment_observability(session_list, output_root, group_name):

    # Load alignment List
    vis_1_alignment_list = Data_Loading_Functions.load_data(session_list, output_root, "Observability_Vis_1_Alignment.npy", truncation=30)
    vis_2_alignment_list = Data_Loading_Functions.load_data(session_list, output_root, "Observability_Vis_2_Alignment.npy", truncation=30)

    x_values = list(range(1, len(vis_1_alignment_list[0]) + 1))


    figure_1 = plt.figure()
    vis_1_axis = figure_1.add_subplot(1, 2, 1)
    vis_2_axis = figure_1.add_subplot(1,2,2)

    for spectrum in vis_1_alignment_list:
        vis_1_axis.plot(x_values, spectrum, alpha=0.5)
        vis_1_axis.scatter(x_values, spectrum, alpha=0.5)

    for spectrum in vis_2_alignment_list:
        vis_2_axis.plot(x_values, spectrum, alpha=0.5)
        vis_2_axis.scatter(x_values, spectrum, alpha=0.5)

    vis_1_axis.set_ylim([-1, 1])
    vis_2_axis.set_ylim([-1, 1])

    vis_1_axis.spines[['right', 'top']].set_visible(False)
    vis_2_axis.spines[['right', 'top']].set_visible(False)

    vis_1_axis.set_title("Vis 1 left alignment " + group_name)
    vis_2_axis.set_title("Vis 2 left alignment " + group_name)

    figure_1.suptitle("Observability Stimulus Alignment")
    plt.show()





def plot_controlability_lick_alignment(session_list, output_root, group_name):

    # Load Eigenspectrum List
    alignment_list = Data_Loading_Functions.load_data(session_list, output_root, "Controlability_lick_Alignment.npy", truncation=30)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    x_values = list(range(1, len(alignment_list[0])+1))

    for spectrum in alignment_list:
        axis_1.plot(x_values, spectrum, alpha=0.5)
        axis_1.scatter(x_values, spectrum, alpha=0.5)

    axis_1.set_ylim([-1, 1])
    #axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_title("Controlability Lick Alignment " + group_name)

    plt.show()



"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                                                                                        Functions for Plotting Average Group Results

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

"""



def compare_eigenspectrum_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Eigenspectrum List
    wt_eigenspectrum_list = Data_Loading_Functions.load_eigenspectrums(wt_session_list, wt_output_root)
    neurexin_eigenspectrum_list = Data_Loading_Functions.load_eigenspectrums(neurexin_session_list, neurexin_output_root)

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_eigenspectrum_list)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(neurexin_eigenspectrum_list)


    x_values = list(range(1, len(wt_mean) + 1))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(x_values, wt_mean)
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2)

    axis_1.plot(x_values, nx_mean)
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2)

    axis_1.set_ylim([0, 1.05])
    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.show()


def compare_right_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Eigenspectrum List
    wt_alignment = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Right_Eigenvectors_Lick_Alignment.npy", truncation=30)
    nx_alignment = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Right_Eigenvectors_Lick_Alignment.npy", truncation=30)

    wt_alignment = np.array(wt_alignment)
    nx_alignment = np.array(nx_alignment)
    wt_alignment = np.squeeze(wt_alignment)
    nx_alignment = np.squeeze(nx_alignment)
    print("wt_alignment", np.shape(wt_alignment))
    print("nx_alignment", np.shape(nx_alignment))

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_alignment)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_alignment)


    x_values = list(range(1, len(wt_mean) + 1))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    axis_1.set_ylim([-1, 1])
    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Right Eigenvector Aligment")
    plt.show()


    # Sum and Plot Totals
    #wt_alignment_sum = np.mean(wt_alignment, axis=1)
    #nx_alignment_sum = np.mean(nx_alignment, axis=1)

    wt_alignment_sum = wt_alignment[:, 0]
    nx_alignment_sum = nx_alignment[:, 0]
    print("wt_alignment_sum", wt_alignment_sum)
    print("nx_alignment_sum", nx_alignment_sum)

    t_stat, p_value = stats.ttest_ind(wt_alignment_sum, nx_alignment_sum)
    print("t_stat", t_stat, "p_value", p_value)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    wt_xvalues = np.zeros(len(wt_alignment_sum)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_alignment_sum))
    nx_xvalues = np.ones(len(nx_alignment_sum)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_alignment_sum))

    axis_1.scatter(wt_xvalues, wt_alignment_sum, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_alignment_sum, c='m', alpha=0.4)

    axis_1.set_xlim([-0.5, 1.5])
    axis_1.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])

    axis_1.set_ylim([-1, 1])
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_title("Right Eigenvector Lick Alignment")

    plt.show()





def compare_left_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Eigenspectrum List
    wt_alignment_vis_1 = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Left_Eigenvectors_Vis_1_Alignment.npy", truncation=30)
    wt_alignment_vis_2 = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Left_Eigenvectors_Vis_2_Alignment.npy", truncation=30)

    nx_alignment_vis_1 = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Left_Eigenvectors_Vis_1_Alignment.npy", truncation=30)
    nx_alignment_vis_2 = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Left_Eigenvectors_Vis_2_Alignment.npy", truncation=30)


    wt_alignment_vis_1 = np.array(wt_alignment_vis_1)
    wt_alignment_vis_2 = np.array(wt_alignment_vis_2)
    nx_alignment_vis_1 = np.array(nx_alignment_vis_1)
    nx_alignment_vis_2 = np.array(nx_alignment_vis_2)

    wt_alignment_vis_1 = np.squeeze(wt_alignment_vis_1)
    wt_alignment_vis_2 = np.squeeze(wt_alignment_vis_2)
    nx_alignment_vis_1 = np.squeeze(nx_alignment_vis_1)
    nx_alignment_vis_2 = np.squeeze(nx_alignment_vis_2)


    wt_mean_vis_1, wt_lower_bound_vis_1, wt_upper_bound_vis_1 = get_mean_and_bounds(wt_alignment_vis_1)
    nx_mean_vis_1, nx_lower_bound_vis_1, nx_upper_bound_vis_1 = get_mean_and_bounds(nx_alignment_vis_1)

    wt_mean_vis_2, wt_lower_bound_vis_2, wt_upper_bound_vis_2 = get_mean_and_bounds(wt_alignment_vis_2)
    nx_mean_vis_2, nx_lower_bound_vis_2, nx_upper_bound_vis_2 = get_mean_and_bounds(nx_alignment_vis_2)


    x_values = list(range(1, len(wt_mean_vis_1) + 1))

    figure_1 = plt.figure()
    vis_1_axis = figure_1.add_subplot(1,2,1)
    vis_2_axis = figure_1.add_subplot(1,2,2)

    vis_1_axis.plot(x_values, wt_mean_vis_1, c='b')
    vis_1_axis.plot(x_values, nx_mean_vis_1, c='m')
    vis_1_axis.fill_between(x_values, wt_lower_bound_vis_1, wt_upper_bound_vis_1, alpha=0.2, color='b')
    vis_1_axis.fill_between(x_values, nx_lower_bound_vis_1, nx_upper_bound_vis_1, alpha=0.2, color='m')

    vis_2_axis.plot(x_values, wt_mean_vis_2, c='b')
    vis_2_axis.plot(x_values, nx_mean_vis_2, c='m')
    vis_2_axis.fill_between(x_values, wt_lower_bound_vis_2, wt_upper_bound_vis_2, alpha=0.2, color='b')
    vis_2_axis.fill_between(x_values, nx_lower_bound_vis_2, nx_upper_bound_vis_2, alpha=0.2, color='m')

    vis_1_axis.set_ylim([-1, 1])
    vis_2_axis.set_ylim([-1, 1])

    vis_1_axis.spines[['right', 'top']].set_visible(False)
    vis_2_axis.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Left Eigenvector Stimulus Alignment")

    plt.show()


    # Sum and Plot Totals
    """
    wt_alignment_sum_vis_1 = np.mean(wt_alignment_vis_1, axis=1)
    nx_alignment_sum_vis_1 = np.mean(nx_alignment_vis_1, axis=1)
    wt_alignment_sum_vis_2 = np.mean(wt_alignment_vis_2, axis=1)
    nx_alignment_sum_vis_2 = np.mean(nx_alignment_vis_2, axis=1)
    """

    wt_alignment_sum_vis_1 = wt_alignment_vis_1[:, 0]
    nx_alignment_sum_vis_1 = nx_alignment_vis_1[:, 0]
    wt_alignment_sum_vis_2 = wt_alignment_vis_2[:, 0]
    nx_alignment_sum_vis_2 = nx_alignment_vis_2[:, 0]

    vis_1_t, vis_1_p = stats.ttest_ind(wt_alignment_sum_vis_1, nx_alignment_sum_vis_1)
    vis_2_t, vis_2_p = stats.ttest_ind(wt_alignment_sum_vis_2, nx_alignment_sum_vis_2)

    print("vis_1_t", vis_1_t, "vis_1_p", vis_1_p)
    print("vis_2_t", vis_2_t, "vis_2_p", vis_2_p)

    figure_1 = plt.figure()
    vis_1_axis = figure_1.add_subplot(1, 2, 1)
    vis_2_axis = figure_1.add_subplot(1,2,2)

    wt_xvalues = np.zeros(len(wt_alignment_sum_vis_1)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_alignment_sum_vis_1))
    nx_xvalues = np.ones(len(nx_alignment_sum_vis_1)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_alignment_sum_vis_1))

    vis_1_axis.scatter(wt_xvalues, wt_alignment_sum_vis_1, c='b', alpha=0.4)
    vis_1_axis.scatter(nx_xvalues, nx_alignment_sum_vis_1, c='m', alpha=0.4)
    vis_2_axis.scatter(wt_xvalues, wt_alignment_sum_vis_2, c='b', alpha=0.4)
    vis_2_axis.scatter(nx_xvalues, nx_alignment_sum_vis_2, c='m', alpha=0.4)

    vis_1_axis.set_xlim([-0.5, 1.5])
    vis_1_axis.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    vis_1_axis.set_ylim([-0.6, 0.6])
    vis_1_axis.spines[['right', 'top']].set_visible(False)

    vis_2_axis.set_xlim([-0.5, 1.5])
    vis_2_axis.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    vis_2_axis.set_ylim([-0.6, 0.6])
    vis_2_axis.spines[['right', 'top']].set_visible(False)

    plt.show()



def compare_non_normality(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Non Normality
    wt_non_normality = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "non_normality.npy")
    nx_non_normality = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "non_normality.npy")


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    wt_xvalues = np.zeros(len(wt_non_normality)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_non_normality))
    nx_xvalues = np.ones(len(nx_non_normality)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_non_normality))

    axis_1.scatter(wt_xvalues, wt_non_normality, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_non_normality, c='m', alpha=0.4)

    t_stat, p_value = stats.ttest_ind(wt_non_normality, nx_non_normality)
    print("t_stat", t_stat)
    print("p_value", p_value)

    axis_1.set_xlim([-0.5, 1.5])
    axis_1.set_ylim([0, 0.03])
    axis_1.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("Non Normality")
    plt.show()




def compare_controlability_lick_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Eigenspectrum List
    wt_alignment = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Controlability_lick_Alignment.npy", truncation=30)
    nx_alignment = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Controlability_lick_Alignment.npy", truncation=30)

    wt_alignment = np.array(wt_alignment)
    nx_alignment = np.array(nx_alignment)
    wt_alignment = np.squeeze(wt_alignment)
    nx_alignment = np.squeeze(nx_alignment)
    print("wt_alignment", np.shape(wt_alignment))
    print("nx_alignment", np.shape(nx_alignment))

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_alignment)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_alignment)


    x_values = list(range(1, len(wt_mean) + 1))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    axis_1.set_ylim([-1, 1])
    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Controlability Eigenvector Lick Aligment")
    plt.show()


    # Sum and Plot Totals
    #wt_alignment_sum = np.mean(wt_alignment, axis=1)
    #nx_alignment_sum = np.mean(nx_alignment, axis=1)
    wt_alignment_sum = wt_alignment[:, 0]
    nx_alignment_sum = nx_alignment[:, 0]

    print("wt_alignment_sum", wt_alignment_sum)
    print("nx_alignment_sum", nx_alignment_sum)
    control_t, control_p = stats.ttest_ind(wt_alignment_sum, nx_alignment_sum)
    print("control_t", control_t, "control_p", control_p)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    wt_xvalues = np.zeros(len(wt_alignment_sum)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_alignment_sum))
    nx_xvalues = np.ones(len(nx_alignment_sum)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_alignment_sum))

    axis_1.scatter(wt_xvalues, wt_alignment_sum, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_alignment_sum, c='m', alpha=0.4)

    axis_1.set_title("Controlability Eigenvector Lick Aligment")

    axis_1.set_xlim([-0.5, 1.5])
    axis_1.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])

    axis_1.set_ylim([-1, 1])
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.show()





def compare_observability_stim_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Eigenspectrum List
    wt_alignment_vis_1 = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Observability_Vis_1_Alignment.npy", truncation=30)
    wt_alignment_vis_2 = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Observability_Vis_2_Alignment.npy", truncation=30)

    nx_alignment_vis_1 = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Observability_Vis_1_Alignment.npy", truncation=30)
    nx_alignment_vis_2 = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Observability_Vis_2_Alignment.npy", truncation=30)


    wt_alignment_vis_1 = np.array(wt_alignment_vis_1)
    wt_alignment_vis_2 = np.array(wt_alignment_vis_2)
    nx_alignment_vis_1 = np.array(nx_alignment_vis_1)
    nx_alignment_vis_2 = np.array(nx_alignment_vis_2)

    wt_alignment_vis_1 = np.squeeze(wt_alignment_vis_1)
    wt_alignment_vis_2 = np.squeeze(wt_alignment_vis_2)
    nx_alignment_vis_1 = np.squeeze(nx_alignment_vis_1)
    nx_alignment_vis_2 = np.squeeze(nx_alignment_vis_2)


    wt_mean_vis_1, wt_lower_bound_vis_1, wt_upper_bound_vis_1 = get_mean_and_bounds(wt_alignment_vis_1)
    nx_mean_vis_1, nx_lower_bound_vis_1, nx_upper_bound_vis_1 = get_mean_and_bounds(nx_alignment_vis_1)

    wt_mean_vis_2, wt_lower_bound_vis_2, wt_upper_bound_vis_2 = get_mean_and_bounds(wt_alignment_vis_2)
    nx_mean_vis_2, nx_lower_bound_vis_2, nx_upper_bound_vis_2 = get_mean_and_bounds(nx_alignment_vis_2)


    x_values = list(range(1, len(wt_mean_vis_1) + 1))

    figure_1 = plt.figure()
    vis_1_axis = figure_1.add_subplot(1,2,1)
    vis_2_axis = figure_1.add_subplot(1,2,2)

    vis_1_axis.plot(x_values, wt_mean_vis_1, c='b')
    vis_1_axis.plot(x_values, nx_mean_vis_1, c='m')
    vis_1_axis.fill_between(x_values, wt_lower_bound_vis_1, wt_upper_bound_vis_1, alpha=0.2, color='b')
    vis_1_axis.fill_between(x_values, nx_lower_bound_vis_1, nx_upper_bound_vis_1, alpha=0.2, color='m')

    vis_2_axis.plot(x_values, wt_mean_vis_2, c='b')
    vis_2_axis.plot(x_values, nx_mean_vis_2, c='m')
    vis_2_axis.fill_between(x_values, wt_lower_bound_vis_2, wt_upper_bound_vis_2, alpha=0.2, color='b')
    vis_2_axis.fill_between(x_values, nx_lower_bound_vis_2, nx_upper_bound_vis_2, alpha=0.2, color='m')

    vis_1_axis.set_ylim([-1, 1])
    vis_2_axis.set_ylim([-1, 1])

    vis_1_axis.spines[['right', 'top']].set_visible(False)
    vis_2_axis.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Observability Eigenvector Stimulus Alignment")

    plt.show()


    # Sum and Plot Totals
    """
    wt_alignment_sum_vis_1 = np.mean(wt_alignment_vis_1, axis=1)
    nx_alignment_sum_vis_1 = np.mean(nx_alignment_vis_1, axis=1)
    wt_alignment_sum_vis_2 = np.mean(wt_alignment_vis_2, axis=1)
    nx_alignment_sum_vis_2 = np.mean(nx_alignment_vis_2, axis=1)
    """
    wt_alignment_sum_vis_1 = wt_alignment_vis_1[:, 0]
    nx_alignment_sum_vis_1 = nx_alignment_vis_1[:, 0]
    wt_alignment_sum_vis_2 = wt_alignment_vis_2[:, 0]
    nx_alignment_sum_vis_2 = nx_alignment_vis_2[:, 0]

    vis_1_t, vis_1_p = stats.ttest_ind(wt_alignment_sum_vis_1, nx_alignment_sum_vis_1)
    vis_2_t, vis_2_p = stats.ttest_ind(wt_alignment_sum_vis_2, nx_alignment_sum_vis_2)
    print("vis_1_t", vis_1_t, "vis_1_p", vis_1_p)
    print("vis_2_t", vis_2_t, "vis_2_p", vis_2_p)

    figure_1 = plt.figure()
    vis_1_axis = figure_1.add_subplot(1, 2, 1)
    vis_2_axis = figure_1.add_subplot(1,2,2)

    wt_xvalues = np.zeros(len(wt_alignment_sum_vis_1)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_alignment_sum_vis_1))
    nx_xvalues = np.ones(len(nx_alignment_sum_vis_1)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_alignment_sum_vis_1))

    vis_1_axis.scatter(wt_xvalues, wt_alignment_sum_vis_1, c='b', alpha=0.4)
    vis_1_axis.scatter(nx_xvalues, nx_alignment_sum_vis_1, c='m', alpha=0.4)
    vis_2_axis.scatter(wt_xvalues, wt_alignment_sum_vis_2, c='b', alpha=0.4)
    vis_2_axis.scatter(nx_xvalues, nx_alignment_sum_vis_2, c='m', alpha=0.4)

    vis_1_axis.set_xlim([-0.5, 1.5])
    vis_1_axis.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    #vis_1_axis.set_ylim([-0.1, 0.1])
    vis_1_axis.spines[['right', 'top']].set_visible(False)

    vis_2_axis.set_xlim([-0.5, 1.5])
    vis_2_axis.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    #vis_2_axis.set_ylim([-0.1, 0.1])
    vis_2_axis.spines[['right', 'top']].set_visible(False)

    plt.show()



def compare_lick_cd_decay_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_decay = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Lick_CD_Decay.npy")
    nx_decay = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Lick_CD_Decay.npy")
    print("wt_decay", np.shape(wt_decay))
    print("nx_decay", np.shape(nx_decay))
    print("wt_decay", wt_decay)
    print("nx_decay", nx_decay)


    t_stats, p_values = stats.ttest_ind(wt_decay, nx_decay, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_values = np.multiply(binary_sig, 0.49)

    print("p values", p_values)
    print("t stats", t_stats)

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_decay)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_decay)

    x_values = list(range(len(wt_mean)))
    x_values = np.multiply(x_values, (1000/6.37))
    print("wt_lower_bound", wt_lower_bound)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    axis_1.set_ylim([0, 0.5])

    axis_1.scatter(x_values, sig_values, alpha=binary_sig, c='k')

    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Lick CD Decay")
    plt.show()



def compare_lick_cd_decay_total_norm_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_decay = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "decay_total_norm.npy")
    nx_decay = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "decay_total_norm.npy")

    t_stats, p_values = stats.ttest_ind(wt_decay, nx_decay, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_values = np.multiply(binary_sig, 0.49)

    print("p values", p_values)

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_decay)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_decay)

    x_values = list(range(len(wt_mean)))
    x_values = np.multiply(x_values, (1000/6.37))

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    axis_1.scatter(x_values, sig_values, alpha=binary_sig, c='k')

    axis_1.set_ylim([0, 0.5])
    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Lick CD Decay Total Norm")
    plt.show()




def compare_lick_reachability_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_reach = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Lick_Reachability.npy")
    nx_reach = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Lick_Reachability.npy")

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    wt_xvalues = np.zeros(len(wt_reach)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_reach))
    nx_xvalues = np.ones(len(nx_reach)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_reach))

    axis_1.scatter(wt_xvalues, wt_reach, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_reach, c='m', alpha=0.4)

    t_stat, p_value = stats.ttest_ind(wt_reach, nx_reach)
    print("t_stat", t_stat)
    print("p_value", p_value)

    axis_1.set_xlim([-0.5, 1.5])
    #axis_1.set_ylim([0, 0.03])
    axis_1.set_xticks([0, 1], labels=['Wildtype', 'Neurexin'])
    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("Lick Reachability")
    plt.show()




def get_hist_density(data, bin_range=150, bin_size=25):

    n_samples = len(data)
    bin_starts = list(range(-bin_range, bin_range))
    bin_stops = np.add(bin_starts, bin_size)
    n_bins = len(bin_starts)

    density = []
    for bin_index in range(n_bins):
        bin_start = bin_starts[bin_index]
        bin_stop = bin_stops[bin_index]
        bin_counts = np.sum(np.where(data >= bin_start and data < bin_stop, 1 , 0))
        bin_counts = float(bin_counts)/n_samples
        density.append(bin_counts)

    return density





def compare_coupling_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    bin_range=20
    bin_size=1
    x_values = list(range(-bin_range, bin_range, bin_size))

    # Load data
    wt_data = Data_Loading_Functions.load_distributions(wt_session_list, wt_output_root, "coupling_effect.npy", bin_range, bin_size)
    nx_data = Data_Loading_Functions.load_distributions(neurexin_session_list, neurexin_output_root, "coupling_effect.npy", bin_range, bin_size)

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_data)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_data)

    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    print("p_values", p_values)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_values = np.multiply(binary_sig, 0.3)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    axis_1.scatter(x_values, sig_values, alpha=binary_sig, c='k')

    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_ylim([0, 0.36])
    axis_1.set_title("Lick Coupling")
    plt.show()





def compare_mean_coupling_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_data = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "coupling_effect.npy")
    nx_data = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "coupling_effect.npy")

    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    print("p_values", p_values)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    wt_xvalues = np.zeros(len(wt_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_data))
    nx_xvalues = np.ones(len(nx_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_data))

    axis_1.scatter(wt_xvalues, wt_data, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_data, c='m', alpha=0.4)

    plt.show()

    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("Mean Coupling")
    plt.show()




def compare_mean_coupling_single_step_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_data = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "single_timestep_effect.npy")
    nx_data = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "single_timestep_effect.npy")

    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    print("p_values", p_values)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    wt_xvalues = np.zeros(len(wt_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_data))
    nx_xvalues = np.ones(len(nx_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_data))

    axis_1.scatter(wt_xvalues, wt_data, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_data, c='m', alpha=0.4)

    plt.show()

    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("Mean Coupling")
    plt.show()





def compare_transfer_trajectories(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "vis_1_transfer_trajectory.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "vis_1_transfer_trajectory.npy")

    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_values = np.multiply(binary_sig, 0.49)

    print("p values", p_values)
    print("t stats", t_stats)

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_data)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_data)

    x_values = list(range(len(wt_mean)))
    x_values = np.multiply(x_values, (1000 / 6.37))
    print("wt_lower_bound", wt_lower_bound)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    #axis_1.set_ylim([0, 0.5])

    axis_1.scatter(x_values, sig_values, alpha=binary_sig, c='k')

    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Vis 1 transfer Trajectory")
    plt.show()





def compare_direct_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_data = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "vis_1_direct_projection.npy")
    nx_data = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "vis_1_direct_projection.npy")

    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    print("p_values", p_values)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    wt_xvalues = np.zeros(len(wt_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_data))
    nx_xvalues = np.ones(len(nx_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_data))

    axis_1.scatter(wt_xvalues, wt_data, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_data, c='m', alpha=0.4)

    plt.show()

    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("Stimulus Direct Projection")
    plt.show()









def compare_prestim_direct_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_data = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "direct_prestim_projection.npy")
    nx_data = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "direct_prestim_projection.npy")

    t_stats, p_values = stats.ttest_ind(wt_data, nx_data, axis=0)
    print("p_values", p_values)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    wt_xvalues = np.zeros(len(wt_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(wt_data))
    nx_xvalues = np.ones(len(nx_data)) + np.random.uniform(low=-0.1, high=0.1, size=len(nx_data))

    axis_1.scatter(wt_xvalues, wt_data, c='b', alpha=0.4)
    axis_1.scatter(nx_xvalues, nx_data, c='m', alpha=0.4)

    plt.show()

    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("Stimulus Direct Projection")
    plt.show()





def compare_prestim_coupling_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load data
    wt_decay = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Prestim_Coupling.npy")
    nx_decay = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Prestim_Coupling.npy")
    print("wt_decay", np.shape(wt_decay))
    print("nx_decay", np.shape(nx_decay))
    print("wt_decay", wt_decay)
    print("nx_decay", nx_decay)


    t_stats, p_values = stats.ttest_ind(wt_decay, nx_decay, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_values = np.multiply(binary_sig, 0.49)

    print("p values", p_values)
    print("t stats", t_stats)

    wt_mean, wt_lower_bound, wt_upper_bound = get_mean_and_bounds(wt_decay)
    nx_mean, nx_lower_bound, nx_upper_bound = get_mean_and_bounds(nx_decay)

    x_values = list(range(len(wt_mean)))
    x_values = np.multiply(x_values, (1000/6.37))
    print("wt_lower_bound", wt_lower_bound)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.plot(x_values, wt_mean, c='b')
    axis_1.fill_between(x_values, wt_lower_bound, wt_upper_bound, alpha=0.2, color='b')

    axis_1.plot(x_values, nx_mean, c='m')
    axis_1.fill_between(x_values, nx_lower_bound, nx_upper_bound, alpha=0.2, color='m')

    axis_1.set_ylim([-0.2, 0.55])

    axis_1.scatter(x_values, sig_values, alpha=binary_sig, c='k')

    # axis_1.set_xticks(list(range(1, len(eigenspectrum_list[0])+1,4)))
    # Hide the right and top spines
    axis_1.spines[['right', 'top']].set_visible(False)

    figure_1.suptitle("Prestim Ramping To Lick Coupling")
    plt.show()

