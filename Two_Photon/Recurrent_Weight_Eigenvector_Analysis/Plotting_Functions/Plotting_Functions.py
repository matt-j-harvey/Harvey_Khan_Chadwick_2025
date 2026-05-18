import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import Plotting_Functions.Data_Loading_Functions as Data_Loading_Functions
import Plotting_Functions.Plotting_Utils as Plotting_Utils


"""
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                                                                                                                        Functions for Plotting Average Group Results

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
"""


def compare_eigenspectrum_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_eigenspectrums(wt_session_list, wt_output_root)
    nx_data = Data_Loading_Functions.load_eigenspectrums(neurexin_session_list, neurexin_output_root)
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Eigenspectrums", x_value_time=False)


def compare_right_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    # Load Eigenspectrum List
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Right_Eigenvectors_Lick_Alignment.npy", truncation=30)
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Right_Eigenvectors_Lick_Alignment.npy", truncation=30)

    # Format Data
    wt_data = np.array(wt_data)
    nx_data = np.array(nx_data)
    wt_data = np.squeeze(wt_data)
    nx_data = np.squeeze(nx_data)

    # Plot Graph
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Right Eigenvector Alignment", x_value_time=False, ylim=[-0.5, 1])

    # Scatter Leading Eigenvector
    wt_data = wt_data[:, 0]
    nx_data = nx_data[:, 0]

    Plotting_Utils.plot_scatter_graph(wt_data, nx_data, "Right Leading Eigenvector Alignment",  ylim=[-1,1])



def compare_non_normality(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "non_normality.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "non_normality.npy")
    Plotting_Utils.plot_scatter_graph(wt_data, nx_data, "Non-Normality")


def compare_lick_reachability_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Lick_Reachability.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Lick_Reachability.npy")
    Plotting_Utils.plot_scatter_graph(wt_data, nx_data, "Lick Reachability")


def compare_controlability_lick_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Controlability_lick_Alignment.npy", truncation=30)
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Controlability_lick_Alignment.npy", truncation=30)
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Controlability Eigenvector Lick CD Alignment", x_value_time=False, ylim=[0, 1])
    Plotting_Utils.plot_scatter_graph(np.array(wt_data)[:, 0], np.array(nx_data)[:, 0], "Controlability Leading Eigenvector Lick CD Alignment", ylim=[-1,1])


def compare_lick_cd_decay_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Lick_CD_Decay.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Lick_CD_Decay.npy")
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Lick CD Decay")


def compare_lick_cd_decay_orthogonal_norm_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "decay_total_norm.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "decay_total_norm.npy")
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Lick CD Decay Orthogonal Norm")

def compare_preparatory_lick_alignment(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "preparatory_cosine_similarity.npy")
    nx_data = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "preparatory_cosine_similarity.npy")
    Plotting_Utils.plot_scatter_graph(wt_data, nx_data, "Preparatory Lick CD Alignment", ylim=[-1,1])

def compare_preparatory_coupling_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Prestim_Coupling.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Prestim_Coupling.npy")
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Preparatory To Lick Dimension Coupling")

    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "Prestim_Coupling_Orthogonal.npy")
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "Prestim_Coupling_Orthogonal.npy")
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Preparatory To Orthogonal Coupling")


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

    Plotting_Utils.plot_dual_line_graph(wt_alignment_vis_1, nx_alignment_vis_1, wt_alignment_vis_2, nx_alignment_vis_2, ["Vis 1 Alignment", "Vis 2 Alignment"], x_value_time=False)
    Plotting_Utils.plot_dual_scatter_graph(wt_alignment_vis_1[:, 0],
                                           nx_alignment_vis_1[:, 0],
                                           wt_alignment_vis_2[:, 0],
                                           nx_alignment_vis_2[:, 0],
                                           ["Vis 1 Leading Alignment", "Vis 2 Leading Alignment"],
                                           ylim=[-1, 1]
                                           )



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

    Plotting_Utils.plot_dual_line_graph(wt_alignment_vis_1,
                                        nx_alignment_vis_1,
                                        wt_alignment_vis_2,
                                        nx_alignment_vis_2,
                                        ["Vis 1 Alignment Observability", "Vis 2 Alignment Observability"],
                                        x_value_time=False,
                                        ylim=[-0.2, 0.5])

    Plotting_Utils.plot_dual_scatter_graph(wt_alignment_vis_1[:, 0],
                                           nx_alignment_vis_1[:, 0],
                                           wt_alignment_vis_2[:, 0],
                                           nx_alignment_vis_2[:, 0],
                                           ["Vis 1 Leading Alignment Observability", "Vis 2 Leading Alignment Observability"],
                                           ylim=[-1, 1]
                                           )



def compare_direct_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data_1 = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "vis_1_direct_projection.npy")
    nx_data_1 = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "vis_1_direct_projection.npy")
    wt_data_2 = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "vis_2_direct_projection.npy")
    nx_data_2 = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "vis_2_direct_projection.npy")
    Plotting_Utils.plot_dual_scatter_graph(wt_data_1, nx_data_1, wt_data_2, nx_data_2, ["Vis 1 Direct Alignment", "Vis 2 Direct Alignment"],  ylim=[-1,1])


def compare_stimuli_lick_cd_alignment(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data_1 = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "vis_1_cosine_simmilarity.npy")
    nx_data_1 = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "vis_1_cosine_simmilarity.npy")
    wt_data_2 = Data_Loading_Functions.load_distribution_means(wt_session_list, wt_output_root, "vis_2_cosine_simmilarity.npy")
    nx_data_2 = Data_Loading_Functions.load_distribution_means(neurexin_session_list, neurexin_output_root, "vis_2_cosine_simmilarity.npy")
    Plotting_Utils.plot_dual_scatter_graph(wt_data_1, nx_data_1, wt_data_2, nx_data_2, ["Vis 1 Direct Alignment", "Vis 2 Direct Alignment"],  ylim=[-1,1])



def compare_random_to_lick_coupling_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):

    bin_range=20
    bin_size=1
    x_values = list(range(-bin_range, bin_range, bin_size))

    # Load data
    wt_data = Data_Loading_Functions.load_distributions(wt_session_list, wt_output_root, "coupling_effect.npy", bin_range, bin_size)
    nx_data = Data_Loading_Functions.load_distributions(neurexin_session_list, neurexin_output_root, "coupling_effect.npy", bin_range, bin_size)
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Random Vector Lick CD Coupling", x_value_time=False, set_x_values=x_values)


def compare_covariance_to_lick_alignment_groups(wt_session_list, wt_output_root, neurexin_session_list, neurexin_output_root):
    wt_data = Data_Loading_Functions.load_data(wt_session_list, wt_output_root, "covariance_eigenvector_lick_cd_alignment.npy", truncation=30)
    nx_data = Data_Loading_Functions.load_data(neurexin_session_list, neurexin_output_root, "covariance_eigenvector_lick_cd_alignment.npy", truncation=30)
    Plotting_Utils.plot_line_graph(wt_data, nx_data, "Covariance Eigenvector Lick CD Alignment", x_value_time=False, ylim=[0, 1.1])
    Plotting_Utils.plot_scatter_graph(np.array(wt_data)[:, 0], np.array(nx_data)[:, 0], "Covariance Leading Eigenvector Lick CD Alignment", ylim=[0,1.1])


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







