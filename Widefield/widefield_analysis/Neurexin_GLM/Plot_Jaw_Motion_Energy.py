import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Session_List
import GLM_Utils




def plot_jaw_motion_traces(control_group, neurexin_group, start_window, stop_window, frame_period):

    # Get Mean and SD
    control_mean, control_lower_bound, control_upper_bound = GLM_Utils.get_mean_sd(control_group)
    neurexin_mean, neurexin_lower_bound, neurexin_upper_bound = GLM_Utils.get_mean_sd(neurexin_group)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_period)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    axis_1.plot(x_values, control_mean, c='b')
    axis_1.plot(x_values, neurexin_mean, c='m')
    axis_1.fill_between(x_values, y1=control_lower_bound, y2=control_upper_bound, color='b', alpha=0.5)
    axis_1.fill_between(x_values, y1=neurexin_lower_bound, y2=neurexin_upper_bound, color='m', alpha=0.5)

    t_stats, p_values = stats.ttest_ind(control_group, neurexin_group, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_scatter = np.multiply(binary_sig, np.max(neurexin_upper_bound))

    axis_1.scatter(x_values, sig_scatter, c='grey', marker='s', alpha=binary_sig)

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.show()


    # Plot Scatter
    # Get Sums
    control_window = np.mean(control_group[:, 28:], axis=1)
    hom_window = np.mean(neurexin_group[:, 28:], axis=1)
    t_stats, p_values = stats.ttest_ind(control_window, hom_window)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.scatter(np.zeros(6), control_window, c='b')
    axis_1.scatter(np.ones(6), hom_window, c='m')

    axis_1.set_xlim([-1,2])
    plt.show()


    print("gentoype window p", p_values)




def jaw_motion_pipeline_odr_2(data_root, session_list, start_window, stop_window, frame_period):

    group_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Get Jaw Motion Energy
            jaw_motion_energy  = np.load(os.path.join(data_root, session, "Mousecam_Analysis", "Mean_Jaw_Motion_Energy.npy"))
            #jaw_motion_energy = stats.zscore(jaw_motion_energy)

            # Get Onsets
            onsets = GLM_Utils.get_odour_2_onsets(behaviour_matrix)

            # Get Data Tensor
            tensor = GLM_Utils.get_data_tensor(jaw_motion_energy, onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)


            # Get Mean
            mean = np.mean(tensor, axis=0)
            mouse_list.append(mean)

        mouse_mean = np.mean(np.array(mouse_list), axis=0)
        group_list.append(mouse_mean)

    # Get Mean and SD
    group_list = np.array(group_list)
    mean, lower_bound, upper_bound = GLM_Utils.get_mean_sd(group_list)
    print("mean", mean)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_period)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, mean, c='b')
    axis_1.fill_between(x_values, y1=lower_bound, y2=upper_bound, color='b', alpha=0.5)



    #axis_1.set_ylim([3, 7])
    plt.show()

    return group_list



def get_group_jaw_motion_energy_across_contexts(data_root, session_list):

    group_mean_vis_2_rel_list = []
    group_mean_vis_2_irrel_list = []

    for mouse in session_list:
        mouse_vis_2_rel_list = []
        mouse_vis_2_irrel_list = []

        for session in mouse:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Get Jaw Motion Energy
            jaw_motion_energy  = np.load(os.path.join(data_root, session, "Mousecam_Analysis", "Mean_Jaw_Motion_Energy.npy"))
            jaw_motion_energy = stats.zscore(jaw_motion_energy)

            # Get Onsets
            vis_2_correct_onsets = GLM_Utils.get_cr_onsets(behaviour_matrix)
            vis_2_irrel_onsets = GLM_Utils.get_irrel_vis_2_onsets(behaviour_matrix)

            # Get Data Tensor
            vis_2_rel_tensor = GLM_Utils.get_data_tensor(jaw_motion_energy, vis_2_correct_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)
            vis_2_irrel_tensor = GLM_Utils.get_data_tensor(jaw_motion_energy, vis_2_irrel_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)

            # Get Mean
            vis_2_rel_mean = np.mean(vis_2_rel_tensor, axis=0)
            vis_2_irrel_mean = np.mean(vis_2_irrel_tensor, axis=0)

            mouse_vis_2_rel_list.append(vis_2_rel_mean)
            mouse_vis_2_irrel_list.append(vis_2_irrel_mean)

        mouse_vis_2_rel_mean = np.mean(np.array(mouse_vis_2_rel_list), axis=0)
        mouse_vis_2_irrel_mean = np.mean(np.array(mouse_vis_2_irrel_list), axis=0)

        group_mean_vis_2_rel_list.append(mouse_vis_2_rel_mean)
        group_mean_vis_2_irrel_list.append(mouse_vis_2_irrel_mean)

    # Get Mean and SD
    group_mean_vis_2_rel_list = np.array(group_mean_vis_2_rel_list)
    group_mean_vis_2_irrel_list = np.array(group_mean_vis_2_irrel_list)

    return group_mean_vis_2_rel_list, group_mean_vis_2_irrel_list



def jaw_motion_pipeline_vis_2(data_root, session_list, start_window, stop_window, frame_period):

    # Get Group Rel and Irel Jaw Motion
    group_mean_vis_2_rel_list, group_mean_vis_2_irrel_list =  get_group_jaw_motion_energy_across_contexts(data_root, session_list)

    # Get Mean and SEM
    rel_mean, rel_lower_bound, rel_upper_bound = GLM_Utils.get_mean_sd(group_mean_vis_2_rel_list)
    irrel_mean, irrel_lower_bound, irrel_upper_bound = GLM_Utils.get_mean_sd(group_mean_vis_2_irrel_list)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_period)

    t_stats, p_values = stats.ttest_rel(group_mean_vis_2_rel_list, group_mean_vis_2_irrel_list, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_scatter = np.multiply(binary_sig, 2.4)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, rel_mean, c='b')
    axis_1.fill_between(x_values, y1=rel_lower_bound, y2=rel_upper_bound, color='b', alpha=0.5)

    axis_1.plot(x_values, irrel_mean, c='grey')
    axis_1.fill_between(x_values, y1=irrel_lower_bound, y2=irrel_upper_bound, color='grey', alpha=0.5)

    axis_1.scatter(x_values, sig_scatter, c='grey', marker='s', alpha=binary_sig)

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_ylim([-0.3, 2.5])

    plt.show()

    # Get Sums
    group_rel_window = np.mean(group_mean_vis_2_rel_list[:, 28:], axis=1)
    group_irrel_window = np.mean(group_mean_vis_2_irrel_list[:, 28:], axis=1)
    print("group_rel_window", group_rel_window)
    print("group_irrel_window", group_irrel_window)
    t_stats, p_values = stats.ttest_rel(group_rel_window, group_irrel_window)
    print("window p", p_values)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    n_mice = len(group_rel_window)
    for mouse_index in range(n_mice):
        axis_1.plot([1,2], [group_irrel_window[mouse_index], group_rel_window[mouse_index]], c='cornflowerblue')
        axis_1.scatter([1,2], [group_irrel_window[mouse_index], group_rel_window[mouse_index]], c='cornflowerblue')
    axis_1.set_ylim([-0.3, 3])
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.show()

    # Plot As Diff
    difference = np.subtract(group_mean_vis_2_rel_list, group_mean_vis_2_irrel_list)
    diff_mean, diff_lower_bound, diff_upper_bound = GLM_Utils.get_mean_sd(difference)


    t_stats, p_values = stats.ttest_1samp(difference, popmean=0, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_scatter = np.multiply(binary_sig, 1.7)

    print("p_values", p_values)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, diff_mean, c='m')
    axis_1.fill_between(x_values, y1=diff_lower_bound, y2=diff_upper_bound, color='m', alpha=0.5)
    axis_1.scatter(x_values, sig_scatter, c='grey', marker='s', alpha=binary_sig)
    axis_1.set_ylim([-0.7, 2])
    plt.show()



def compare_genotype_jaw_motion_energy(control_data_root, control_session_list, hom_data_root, hom_session_list, start_window, stop_window):

    # Get Group Jaw Motion Energy
    control_vis_2_rel, control_vis_2_irrel =  get_group_jaw_motion_energy_across_contexts(control_data_root, control_session_list)
    hom_vis_2_rel, hom_vis_2_irrel =  get_group_jaw_motion_energy_across_contexts(hom_data_root, hom_session_list)

    # Get Mean and SEM
    control_mean, control_lower_bound, control_upper_bound = GLM_Utils.get_mean_sd(control_vis_2_rel)
    hom_mean, hom_lower_bound, hom_upper_bound = GLM_Utils.get_mean_sd(hom_vis_2_rel)

    # Test Signficance
    t_stats, p_values = stats.ttest_ind(control_vis_2_rel, hom_vis_2_rel, axis=0)
    binary_sig = np.where(p_values < 0.05, 1, 0)
    sig_scatter = np.multiply(binary_sig, np.max(hom_upper_bound))

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_period)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, control_mean, c='b')
    axis_1.fill_between(x_values, y1=control_lower_bound, y2=control_upper_bound, color='b', alpha=0.5)

    axis_1.plot(x_values, hom_mean, c='m')
    axis_1.fill_between(x_values, y1=hom_lower_bound, y2=hom_upper_bound, color='m', alpha=0.5)

    axis_1.scatter(x_values, sig_scatter, c='grey', marker='s', alpha=binary_sig)
    #axis_1.set_ylim([-0.7, 1.7])
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.axvline(0, c='k', linestyle='dashed')
    plt.show()


    # Get Sums
    control_window = np.mean(control_vis_2_rel[:, np.abs(start_window):], axis=1)
    hom_window = np.mean(hom_vis_2_rel[:, np.abs(start_window):], axis=1)

    print("group_rel_window", control_window)
    print("group_irrel_window", hom_window)
    t_stats, p_values = stats.ttest_ind(control_window, hom_window)
    #t_stats, p_values = stats.mannwhitneyu(control_window, hom_window)
    print("window p", p_values)


    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.scatter(np.zeros(len(control_window)), control_window, c='cornflowerblue')
    axis_1.scatter(np.ones(len(hom_window)), hom_window, c='m')
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_xlim(-0.5, 1.5)
    axis_1.set_xticks([0, 1], labels=["Wildtype", "Neurexin"])
    plt.show()






frame_period = 36



control_session_list = Session_List.control_switching
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"

hom_session_list = Session_List.neurexin_switching
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"

start_window_ms = -1000
stop_window_ms = 1000
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

"""
jaw_motion_pipeline_vis_2(control_data_root, control_session_list, start_window, stop_window, frame_period)
jaw_motion_pipeline_vis_2(hom_data_root, hom_session_list, start_window, stop_window, frame_period)

control_motion = jaw_motion_pipeline_odr_2(control_data_root, control_session_list, start_window, stop_window, frame_period)
hom_motion = jaw_motion_pipeline_odr_2(hom_data_root, hom_session_list, start_window, stop_window, frame_period)
plot_jaw_motion_traces(control_motion, hom_motion,  start_window, stop_window, frame_period)
"""

compare_genotype_jaw_motion_energy(control_data_root, control_session_list, hom_data_root, hom_session_list, start_window, stop_window)
