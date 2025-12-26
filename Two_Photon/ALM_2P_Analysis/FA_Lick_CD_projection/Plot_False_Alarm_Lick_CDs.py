import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import lineStyles
from scipy import stats

import Get_DF
import Get_Lick_CD
import Fa_Lick_CD_Utils
import Get_Data_Tensor


def plot_false_alarm_lick_cd(data_directory_root, session_list, output_directory_root):

    cr_projection_list = []
    fa_projection_list = []

    for session in session_list:

        # Get Session Directory
        session_directory = os.path.join(data_directory_root, session)

        # Load dF Matrix
        df_matrix = Get_DF.load_df_matrix(session_directory)

        # Get Lick CD Excluding FAs (only hits)
        lick_cd = Get_Lick_CD.get_lick_cd(session_directory, df_matrix)

        # Load Behaviour Matrix
        behaviour_matrix = np.load(os.path.join(session_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

        # Get Onsets
        cr_onsets = Fa_Lick_CD_Utils.get_cr_vis_onset_frames(behaviour_matrix)
        fa_onsets = Fa_Lick_CD_Utils.get_fa_vis_onset_frames(behaviour_matrix)

        # Load Frame Rate
        frame_rate = np.load(os.path.join(session_directory, "Frame_Rate.npy"))

        # Get Tensors for CRs and FAs
        start_window = -16
        stop_window = 9
        baseline_correct = True
        baseline_window = 3
        cr_tensor = Get_Data_Tensor. get_activity_tensors(df_matrix, cr_onsets, start_window, stop_window, baseline_correct, baseline_window)
        fa_tensor = Get_Data_Tensor.get_activity_tensors(df_matrix, fa_onsets, start_window, stop_window, baseline_correct, baseline_window)

        # Get Mean
        cr_mean = np.mean(cr_tensor, axis=0)
        fa_mean = np.mean(fa_tensor, axis=0)

        # Project Onto Lick CDs
        cr_projection = np.dot(cr_mean, lick_cd)
        fa_projection = np.dot(fa_mean, lick_cd)

        cr_projection_list.append(cr_projection)
        fa_projection_list.append(fa_projection)

    # Get Group Mean Projection
    cr_projection_list = np.array(cr_projection_list)
    fa_projection_list = np.array(fa_projection_list)
    mean_cr_projection = np.mean(cr_projection_list, axis=0)
    mean_fa_projection = np.mean(fa_projection_list, axis=0)


    # Get SEM
    cr_sem = stats.sem(cr_projection_list, axis=0)
    fa_sem = stats.sem(fa_projection_list, axis=0)

    cr_upper_bound = np.add(mean_cr_projection, cr_sem)
    cr_lower_bound = np.subtract(mean_cr_projection, cr_sem)

    fa_upper_bound = np.add(mean_fa_projection, fa_sem)
    fa_lower_bound = np.subtract(mean_fa_projection, fa_sem)


    # Get Significance
    t_stats, p_values = stats.ttest_rel(cr_projection_list, fa_projection_list, axis=0)
    print("p values", p_values)
    binary_significance = np.where(p_values < 0.05, 1, 0)

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 1.0 / frame_rate)

    # Plot
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.plot(x_values, mean_cr_projection, c='g')
    axis_1.plot(x_values, mean_fa_projection, c='r')

    axis_1.fill_between(x=x_values, y1=cr_lower_bound, y2=cr_upper_bound, color='g', alpha=0.4)
    axis_1.fill_between(x=x_values, y1=fa_lower_bound, y2=fa_upper_bound, color='r', alpha=0.4)

    max_value = np.max(np.concatenate([cr_upper_bound, fa_upper_bound]))
    significance_markers = np.multiply(binary_significance, max_value)
    axis_1.scatter(x_values, significance_markers, c='Grey', marker='s', alpha=binary_significance)

    axis_1.axvline(0, c='k', linestyle='dashed') #linestyle='dashed'

    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_xlabel("Time (s)")
    axis_1.set_ylabel("Lick CD Projection")
    axis_1.set_ylim([-3, 13])



    plt.show()


control_data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]




hom_data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"
hom_session_list = [
    r"64.1B\2024_09_09_Switching",
    r"70.1A\2024_09_09_Switching",
    r"70.1B\2024_09_12_Switching",
    r"72.1E\2024_08_23_Switching",
]


plot_false_alarm_lick_cd(hom_data_directory, hom_session_list, None)
plot_false_alarm_lick_cd(control_data_directory, control_session_list, None)