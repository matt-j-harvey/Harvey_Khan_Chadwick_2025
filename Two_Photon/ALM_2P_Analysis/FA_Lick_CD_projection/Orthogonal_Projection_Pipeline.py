import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Get_DF
import Get_Data_Tensor
import Get_Lick_CD
import Orthogonal_Subspace_Projection


def get_mean_and_sem(data_matrix):
    data_mean = np.mean(data_matrix, axis=0)
    data_sem = stats.sem(data_matrix, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound


def plot_graph(condition_1_data, condition_2_data, x_values):

    # Get Mean and SEM Bounds
    condition_1_data_mean, condition_1_lower_bound, condition_1_upper_bound = get_mean_and_sem(condition_1_data)
    condition_2_data_mean, condition_2_lower_bound, condition_2_upper_bound = get_mean_and_sem(condition_2_data)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, condition_1_data_mean, c='b')
    axis_1.fill_between(x=x_values, y1=condition_1_lower_bound, y2=condition_1_upper_bound, color='b', alpha=0.5)

    axis_1.plot(x_values, condition_2_data_mean, c='g')
    axis_1.fill_between(x=x_values, y1=condition_2_lower_bound, y2=condition_2_upper_bound, color='g', alpha=0.5)

    axis_1.axvline(0, c='k', linestyle='dashed')

    axis_1.set_ylim([-3, 11])
    plt.show()


def get_hit_onsets(behaviour_matrix):

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        reaction_time = trial[23]
        onset_frame = trial[18]

        if trial_type == 1:
            if correct == 1:
                onset_list.append(onset_frame)

    return onset_list





def get_cr_onsets(behaviour_matrix):

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        reaction_time = trial[23]
        onset_frame = trial[18]

        if trial_type == 2:
            if correct == 1:
                onset_list.append(onset_frame)

    return onset_list




def get_fa_onsets(behaviour_matrix):

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        reaction_time = trial[23]
        onset_frame = trial[18]

        if trial_type == 2:
            if correct == 0:
                onset_list.append(onset_frame)

    return onset_list


def orthogonal_projection_pipeline(data_directory_root, session_list, output_directory):

    # 2.5s Before
    # 2.5s Post
    start_window = -16
    stop_window = 16

    lick_cd_projection_list = []
    orthogonal_projection_list = []

    for session in session_list:

        # Get Session Directory
        session_directory = os.path.join(data_directory_root, session)

        # Load Behaviour Matrix
        behaviour_matrix = np.load(os.path.join(session_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

        # Get Hit Onsets
        #selected_onsets = get_hit_onsets(behaviour_matrix)
        #selected_onsets = get_cr_onsets(behaviour_matrix)
        selected_onsets = get_fa_onsets(behaviour_matrix)

        # Load dF Matrix
        df_matrix = Get_DF.load_df_matrix(session_directory)
        df_matrix = np.nan_to_num(df_matrix)

        # Load Frame Rate
        frame_rate = np.load(os.path.join(session_directory, "Frame_Rate.npy"))

        # Get Tensor
        baseline_correct = True
        baseline_window = 3
        tensor = Get_Data_Tensor.get_activity_tensors(df_matrix, selected_onsets, start_window, stop_window, baseline_correct, baseline_window)
        print("tensor", np.shape(tensor))

        # Get Mean Activity
        mean_activity = np.mean(tensor, axis=0)
        print("mean_activity", np.shape(mean_activity))

        # Load Lick CD
        lick_cd = Get_Lick_CD.get_lick_cd(session_directory, df_matrix)

        # Get Lick CD Projection
        lick_cd_projection = np.dot(mean_activity, lick_cd)

        # Get Orthogonal Projection
        orthogonal_projection = Orthogonal_Subspace_Projection.get_magnitude_of_orthogonal_projection(mean_activity, lick_cd)
        #orthogonal_projection = Orthogonal_Subspace_Projection.get_positive_only_activity(mean_activity, lick_cd)

        lick_cd_projection_list.append(lick_cd_projection)
        orthogonal_projection_list.append(orthogonal_projection)

    # Convert To Arrays
    lick_cd_projection_list = np.array(lick_cd_projection_list)
    orthogonal_projection_list = np.array(orthogonal_projection_list)
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 1.0 / frame_rate)


    #plt.plot(np.mean(orthogonal_projection_list, axis=0))
    #plt.show()
    plot_graph(lick_cd_projection_list, orthogonal_projection_list, x_values)



control_data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
control_output_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Meeting_With_Adil\2025_12_09\Lick_Aligned_Stim_Aligned\Controls"
control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


hom_data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"
hom_output_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Meeting_With_Adil\2025_12_09\Lick_Aligned_Stim_Aligned\Homs"
hom_session_list = [
    r"64.1B\2024_09_09_Switching",
    #r"70.1A\2024_09_09_Switching",
    r"70.1B\2024_09_12_Switching",
    r"72.1E\2024_08_23_Switching",
]




orthogonal_projection_pipeline(control_data_directory, control_session_list, control_output_directory)
orthogonal_projection_pipeline(hom_data_directory, hom_session_list, hom_output_directory)
