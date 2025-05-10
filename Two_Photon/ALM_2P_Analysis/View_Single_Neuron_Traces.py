import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import Get_Data_Tensor
import Get_Mean_SEM_Bounds
import View_PSTH



def plot_single_neurons(save_directory, tensor, start_window, stop_window, period):

    # Get Data Structure
    n_trials, n_timepoints, n_neurons = np.shape(tensor)

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, period)

    print("Tensor", np.shape(tensor))

    for neuron_index in range(n_neurons):

        # Get Neuron Data
        neuron_data = tensor[:, :, neuron_index]

        # Get Mean and Bounds
        mean, upper_bound, lower_bound = Get_Mean_SEM_Bounds.get_sem_and_bounds(neuron_data)

        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(1,1,1)
        axis_1.plot(x_values, mean)
        axis_1.fill_between(x=x_values, y1=lower_bound, y2=upper_bound, alpha=0.5)

        plt.show()


def view_nm_neuron_fas(session_list):



    psth_list = []

    for base_directory in session_list:

        # Load DF Matrix
        df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
        df_matrix = np.transpose(df_matrix)
        df_matrix = stats.zscore(df_matrix, axis=0)
        df_matrix = np.nan_to_num(df_matrix)
        print("df matrix", np.shape(df_matrix))

        # Load Frame Rate
        frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
        period = 1.0/frame_rate

        # Load Onsets
        #lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"))
        cr_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))

        # Load Negatively Tuned Neurons
        selected_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", selected_group + ".npy"))

        df_matrix = df_matrix[:, negatively_tuned_neurons]
        print("df matrix", np.shape(df_matrix))

        # Get Tensors
        start_window = -18
        stop_window = 18
        cr_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, cr_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)



    for base_directory in session_list:

        # Load DF Matrix
        df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
        df_matrix = np.transpose(df_matrix)
        df_matrix = stats.zscore(df_matrix, axis=0)
        df_matrix = np.nan_to_num(df_matrix)
        print("df matrix", np.shape(df_matrix))

        # Load Frame Rate
        frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
        period = 1.0/frame_rate

        # Load Onsets
        lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"))
        cr_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))

        # Load Negatively Tuned Neurons
        negatively_tuned_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", "negative_cell_indexes.npy"))
        #negatively_tuned_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", "positive_cell_indexes.npy"))
        df_matrix = df_matrix[:, negatively_tuned_neurons]
        print("df matrix", np.shape(df_matrix))

        # Get Tensors
        start_window = -18
        stop_window = 18
        cr_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, cr_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

        save_directory = os.path.join(base_directory, "Single_Neuron_Responses", "Negative_Lick_Tuned_CRs")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)


        View_PSTH.view_single_psth(cr_tensor, start_window, stop_window, period, save_directory, "CR", 18, 24)
        #plot_single_neurons(save_directory, cr_tensor, start_window, stop_window, period)

        print("lick onsets", len(lick_onsets))





session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]

view_nm_neuron_fas(session_list)