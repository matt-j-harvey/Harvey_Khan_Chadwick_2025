import os
import numpy as np
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

import Two_Photon_Utils
import Get_Data_Tensor
import View_PSTH



def asses_cell_signficance(tensor, comparison_window_start, comparison_window_stop):

    n_neurons = np.shape(tensor)[2]
    significance_vector = np.zeros(n_neurons)
    positive_modulated_cell_indexes = []
    negative_modulated_cell_indexes = []

    for neuron_index in range(n_neurons):
        neuron_values = tensor[:, comparison_window_start:comparison_window_stop, neuron_index]
        neuron_values = np.mean(neuron_values, axis=1)

        t_stat, p_value = stats.ttest_1samp(a=neuron_values, popmean=0)
        p_value = np.nan_to_num(p_value, nan=1)

        if p_value < 0.05:
            significance_vector[neuron_index] = 1

            if t_stat > 0:
                positive_modulated_cell_indexes.append(neuron_index)

            elif t_stat < 0:
                negative_modulated_cell_indexes.append(neuron_index)

    return significance_vector, positive_modulated_cell_indexes, negative_modulated_cell_indexes



def view_lick_mega_raster(session_list, save_directory_root):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(session_list[0], "Frame_Rate.npy"))
    period = float(1) / frame_rate

    start_window = -int(2.5 * frame_rate)
    stop_window = int(1 * frame_rate)

    save_directory = None
    condition_name = "Lick"
    sorting_window_start = int((np.abs(start_window) - frame_rate))
    sorting_window_stop = np.abs(start_window)
    print("sorting_window_start", sorting_window_start)
    print("sorting_window_stop", sorting_window_stop)

    mean_activity_list = []
    sig_mean_activity_list = []
    sig_vector_list = []

    for base_directory in tqdm(session_list):

        # Load DF Data
        df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
        df_matrix = np.transpose(df_matrix)
        df_matrix = stats.zscore(df_matrix, axis=0)
        df_matrix = np.nan_to_num(df_matrix)
        print("df matrix", np.shape(df_matrix))

        # Load Lick Onsets
        lick_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Lick_Onset_Frames.npy"))
        print("lick onsets", len(lick_onsets))

        # Load Frame Rate
        frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
        period = float(1) / frame_rate

        # Get Data Tensor
        start_window = -int(2.5 * frame_rate)
        stop_window = int(1 * frame_rate)
        print("start_window", start_window, "stop_window", stop_window)

        lick_df_tensor = Get_Data_Tensor.get_data_tensor(df_matrix,
                                                         lick_onsets,
                                                         start_window,
                                                         stop_window,
                                                         baseline_correction=True,
                                                         baseline_start=0,
                                                         baseline_stop=5)

        # Get Mean Activity
        mean_activity = np.nanmean(lick_df_tensor, axis=0)

        # Test Significance
        significance_map = View_PSTH.test_signficance_one_sided(lick_df_tensor)

        significance_vector, positive_cell_indexes, negative_cell_indexes = asses_cell_signficance(lick_df_tensor, sorting_window_start, sorting_window_stop)
        sig_vector_list.append(significance_vector)

        # Save Sig Results
        save_directory = os.path.join(base_directory, "Cell Significance Testing")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        np.save(os.path.join(save_directory, "lick_mean_activity.npy"), mean_activity)
        np.save(os.path.join(save_directory, "significance_vector.npy"), significance_vector)
        np.save(os.path.join(save_directory, "positive_cell_indexes.npy"), positive_cell_indexes)
        np.save(os.path.join(save_directory, "negative_cell_indexes.npy"), negative_cell_indexes)

        # Get Activity Thresholded By Signficance
        sig_activity = np.multiply(mean_activity, significance_map)

        mean_activity_list.append(mean_activity)
        sig_mean_activity_list.append(sig_activity)

    grand_mean_activity = np.hstack(mean_activity_list)
    grand_sig_activity = np.hstack(sig_mean_activity_list)
    grand_sig_vector = np.concatenate(sig_vector_list)

    print("grand_mean_activity", np.shape(grand_mean_activity))
    print("grand_sig_activity", np.shape(grand_sig_activity))
    print("grand_sig_vector", np.shape(grand_sig_vector))

    window_sig_cells = np.multiply(grand_mean_activity, grand_sig_vector)
    print("window_sig_cells", np.shape(window_sig_cells))

    # Create Save Directory
    save_directory = os.path.join(save_directory_root, "Lick_Modulation")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    View_PSTH.view_single_psth_sig_pre_computed(grand_mean_activity,
                                                grand_sig_activity,
                                                window_sig_cells,
                                                start_window,
                                                stop_window,
                                                period,
                                                save_directory,
                                                condition_name,
                                                sorting_window_start,
                                                sorting_window_stop,
                                                plot_titles=["Mean Activity",
                                                             "Signficant Timepoints",
                                                             "Signficant Neurons"])


    np.save(os.path.join(save_directory, "grand_mean_activity.npy"), grand_mean_activity)
    np.save(os.path.join(save_directory, "grand_sig_activity.npy"), grand_sig_activity)
    np.save(os.path.join(save_directory, "grand_sig_vector.npy"), grand_sig_vector)




session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]

save_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results"



session_list = [
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\72.3C\2024_09_10_Switching",
]


view_lick_mega_raster(session_list, save_directory)


session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK64.1B\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1A\2024_09_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1B\2024_09_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK72.1E\2024_08_23_Switching",
]

save_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Hom_Results"