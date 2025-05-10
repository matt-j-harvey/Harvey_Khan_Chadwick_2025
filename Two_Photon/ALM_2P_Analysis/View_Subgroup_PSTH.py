import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import Get_Data_Tensor
import Get_Mean_SEM_Bounds
import View_PSTH




def view_subgroup_psth(session_list, cell_subgroup, onsets_file, start_window, stop_window, sorting_window_start, sorting_window_stop):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(session_list[0], "Frame_Rate.npy"))
    period = 1.0 / frame_rate

    mean_activity_list = []
    sig_activity_list = []
    sig_cell_list = []

    for base_directory in session_list:

        # Load DF Matrix
        df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
        df_matrix = np.transpose(df_matrix)
        df_matrix = stats.zscore(df_matrix, axis=0)
        df_matrix = np.nan_to_num(df_matrix)
        print("df matrix", np.shape(df_matrix))

        # Load Onsets
        onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", onsets_file + ".npy"))

        # Load Selected Neurons
        selected_neurons = np.load(os.path.join(base_directory, "Cell Significance Testing", cell_subgroup + ".npy"))
        df_matrix = df_matrix[:, selected_neurons]
        print("df matrix", np.shape(df_matrix))

        # Get Tensors
        tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onsets_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)

        # Get PSTH
        mean_activity, sig_activity = View_PSTH.compute_single_raster(tensor)
        print("Sig activity", np.shape(sig_activity))
        sig_window_activity = View_PSTH.test_signficance_one_sided_window(tensor, window_start=18, window_stop=24)

        # Add To List
        mean_activity_list.append(mean_activity)
        sig_activity_list.append(sig_activity)
        sig_cell_list.append(sig_window_activity)



    mean_activity_list = np.hstack(mean_activity_list)
    sig_activity_list = np.hstack(sig_activity_list)
    sig_cell_list = np.hstack(sig_cell_list)

    View_PSTH.view_single_psth_sig_pre_computed(mean_activity_list,
                                      sig_activity_list,
                                      sig_cell_list,
                                      start_window,
                                      stop_window,
                                      period,
                                      save_directory=None,
                                      condition_name=cell_subgroup,
                                      sorting_window_start=sorting_window_start,
                                      sorting_window_stop=sorting_window_stop,
                                      plot_titles=None,
                                    magnitude=0.6)

    #View_PSTH.view_single_psth(tensor, start_window, stop_window, period, save_directory, "CR", 18, 24)
    #print("lick onsets", len(lick_onsets))





session_list = [
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\72.3C\2024_09_10_Switching",
]


start_window = -18
stop_window = 18


cell_subgroup = "negative_cell_indexes"
#onsets_file = "odour_2_cued_onsets"
#onsets_file = "odour_context_stable_vis_2_onsets"

onsets_file = "visual_context_stable_vis_2_onsets"
#cell_subgroup = "positive_cell_indexes"
view_subgroup_psth(session_list, cell_subgroup, onsets_file, start_window, stop_window, sorting_window_start=19, sorting_window_stop=24)
