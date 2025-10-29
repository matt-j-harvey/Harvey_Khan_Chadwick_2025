import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

import ALM_Analysis_Utils


def plot_piechart(lick_p_values, lick_t_stats, threshold=0.05):

    total_cells = len(lick_p_values)
    sig_modulated_cells = np.where(lick_p_values < threshold)
    sig_modulated_t_stats = lick_t_stats[sig_modulated_cells]
    total_positive_cells = np.sum(np.where(sig_modulated_t_stats > 0, 1, 0))
    total_negative_cells = np.sum(np.where(sig_modulated_t_stats < 0, 1, 0))

    total_unmodulated_cells = total_cells - (total_positive_cells + total_negative_cells)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    axis_1.pie([total_unmodulated_cells,
                total_positive_cells,
                total_negative_cells],
               labels=["Unmodulated Cells",
                       "Positive Cells",
                       "Negative Cells"],
               colors=['Grey', 'crimson', 'dodgerblue'],
               autopct='%1.1f%%')
    axis_1.set_title("Pooled Cell Fractions")
    plt.show()



def quantify_lick_tuning_session(data_root, session, output_root):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(output_root, session, "df_over_f_matrix.npy"))

    # Load Lick Onsets
    lick_onsets = np.load(os.path.join(output_root, session, "Behaviour", "Correct_Lick_Onset_Frames.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))

    # Get Data Tensor
    start_window = -int(2 * frame_rate)
    stop_window = 0
    print("start_window", start_window, "stop_window", stop_window)

    lick_df_tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix,
                                                        lick_onsets,
                                                        start_window,
                                                        stop_window,
                                                        baseline_correction=True,
                                                        baseline_start=0,
                                                        baseline_stop=5)
    print("lick df tensor", np.shape(lick_df_tensor))

    # Get Mean in 1S Preceeding
    lick_df_tensor_preceeding = np.mean(lick_df_tensor[:, -int(frame_rate):], axis=1)
    print("lick df tensor preceeding", np.shape(lick_df_tensor_preceeding))

    # Calculate Significance
    t_stats, p_value = stats.ttest_1samp(lick_df_tensor_preceeding, popmean=0, axis=0)

    # Save These
    save_directory = os.path.join(output_root, session, "Lick_Coding")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory), "Lick_Tuning_P_Values.npy", p_value)
    np.save(os.path.join(save_directory), "Lick_Tuning_T_Stats.npy", t_stats)

    plot_piechart(p_value, t_stats, threshold=0.05)