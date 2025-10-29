import numpy as np
import os
from scipy import stats
import matplotlib.pyplot as plt

import Plot_PSTH
import ALM_Analysis_Utils
import Plot_Swarmplot

def get_lick_tuning(data_root, session, output_root):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(output_root, session, "df_over_f_matrix.npy"))

    # Load Lick Onsets
    lick_onsets = np.load(os.path.join(output_root, session, "Behaviour", "Correct_Lick_Onset_Frames.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))

    # Get Data Tensor
    start_window = -int(2 * frame_rate)
    stop_window = int(1 * frame_rate)
    print("start_window", start_window, "stop_window", stop_window)

    lick_df_tensor = ALM_Analysis_Utils.get_data_tensor(df_matrix,
                                                        lick_onsets,
                                                        start_window,
                                                        stop_window,
                                                        baseline_correction=True,
                                                        baseline_start=0,
                                                        baseline_stop=5)

    # Get Mean Response
    mean_response = np.mean(lick_df_tensor, axis=0)

    # Get Significance
    preceeding_window_start = int(abs(start_window) - frame_rate)
    preceeding_window_stop = abs(start_window)
    lick_preceeding_tensor = lick_df_tensor[:, preceeding_window_start:preceeding_window_stop]
    lick_preceeding_mean = np.mean(lick_preceeding_tensor, axis=1)
    t_stats, p_values = stats.ttest_1samp(lick_preceeding_mean, popmean=0)
    print("t_stats", len(t_stats))

    # Get Modulated Indicies
    sig_modulated_cells = np.where(p_values < 0.05)[0]
    positively_modulated_cells = []
    negatively_modulated_cells = []
    for cell_index in sig_modulated_cells:
        print("cell_index", cell_index)
        cell_t_stat = t_stats[cell_index]

        print("cell_t_stat", cell_t_stat)
        if cell_t_stat > 0:
            positively_modulated_cells.append(cell_index)
        elif cell_t_stat < 0:
            negatively_modulated_cells.append(cell_index)

    # Save Response and Significance Values
    save_directory = os.path.join(output_root, session, "Lick_Coding")
    np.save(os.path.join(save_directory, "Mean_response.npy"), mean_response)
    np.save(os.path.join(save_directory, "start_window.npy"), start_window)
    np.save(os.path.join(save_directory, "stop_window.npy"), stop_window)
    np.save(os.path.join(save_directory, "p_values.npy"), p_values)
    np.save(os.path.join(save_directory, "t_stats.npy"), t_stats)
    np.save(os.path.join(save_directory, "positively_modulated_cells.npy"), positively_modulated_cells)
    np.save(os.path.join(save_directory, "negatively_modulated_cells.npy"), negatively_modulated_cells)

    """
    Plot_PSTH.view_mean_psth(mean_response,
                  start_window,
                  stop_window,
                  None,
                  save_directory,
                  "Lick Tuning",
                  0,
                  -1,
                  magnitude=None)
    """


def plot_group_lick_tuning(data_root, session_list, output_root):

    group_raster = []

    for session in session_list:

        # Load lick Tensor
        lick_tensor = np.load(os.path.join(output_root, session, "Lick_Coding", "Mean_response.npy"))

        # Load Sig Values
        p_values = np.load(os.path.join(output_root, session, "Lick_Coding", "p_values.npy"))
        non_sig_modulated_indicies = np.where(p_values >= 0.05)
        lick_tensor[:, non_sig_modulated_indicies] = 0

        print("lick tensor", np.shape(lick_tensor))
        group_raster.append(lick_tensor)

    group_raster  = np.hstack(group_raster)

    # Load Start and Stop Window
    start_window = np.load(os.path.join(output_root, session, "Lick_Coding", "start_window.npy"))
    stop_window = np.load(os.path.join(output_root, session, "Lick_Coding", "stop_window.npy"))
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))


    save_directory = os.path.join(output_root, "Group_Plots")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    Plot_PSTH.view_mean_psth(group_raster,
                             start_window,
                             stop_window,
                             frame_rate,
                             save_directory,
                             "Lick Tuning",
                             0,
                             -1,
                             magnitude=None)



def plot_group_pichart(session_list, output_root):

    total_cells = 0
    total_negative_cells = 0
    total_positive_cells = 0
    total_unmodulated_cells = 0

    for session in session_list:
        negative_celL_indicies = np.load(os.path.join(output_root, session, "Lick_Coding", "negatively_modulated_cells.npy"))
        positive_cell_indicies = np.load(os.path.join(output_root, session, "Lick_Coding", "positively_modulated_cells.npy"))
        all_cell_vector = np.load(os.path.join(output_root, session, "Lick_Coding", "p_values.npy"))

        session_negative_cells = len(negative_celL_indicies)
        session_positive_cells = len(positive_cell_indicies)
        session_n_cells = len(all_cell_vector)
        session_unmodulated_cells = session_n_cells - (session_positive_cells + session_negative_cells)

        # Add Cells to Pool
        total_cells += session_n_cells
        total_positive_cells += session_positive_cells
        total_negative_cells += session_negative_cells
        total_unmodulated_cells += session_unmodulated_cells


    # Pi Chart of Pooled Cells
    print("total_cells", total_cells)
    print("total_positive_cells", total_positive_cells, float(total_positive_cells) / total_cells * 100)
    print("total_negative_cells", total_negative_cells, float(total_negative_cells) / total_cells * 100)
    print("total_unmodulated_cells", total_unmodulated_cells, float(total_unmodulated_cells) / total_cells * 100)

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

    save_directory = os.path.join(output_root, "Group_Plots")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Group_Lick_Raster.png"))
    plt.close()



def plot_group_scatter(session_list, output_root):

    negative_fraction_list = []
    positive_fraction_list = []

    for session in session_list:

        negative_celL_indicies = np.load(os.path.join(output_root, session, "Lick_Coding", "negatively_modulated_cells.npy"))
        positive_cell_indicies = np.load(os.path.join(output_root, session, "Lick_Coding", "positively_modulated_cells.npy"))
        all_cell_vector = np.load(os.path.join(output_root, session, "Lick_Coding", "p_values.npy"))

        session_negative_cells = len(negative_celL_indicies)
        session_positive_cells = len(positive_cell_indicies)
        session_n_cells = len(all_cell_vector)

        session_positive_fraction = (float(session_positive_cells) / float(session_n_cells)) * 100
        session_negative_fraction = (float(session_negative_cells) / float(session_n_cells)) * 100

        negative_fraction_list.append(session_negative_fraction)
        positive_fraction_list.append(session_positive_fraction)


    # Plot Swarmplots
    negative_cmap = plt.get_cmap("Blues")
    positive_cmap = plt.get_cmap("Reds")

    save_directory = os.path.join(output_root, "Group_Plots")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    Plot_Swarmplot.swarmplot(negative_fraction_list,
                             positive_fraction_list,
                             negative_cmap,
                             positive_cmap,
                             save_directory=save_directory,
                             plot_name="Across Mouse Fractions",
                             y_lim=[0,100],
                             x_labels=["Negatively Modulated","Positively Modulated"],
                             y_label="Fraction Modulated",
                             plot_significance=False,
                             plot_confidence_interval=True)
