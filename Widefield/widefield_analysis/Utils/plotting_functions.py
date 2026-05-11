import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from widefield_analysis.utils import widefield_utils, figure_colourmaps, plotting_utils



def create_image(data, colourmap, atlas_pixels, background_pixels):

    # Convert To Colour
    data = colourmap.to_rgba(data)

    # Set Atlas Outlines To White
    data[atlas_pixels] = (1, 1, 1, 1)

    # Set Background To Black
    data[background_pixels] = (1, 1, 1, 0)
    return data


def plot_roi_activity(wt_full_activity, nx_full_activity, save_directory, start_window, stop_window, roi_list, graph_name, ylim=None):

    # Load Atlas
    atlas = widefield_utils.load_atlas()
    atlas = np.abs(atlas)

    # Get Mean in ROIs
    wt_roi = widefield_utils.get_roi_mean(wt_full_activity, atlas, roi_list)
    nx_roi = widefield_utils.get_roi_mean(nx_full_activity, atlas, roi_list)

    # Plot Linegraph
    plotting_utils.plot_linegraph(wt_roi, nx_roi, start_window, stop_window, save_directory, graph_name, ylim=ylim)




def plot_roi_activity_genotype_learning(wt_int,
                                        wt_post,
                                        nx_int,
                                        nx_post,
                                        save_directory,
                                        roi_list,
                                        comparison_window_start, comparison_window_stop,
                                        graph_name):

    # Load Atlas
    atlas = widefield_utils.load_atlas()
    atlas = np.abs(atlas)

    # Get Mean in ROIs
    wt_int = widefield_utils.get_roi_mean(wt_int, atlas, roi_list)
    wt_post = widefield_utils.get_roi_mean(wt_post, atlas, roi_list)
    nx_int = widefield_utils.get_roi_mean(nx_int, atlas, roi_list)
    nx_post = widefield_utils.get_roi_mean(nx_post, atlas, roi_list)
    print("wt int roi", np.shape(wt_int))

    # Get Mean in Time Windows
    wt_int = np.mean(wt_int[:, comparison_window_start:comparison_window_stop], axis=1)

    wt_post = np.mean(wt_post[:, comparison_window_start:comparison_window_stop], axis=1)
    nx_int = np.mean(nx_int[:, comparison_window_start:comparison_window_stop], axis=1)
    nx_post = np.mean(nx_post[:, comparison_window_start:comparison_window_stop], axis=1)

    # Plot Scatter
    plotting_utils.plot_genotype_time_scatter(wt_int, wt_post, nx_int, nx_post, save_directory, graph_name)