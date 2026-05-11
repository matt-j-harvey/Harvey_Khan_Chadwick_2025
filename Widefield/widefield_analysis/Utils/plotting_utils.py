import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection

from widefield_analysis.utils import widefield_utils, figure_colourmaps



def plot_linegraph(wt_activity, nx_activity, start_window, stop_window, save_directory, graph_name, ylim=None):

    # Get Mean and SD
    wt_mean, wt_lower_bound, wt_upper_bound = widefield_utils.get_mean_sd(wt_activity)
    nx_mean, nx_lower_bound, nx_upper_bound = widefield_utils.get_mean_sd(nx_activity)

    if ylim is None:
        max_value = np.max(np.concatenate([wt_upper_bound, nx_upper_bound]))
        max_value = max_value * 1.1

    else:
        max_value = ylim[1] * 0.97

    # Load Colour Maps
    wt_cmap = figure_colourmaps.get_wt_colourmap()
    nx_cmap = figure_colourmaps.get_nx_colourmap()

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    # Create Figure
    figure_1 = plt.figure(figsize=(10, 5))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    # Plot Data
    colourmap_intensity = 0.7
    axis_1.plot(x_values, wt_mean, c=wt_cmap(colourmap_intensity))
    axis_1.plot(x_values, nx_mean, c=nx_cmap(colourmap_intensity))

    axis_1.fill_between(x=x_values, y1=wt_lower_bound, y2=wt_upper_bound, alpha=0.3, color=wt_cmap(colourmap_intensity))
    axis_1.fill_between(x=x_values, y1=nx_lower_bound, y2=nx_upper_bound, alpha=0.3, color=nx_cmap(colourmap_intensity))

    # Add Significance
    t_stats, p_values = stats.ttest_ind(wt_activity, nx_activity, axis=0)
    binary_signficance = np.where(p_values < 0.05, 1, 0)
    signficance_markers = np.multiply(binary_signficance, max_value)
    axis_1.scatter(x_values, signficance_markers, alpha=binary_signficance, color='Grey', marker='s')

    # Set Figure Details
    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_xlabel("Time (ms)")
    axis_1.set_ylabel("Z Score dF/F")
    axis_1.spines[['right', 'top']].set_visible(False)

    if ylim is not None:
        axis_1.set_ylim(ylim)

    # Save Figure
    plt.savefig(os.path.join(save_directory, graph_name + ".svg"))
    plt.close()



def plot_genotype_time_scatter(wt_pre, wt_post, nx_pre, nx_post, save_directory, graph_name):

    # Load Colour Maps
    wt_cmap = figure_colourmaps.get_wt_colourmap()
    nx_cmap = figure_colourmaps.get_nx_colourmap()

    # Create Figure
    figure_1 = plt.figure(figsize=(5, 5))
    axis_1 = figure_1.add_subplot(1, 1, 1)

    colourmap_intensity = 0.7

    nx_learning_t, nx_learning_p = stats.ttest_rel(nx_pre, nx_post)
    print("nx_learning_p", nx_learning_p)

    genotype_pre_t, genotype_pre_p = stats.ttest_ind(wt_pre, nx_pre)
    print("genotype_pre_p", genotype_pre_p)

    genotype_post_t, genotype_post_p = stats.ttest_ind(wt_post, nx_post)
    print("genotype_post_p", genotype_post_p)

    wt_learning_t, wt_learning_p = stats.ttest_rel(wt_pre, wt_post)
    print("wt learning p ", wt_learning_p)

    # Plot WT
    n_wt_mice = len(wt_pre)
    for wt_mouse_index in range(n_wt_mice):
        mouse_x_values = [1,2]
        mouse_y_values = [wt_pre[wt_mouse_index], wt_post[wt_mouse_index]]
        axis_1.plot(mouse_x_values, mouse_y_values, c=wt_cmap(colourmap_intensity))
        axis_1.scatter(mouse_x_values, mouse_y_values, c=wt_cmap(colourmap_intensity))

    # Plot NX
    n_nx_mice = len(nx_pre)
    for nx_mouse_index in range(n_nx_mice):
        mouse_x_values = [3,4]
        mouse_y_values = [nx_pre[nx_mouse_index], nx_post[nx_mouse_index]]
        axis_1.plot(mouse_x_values, mouse_y_values, c=nx_cmap(colourmap_intensity))
        axis_1.scatter(mouse_x_values, mouse_y_values, c=nx_cmap(colourmap_intensity))


    axis_1.set_xticks([1,2,3,4], labels=["WT Int", "WT Post", "NX Int", "NX Post"])
    axis_1.set_xlim([0.5, 4.5])

    axis_1.set_ylabel("Z Score dF/F")
    axis_1.spines[['right', 'top']].set_visible(False)

    # Save Figure
    plt.savefig(os.path.join(save_directory, graph_name + ".svg"))
    plt.close()

