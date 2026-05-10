import GLM_Utils
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

def get_roi_mean(regressor, atlas, roi_list):

    # Load Atlas
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()

    # Get Brain Pixels
    n_mice, n_timepoints, image_height, image_width = np.shape(regressor)
    regressor = np.reshape(regressor, (n_mice, n_timepoints, image_height * image_width))
    regressor = regressor[:, :, indicies]
    regressor = np.squeeze(regressor)

    print("regressor", np.shape(regressor))
    # Get ROI Pixels
    roi_pixels = GLM_Utils.get_roi_pixels(atlas, roi_list)


    regressor = regressor[:, :, roi_pixels]
    print("regressor roi", regressor)
    regressor = np.nanmean(regressor, axis=2)
    return regressor


def baseline_correct_regressors(regressor):

    baseline_corrected_regressor = []
    for mouse in regressor:
        mouse_baseline = mouse[0:14]
        mouse_baseline = np.mean(mouse_baseline, axis=0)
        mouse = np.subtract(mouse, mouse_baseline)
        baseline_corrected_regressor.append(mouse)
    baseline_corrected_regressor = np.array(baseline_corrected_regressor)
    return baseline_corrected_regressor


def compare_contribution_over_learning(pre_output_root,
                                       pre_start_window,
                                       pre_stop_window,
                                       post_output_root,
                                       post_start_window,
                                       post_stop_window):

    # Load Atlas
    atlas = GLM_Utils.load_atlas()
    atlas = np.abs(atlas)

    # Load Coefs
    pre_activity = np.load(os.path.join(pre_output_root, "Group_Coefs", "Group_Mousecam_Activity_Vis_2.npy"))
    post_activity = np.load(os.path.join(post_output_root, "Group_Coefs", "Group_Mousecam_Activity_Vis_2.npy"))
    print("pre_activity", np.shape(pre_activity))
    print("post_activity", np.shape(post_activity))

    # Get ROI Means
    pre_roi_mean = get_roi_mean(pre_activity, atlas, [14,15,16])
    post_roi_mean = get_roi_mean(post_activity, atlas, [14,15,16])

    # Baseline Correct
    pre_roi_mean = baseline_correct_regressors(pre_roi_mean)
    post_roi_mean = baseline_correct_regressors(post_roi_mean)
    print("pre_roi_mean", np.shape(pre_roi_mean))
    print("post_roi_mean", np.shape(post_roi_mean))

    group_pre_mean = np.mean(pre_roi_mean, axis=0)
    plt.plot(group_pre_mean)
    plt.show()

    # Get Time Window Means
    window_size = 27
    pre_stim_window_start = np.abs(pre_start_window)
    pre_stim_window_stop = pre_stim_window_start + window_size
    pre_roi_mean_window = np.nanmean(pre_roi_mean[:, pre_stim_window_start:pre_stim_window_stop], axis=1)
    post_roi_mean_window = np.nanmean(post_roi_mean[:, np.abs(post_start_window):np.abs(post_start_window) + window_size], axis=1)

    t_stat, p_value = stats.ttest_rel(pre_roi_mean_window, post_roi_mean_window)
    print("t_stat", t_stat, "p_value", p_value)

    print("pre_roi_mean_window", pre_roi_mean_window)
    print("post_roi_mean_window", post_roi_mean_window)

    figure_1 = plt.figure(figsize=(2.5,5))
    axis_1 = figure_1.add_subplot(1,1,1)

    n_mice = len(pre_roi_mean_window)
    for mouse_index in range(n_mice):
        axis_1.plot([0, 1], [pre_roi_mean_window[mouse_index], post_roi_mean_window[mouse_index]])
        axis_1.scatter([0, 1], [pre_roi_mean_window[mouse_index], post_roi_mean_window[mouse_index]])

    axis_1.set_ylim([-0.1, 0.45])
    axis_1.set_xlim([-0.2, 1.2])
    axis_1.spines[['right', 'top']].set_visible(False)

    plt.show()




control_pre_output_root = r"C:\Neurexin_GLM\Intermediate_Learning\Controls"
control_post_output_root = r"C:\Neurexin_GLM\Post_Learning\Controls"

hom_pre_output_root = r"C:\Neurexin_GLM\Intermediate_Learning\Homs"
hom_post_output_root = r"C:\Neurexin_GLM\Post_Learning\Homs"

max_mousecam_components = 500

frame_period = 37

pre_start_window_ms = -2500
pre_stop_window_ms = 2500
pre_start_window = int(pre_start_window_ms/frame_period)
pre_stop_window = int(pre_stop_window_ms/frame_period)

post_start_window_ms = -2800
post_stop_window_ms = 2500
post_start_window = int(pre_start_window_ms/frame_period)
post_stop_window = int(pre_stop_window_ms/frame_period)


compare_contribution_over_learning(control_pre_output_root,
                                   pre_start_window,
                                   pre_stop_window,
                                   control_post_output_root,
                                   post_start_window,
                                   post_stop_window)

compare_contribution_over_learning(hom_pre_output_root,
                                   pre_start_window,
                                   pre_stop_window,
                                   hom_post_output_root,
                                   post_start_window,
                                   post_stop_window)