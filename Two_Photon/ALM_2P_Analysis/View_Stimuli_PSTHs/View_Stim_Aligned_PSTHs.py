import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

import Get_Data_Tensor

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_significant_responses(data_root, df_matrix, session, trial_type, start_window, stop_window):

    # Load Onsets
    onset_list = np.load(os.path.join(data_root, session, "Stimuli_Onsets", trial_type + "_onsets.npy"))

    # Get Tensor
    tensor = Get_Data_Tensor.get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=5)
    print("tensor", np.shape(tensor))

    # Get Sig Timepoints
    t_stats, p_values = stats.ttest_1samp(tensor, axis=0, popmean=0)
    binary_sig = np.where(p_values <0.05, 1, 0)

    # Get Mean
    mean_response = np.mean(tensor, axis=0)

    # Get Sig Mean
    sig_response = np.multiply(mean_response, binary_sig)

    return sig_response


def sort_raster_by_lick_cd(raster, dimension):
    indicies = np.argsort(dimension)
    raster = raster[:, indicies]
    return raster

def sort_raster(raster, start_window):
    mean_response = np.mean(raster[np.abs(start_window):], axis=0)
    indicies = np.argsort(mean_response)
    raster = raster[:, indicies]
    return raster


def view_raster(raster, start_window, stop_window, title):

    # sort raster
    raster = sort_raster(raster, start_window)

    # Get magnitude
    raster_magnitude = np.percentile(np.abs(raster), q=95)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(np.transpose(raster), cmap="bwr", vmin=-raster_magnitude, vmax=raster_magnitude)

    axis_1.set_title(title)
    forceAspect(axis_1)

    plt.show()


def view_psth_pipeline(data_root, session, output_root, start_winow, stop_window):

    # Load DF
    df_matrix = np.load(os.path.join(data_root, session, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)
    df_matrix = stats.zscore(df_matrix, axis=0)
    print("df matrix", np.shape(df_matrix))

    raster_magnitude = np.percentile(np.abs(df_matrix), q=95)
    plt.imshow(np.transpose(df_matrix), cmap="bwr", vmin=-raster_magnitude, vmax=raster_magnitude)
    forceAspect(plt.gca())
    plt.show()

    raster_means = np.mean(df_matrix, axis=0)
    plt.hist(raster_means)
    plt.show()

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    start_frames = int(np.around(start_winow * frame_rate, 0))
    stop_frames = int(np.around(stop_window * frame_rate, 0))
    print("frame_rate", frame_rate)
    print("start_frames", start_frames)
    print("stop_frame", stop_frames)

    # Load Lick CD
    lick_cd = np.load(os.path.join(data_root, session, "Coding_Dimensions", "Lick_CD.npy"))

    # Get Sig Responses
    vis_context_vis_1_response = get_significant_responses(data_root, df_matrix, session, "visual_context_stable_vis_1", start_frames, stop_frames)
    vis_context_vis_2_response = get_significant_responses(data_root, df_matrix, session, "visual_context_stable_vis_2", start_frames, stop_frames)
    odr_context_vis_1_response = get_significant_responses(data_root, df_matrix, session, "odour_context_stable_vis_1", start_frames, stop_frames)
    odr_context_vis_2_response = get_significant_responses(data_root, df_matrix, session, "odour_context_stable_vis_2", start_frames, stop_frames)

    # Sort Responses
    vis_context_vis_1_response = sort_raster(vis_context_vis_1_response, start_winow)
    vis_context_vis_2_response = sort_raster(vis_context_vis_2_response, start_winow)
    odr_context_vis_1_response = sort_raster(odr_context_vis_1_response, start_winow)
    odr_context_vis_2_response = sort_raster(odr_context_vis_2_response, start_winow)

    # View Responses
    view_raster(vis_context_vis_1_response, start_winow, stop_window, "visual context vis 1")
    view_raster(odr_context_vis_1_response, start_winow, stop_window, "odour context vis 1")
    view_raster(vis_context_vis_2_response, start_winow, stop_window, "visual context vis 2")
    view_raster(odr_context_vis_2_response, start_winow, stop_window, "odour context vis 2")
    print("vis_context_vis_1_response", np.shape(vis_context_vis_1_response))







# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Stim_Aligned_PSTH"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


start_winow = -2
stop_window = 2

for session in control_session_list:
    view_psth_pipeline(data_root, session, output_root, start_winow, stop_window)



