import numpy as np
from numpy.linalg import norm

import matplotlib.pyplot as plt
import os

from tqdm import tqdm
from scipy import stats

import Two_Photon_Utils
import Get_Data_Tensor

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def load_df_matrix(base_directory):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(base_directory, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)

    # Z Score
    df_matrix = stats.zscore(df_matrix, axis=0)

    # Remove NaN
    df_matrix = np.nan_to_num(df_matrix)

    # Return
    return df_matrix



def sort_raster(raster, sorting_window_start, sorting_window_stop):

    # Get Mean Response in Sorting Window
    response = raster[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)

    # Get Sorted Indicies
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    # Sort Rasters
    sorted_raster = raster[:, sorted_indicies]

    return sorted_raster




def get_downsampled_lick_trace(base_directory):

    # Load AI Data
    ai_data = np.load(os.path.join(base_directory, "Behaviour", "AI_Matrix.npy"))
    print("ai_data", np.shape(ai_data))

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))
    print("stack_onsets", len(stack_onsets))

    # Load Stimuli Dict
    stimuli_dict = Two_Photon_Utils.load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[:, stimuli_dict["Lick"]]

    # Downsample Lick Trace
    downsampled_lick_trace = downsample_ai_trace(lick_trace, stack_onsets)


    return downsampled_lick_trace




def downsample_ai_trace(ai_trace, stack_onsets):

    # Get Average Stack Duration
    stack_duration_list = np.diff(stack_onsets)
    mean_stack_duration = int(np.mean(stack_duration_list))

    downsampled_trace = []
    n_stacks = len(stack_onsets)
    for stack_index in range(n_stacks-1):
        stack_start = stack_onsets[stack_index]
        stack_stop = stack_onsets[stack_index + 1]
        stack_data = ai_trace[stack_start:stack_stop]
        stack_data = np.mean(stack_data)
        downsampled_trace.append(stack_data)

    # Add Last
    final_data = ai_trace[stack_onsets[-1]:stack_onsets[-1] + mean_stack_duration]
    final_data = np.mean(final_data)
    downsampled_trace.append(final_data)

    return downsampled_trace



def get_next_onset(trace, onset, threshold, max_rt):

    index = 0
    above = False
    while above == False:
        instantaneous_value = trace[onset + index]
        if instantaneous_value > threshold:
            return onset + index

        else:
            index += 1
            if index > max_rt:
                return False





def get_prelick_onsets(base_directory, frame_rate, min_rt=0.5, max_rt=2.5, lick_threshold=600):

    # Load Lick Trace
    lick_trace = get_downsampled_lick_trace(base_directory)
    min_rt_frames = min_rt * frame_rate
    max_rt_frames = max_rt * frame_rate

    # Load Visual Hit Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))

    # Load VIs 1 Lick Onsets
    stim_onset_list = []
    lick_onset_list = []
    for onset in vis_1_onsets:
        trial_lick_onset = get_next_onset(lick_trace, onset, lick_threshold, max_rt_frames)

        if trial_lick_onset - onset > min_rt_frames:
            stim_onset_list.append(onset)
            lick_onset_list.append(trial_lick_onset)

    return stim_onset_list, lick_onset_list




def view_psth(tensor, start_window, stop_window, frame_rate, save_directory, condition_name, sorting_window_start, sorting_window_stop):

    # Get PSTHs
    mean_activity = np.mean(tensor, axis=0)

    # Sort Rasters
    sorted_mean_activity = sort_raster(mean_activity, sorting_window_start, sorting_window_stop)

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, frame_rate)

    # Plot Raster
    n_neurons = np.shape(tensor)[2]
    magnitude = np.percentile(np.abs(mean_activity), q=99.5)

    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1,2,1)

    axis_1.imshow(np.transpose(sorted_mean_activity), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent = [x_values[0], x_values[-1], 0, n_neurons])

    axis_1.axvline(0, linestyle='dashed', c='k')

    axis_1.set_xlabel("Time (S)")

    axis_1.set_ylabel("Neurons")

    forceAspect(axis_1)

    figure_1.suptitle(condition_name)
    plt.show()



def view_mean_vis_context_lick_cd_projection(data_root, session, df_matrix, lick_cd, start_window=-12, stop_window=12):

    vis_1_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))
    vis_2_onsets = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))

    vis_1_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, vis_1_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=6)
    vis_2_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, vis_2_onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=6)

    vis_1_mean = np.mean(vis_1_tensor, axis=0)
    vis_2_mean = np.mean(vis_2_tensor, axis=0)

    vis_1_projection = np.dot(vis_1_mean, lick_cd)
    vis_2_projection = np.dot(vis_2_mean, lick_cd)

    plt.plot(vis_1_projection, c='b')
    plt.plot(vis_2_projection, c='r')
    plt.show()



def get_visual_context_lick_cd(data_root, session, output_root, start_window=-12, stop_window=12):

    # start_window=-12, stop_window=12

    save_directory = os.path.join(output_root, session, "Visual Context Lick CD")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)


    base_directory = os.path.join(data_root, session)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    period = float(1)/frame_rate

    # Load Onsets
    stim_onset_list, lick_onset_list = get_prelick_onsets(base_directory, frame_rate, min_rt=0.5, max_rt=2.5, lick_threshold=600)

    # Load DF Matrix
    df_matrix = load_df_matrix(base_directory)

    # Get Tensors
    visual_tensor = Get_Data_Tensor.get_data_tensor(df_matrix, lick_onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=6)

    trial_mean = np.mean(visual_tensor, axis=0)
    mean_lick_response = np.mean(trial_mean[6:12], axis=0)
    print("mean_lick_response", np.shape(mean_lick_response))
    lick_cd = mean_lick_response / np.linalg.norm(mean_lick_response)
    print("lick_cd", np.shape(lick_cd))

    np.save(os.path.join(save_directory, "vis_context_lick_cd.npy"), lick_cd)
    view_psth(visual_tensor, start_window, stop_window, period, save_directory, "Vis Context Pre Lick", 6, 12)

    view_mean_vis_context_lick_cd_projection(data_root, session, df_matrix, lick_cd, start_window=-12, stop_window=12)


# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

for session in control_session_list:
    get_visual_context_lick_cd(data_root, session, output_root)
