import os
import numpy as np
import matplotlib.pyplot as plt
from Shared_Utils.Get_DF import load_df_matrix



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def get_data_tensor(df_matrix, onset_list, start_window, stop_window):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start > 0 and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            trial_baseline = trial_data[0:3]
            trial_baseline = np.mean(trial_baseline, axis=0)
            trial_data = np.subtract(trial_data, trial_baseline)
            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor


def load_onsets(base_directory):

    if os.path.isdir(os.path.join(base_directory, "Stimuli_Onsets", "visual_1_correct_onsets.npy")):
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_1_correct_onsets.npy"))
    elif os.path.isdir(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy")):
        vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))

    return vis_1_onsets


def view_df_matrix(df_matrix):
    magnitude = np.percentile(df_matrix, q=95)
    plt.imshow(np.transpose(df_matrix), vmin=0, vmax=magnitude)
    forceAspect(plt.gca())
    plt.title("df matrix")
    plt.show()


def sanity_check_session(session, data_root):
    base_directory = os.path.join(data_root, session)

    # Load DF Matrix
    df_matrix = load_df_matrix(base_directory)
    print("df_matrix", np.shape(df_matrix))

    view_df_matrix(df_matrix)

    frame_rate= np.load(os.path.join(base_directory, "Frame_Rate.npy"))
    print("frame rate", frame_rate)

    # Load Onsets
    vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))
    #load_onsets(base_directory)

    # Get Data Tensor
    start_window = -15
    stop_window = 9
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 1000/6.37)
    vis_1_tensor = get_data_tensor(df_matrix, vis_1_onsets, start_window=start_window, stop_window=stop_window)

    mean_response = np.mean(vis_1_tensor, axis=0)

    mean_over_time = np.mean(mean_response[np.abs(start_window):], axis=0)
    sorting_indicies = np.argsort(mean_over_time)

    #view_individual_trials(vis_1_tensor, sorting_indicies)

    mean_response = mean_response[:, sorting_indicies]
    magnitude = np.percentile(np.abs(mean_response), q=95)

    plt.title(session)
    plt.imshow(np.transpose(mean_response), cmap="bwr", vmin=-magnitude, vmax=magnitude, extent=[x_values[0], x_values[-1],0,np.shape(mean_response)[1]])
    forceAspect(plt.gca())
    plt.title(str(frame_rate))
    plt.show()

    
def view_individual_trials(tensor, sorting_indicies):
    
    tensor = tensor[:, :, sorting_indicies]

    magnitude = np.percentile(np.abs(tensor), q=95)

    count = 0
    for trial in tensor:
        plt.imshow(np.transpose(trial), cmap="bwr", vmin=-magnitude, vmax=magnitude)
        forceAspect(plt.gca())
        plt.title(str(count))
        plt.show()
        count += 1
    



    #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls\65.2b\2024_07_25_Switching",
    #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\70.1A\2024_08_07_Discrimination",
session_list = [
    r"70.1A\2024_09_03_Switching",
    r"70.1A\2024_09_09_Switching",
    r"70.1A\2024_09_19_Switching",

    r"64.1B\2024_09_09_Switching",
     r"64.1B\2024_09_06_Switching",
     r"64.1B\2024_09_19_Switching",

    r"70.1B\2024_09_12_Switching",
     r"70.1B\2024_09_06_Switching",
     r"70.1B\2024_09_20_Switching",

    r"72.1E\2024_08_23_Switching",
     r"72.1E\2024_08_27_Switching",
     r"72.1E\2024_08_29_Switching",

]




data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"

for session in session_list:
    sanity_check_session(session, data_root)