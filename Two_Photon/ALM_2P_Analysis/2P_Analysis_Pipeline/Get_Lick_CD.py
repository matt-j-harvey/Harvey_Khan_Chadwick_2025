import numpy as np
import os
import matplotlib.pyplot as plt

import ALM_Analysis_Utils



def extract_correct_lick_onset_times(behaviour_matrix):

    lick_onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        ignore_irrel = trial[7]
        lick_onset = trial[22]

        if correct == 1:
            if trial_type == 1:
                lick_onset_list.append(lick_onset)

            elif trial_type == 3:
                if ignore_irrel == 1:
                    lick_onset_list.append(lick_onset)

    return lick_onset_list


def get_correct_lick_onsets(data_root, session, output_root):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Correct Lick Onsets
    lick_onset_times = extract_correct_lick_onset_times(behaviour_matrix)

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(data_root, session, "Behaviour", "Stack_Onsets.npy"))

    # Get Nearest Frames
    lick_onset_frames = ALM_Analysis_Utils.get_nearest_frames_to_onsets(lick_onset_times, stack_onsets)

    # Save These
    np.save(os.path.join(output_root, session, "Behaviour", "Correct_Lick_Onset_Frames.npy"), lick_onset_frames)





def get_lick_onsets(data_root, session, output_root):

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(data_root, session, "Behaviour", "Lick_Threshold.npy"))

    # Load Downsampled AI Matrix
    ai_data = np.load(os.path.join(output_root, session, "Behaviour", "Downsampled_AI_Matrix_Framewise.npy"))

    # Load Stimuli Dict
    stimuli_dict = ALM_Analysis_Utils.load_rig_1_channel_dict()

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))

    # Get Lick Trace
    downsampled_lick_trace = ai_data[stimuli_dict["Lick"]]

    # Get onsets
    preceeding_window = int(np.ceil(1.5 * frame_rate))
    lick_onsets = ALM_Analysis_Utils.get_onsets(downsampled_lick_trace, lick_threshold, preceeding_window=preceeding_window)

    plt.plot(downsampled_lick_trace)
    plt.scatter(lick_onsets, np.multiply(np.ones(len(lick_onsets)), lick_threshold), c='k')
    plt.show()

    # Save Onset Frames
    np.save(os.path.join(output_root, session, "Behaviour", "Lick_Onset_Frames.npy"), lick_onsets)




def get_lick_cd(data_root, session, output_root):

    # Load DF Data
    df_matrix = np.load(os.path.join(output_root, session, "df_over_f_matrix.npy"))
    print("df matrix", np.shape(df_matrix))

    # Get Lick Onsets
    #get_lick_onsets(data_root, session, output_root)
    get_correct_lick_onsets(data_root, session, output_root)

    # Load Lick Onsets
    #lick_onsets = np.load(os.path.join(output_root, session, "Behaviour", "Lick_Onset_Frames.npy"))
    lick_onsets = np.load(os.path.join(output_root, session, "Behaviour", "Correct_Lick_Onset_Frames.npy"))
    print("lick onsets", len(lick_onsets))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    period = float(1) / frame_rate

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

    # Get Lick Coding Dimension
    lick_cd_window_start = int(abs(start_window) - frame_rate)
    lick_cd_window_stop = abs(start_window)

    # Get Lick Preceeding Tensor
    lick_preceeding_tensor = lick_df_tensor[:, lick_cd_window_start:lick_cd_window_stop]

    # Get Mean Activity
    mean_lick_activity = np.mean(lick_preceeding_tensor, axis=1)  # Across Time
    mean_lick_activity = np.mean(mean_lick_activity, axis=0)  # Across Trials

    # Normalise
    norm = np.linalg.norm(mean_lick_activity)
    coding_dimension = np.divide(mean_lick_activity, norm)

    # Save
    save_directory = os.path.join(output_root, session, "Lick_Coding")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Lick_Coding_Dimension.npy"), coding_dimension)


