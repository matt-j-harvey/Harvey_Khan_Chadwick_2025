import numpy as np
import os
import pandas as pd
from tqdm import tqdm



def add_group(df_rows, mouse_id, session_list, data_root, genotype, selected_trial_type, start_window, stop_window):

    for mouse_sessions in tqdm(session_list):
        for session in mouse_sessions:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Load Corrected SVT
            svt = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Corrected_SVT.npy"))
            n_timepoints = np.shape(svt)[1]

            # Iterate Through Each Trial - and add it to the dataframe if it meets the criteria
            for trial in behaviour_matrix:
                trial_id = trial[0]
                trial_type = trial[1]
                trial_rt = trial[23]
                trial_onset_frame = trial[18]

                if trial_type == selected_trial_type:
                    if trial_onset_frame != None:
                        if not np.isnan(trial_rt):

                            trial_start = trial_onset_frame + start_window
                            trial_stop = trial_onset_frame + stop_window

                            if trial_start > 0:
                                if trial_stop < n_timepoints:
                                    # inside your loop
                                    df_rows.append({
                                        "mouse": mouse_id,
                                        "session": session,
                                        "group": genotype,
                                        "trial_id": trial_id,
                                        "reaction_time": trial_rt,
                                        "trial_start_frame": trial_start,
                                        "trial_stop_frame": trial_stop,
                                    })

        mouse_id += 1
    return df_rows, mouse_id


def create_combined_dataframe(wt_session_list, nx_session_list, wt_data_root, nx_data_root, selected_trial_type, start_window, stop_window):

    """
    Create a combined Pandas dataframe with the following fields:

    "mouse": mouse_ids,
    "session": session_name,
    "group": group_labels,
    "trial_id": trial_ids,
    "reaction_time": rt_values,
    "trial_start_frame": start_frames,
    "trial_stop_frame": stop_frames,

    """

    df_rows = []

    mouse_id = 0

    # Add WT Data
    df_rows, mouse_id = add_group(df_rows, mouse_id, wt_session_list, wt_data_root, 0, selected_trial_type, start_window, stop_window)
    df_rows, mouse_id = add_group(df_rows, mouse_id, nx_session_list, nx_data_root, 1, selected_trial_type, start_window, stop_window)

    df = pd.DataFrame(df_rows)

    return df
