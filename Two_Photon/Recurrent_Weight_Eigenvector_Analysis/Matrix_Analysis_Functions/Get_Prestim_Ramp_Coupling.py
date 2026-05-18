import pickle
import os
import numpy as np
import matplotlib.pyplot as plt
from Matrix_Analysis_Functions import Matrix_Analysis_Functions

def open_tensor(file_location):

    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        print("session trial dict", session_trial_tensor_dict.keys())
        activity_tensor = session_trial_tensor_dict["tensor"]

        # Swap Axes To Fit Angus Convention
        activity_tensor = np.swapaxes(activity_tensor, 0, 1)

    return activity_tensor




def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=5):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= 0 and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

    tensor = np.array(tensor)
    return tensor



def load_onsets(data_root_directory, session, onsets_file, start_window, stop_window, number_of_timepoints):

    onset_file_path = os.path.join(data_root_directory, session, "Stimuli_Onsets", onsets_file)
    raw_onsets_list = np.load(onset_file_path, allow_pickle=True)

    checked_onset_list = []
    for trial_onset in raw_onsets_list:
        if trial_onset != None:
            trial_start = trial_onset + start_window
            trial_stop = trial_onset + stop_window
            if trial_start > 0 and trial_stop < number_of_timepoints:
                checked_onset_list.append(trial_onset)

    return checked_onset_list



def compute_projection_and_orthogonal_component(vector, projection_dimension):

    # Compute Direct Overlap
    direct_projection = np.dot(vector, projection_dimension)

    # Get Orthogonal Components
    parallel = direct_projection * projection_dimension
    orthogonal_component = vector - parallel
    orth_norm = np.linalg.norm(orthogonal_component)

    return direct_projection, orth_norm



def compute_prestim_coupling(stim_vector, recurrent_matrix, lick_cd, normalise=True):

    # Normalise Both
    lick_cd = np.squeeze(lick_cd)
    lick_norm = np.linalg.norm(lick_cd)
    lick_cd = lick_cd / lick_norm

    if normalise == True:
        stim_vector = np.squeeze(stim_vector)
        stim_norm = np.linalg.norm(stim_vector)
        stim_vector = stim_vector / stim_norm

    # Compute Cosine Similarity
    cosine_similarity = Matrix_Analysis_Functions.get_cosine_simmilarity(stim_vector, lick_cd)

    # Compute Direct Overlap
    direct_projection = np.dot(stim_vector, lick_cd)

    # Get Orthogonal Components
    parallel = direct_projection * lick_cd
    orthogonal_component = stim_vector - parallel

    if normalise == True:
        orth_norm = np.linalg.norm(orthogonal_component)
        orthogonal_component = orthogonal_component / orth_norm
        current_state = orthogonal_component

    else:
        current_state = stim_vector

    lick_projection_trajectory = []
    orthogonal_magnitude_trajectory = []

    for x in range(16):

        # Get Lick and Orthogonal projections
        current_projection, orth_norm = compute_projection_and_orthogonal_component(current_state, lick_cd)
        lick_projection_trajectory.append(current_projection)
        orthogonal_magnitude_trajectory.append(orth_norm)

        # Get new State
        current_state = recurrent_matrix @ current_state

    return direct_projection, lick_projection_trajectory, orthogonal_magnitude_trajectory, cosine_similarity




def prestim_coupling(session, data_root, df_matrix, recurrent_matrix, lick_cd, pre_learning=False):


    stop_window = 0

    # Load Onsets
    n_timepoints = np.shape(df_matrix)[0]
    if pre_learning == True:
        start_window = -9
        vis_1_onsets = load_onsets(data_root, session, "visual_1_all_onsets.npy", start_window, stop_window, n_timepoints)
        vis_2_onsets = load_onsets(data_root, session, "visual_2_all_onsets.npy", start_window, stop_window, n_timepoints)

    else:
        start_window = -16
        vis_1_onsets = load_onsets(data_root, session, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window, n_timepoints)
        vis_2_onsets = load_onsets(data_root, session, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window, n_timepoints)

    onset_list = vis_1_onsets + vis_2_onsets

    # Get Activity Tensor
    activity_tensor = get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=3)

    # Get Mean Activity
    mean_activity = np.mean(activity_tensor, axis=0) # Mean Over Trials
    mean_activity = np.mean(mean_activity, axis=0) # Mean Over Time

    # Compute Prestim Coupling
    direct_projection, lick_projection_trajectory, orthogonal_magnitude_trajectory, cosine_similarity = compute_prestim_coupling(mean_activity, recurrent_matrix, lick_cd, normalise=False)

    return direct_projection, lick_projection_trajectory, orthogonal_magnitude_trajectory, cosine_similarity



