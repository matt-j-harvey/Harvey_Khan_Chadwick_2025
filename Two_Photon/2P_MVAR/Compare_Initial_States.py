import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy import stats

import MVAR_Utils_2P



def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        print("tenspr dct", session_trial_tensor_dict.keys())
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor


def get_start_stop_windows(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        start_window = session_trial_tensor_dict["start_window"]
        stop_window = session_trial_tensor_dict["stop_window"]

    return start_window, stop_window



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


"""
def run_trajectory(start_point_list, recurrent_weights, stimulus_weights, stimulus_cd, n_timepoints=9):

    print("start_point_list", np.shape(start_point_list))
    print("stimulus_weights", np.shape(stimulus_weights))
    # Start at Time 0 - go for 6 iterations

    plt.imshow(stimulus_weights)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    simulated_tensor = []
    for initial_condition in start_point_list:

        trajectory = []
        current_state = initial_condition
        trajectory.append(current_state)
        for timepoint in range(n_timepoints):
            print("Recurrent weights", np.shape(recurrent_weights))
            print("current state", np.shape(current_state))
            recurrent_contribution = np.matmul(recurrent_weights, current_state)

            stimulus_contribution = stimulus_weights[:, timepoint]
            print("recurrent_contribution", np.shape(recurrent_contribution))
            print("stimulus_contribution", np.shape(stimulus_contribution))

            new_state = recurrent_contribution + stimulus_contribution
            trajectory.append(new_state)
            current_state = new_state

        trajectory_projection = np.dot(trajectory, stimulus_cd)
        plt.plot(trajectory_projection)
        simulated_tensor.append(trajectory)

    plt.show()
    simulated_tensor = np.array(simulated_tensor)
    print("simulated_tensor", np.shape(simulated_tensor))

    simulated_mean = np.mean(simulated_tensor, axis=0)
    print("simulated_mean", np.shape(simulated_mean))

    stimulus_projection = np.dot(simulated_mean, stimulus_cd)

    return stimulus_projection
"""


def run_trajectory(initial_state_list, recurrent_weights, stimulus_vector, lick_cd):

    trajectory_tensor = []



    print("Initial states", len(initial_state_list))
    for initial_state in initial_state_list:

        trial_vector = []
        current_value = initial_state
        print("Initial state", np.shape(initial_state))
        print("recurrent_weights", np.shape(recurrent_weights))
        for x in range(9):
            recurrent_contribution = np.matmul(recurrent_weights, current_value)
            print("recurrent_contribution", np.shape(recurrent_contribution))

            new_state = np.add(recurrent_contribution, stimulus_vector)
            print("new_state", np.shape(new_state))

            print("new state", np.shape(new_state))
            current_value = new_state
            trial_vector.append(current_value)

        trial_vector = np.array(trial_vector)
        trajectory_tensor.append(trial_vector)

    trajectory_tensor = np.array(trajectory_tensor)

    mean_trajectory = np.mean(trajectory_tensor, axis=0)

    mean_projection = np.dot(mean_trajectory, lick_cd)
    return mean_projection




def get_initial_states(base_directory, df_matrix, stimulus_type):

    onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", stimulus_type + "_onsets.npy"))
    initial_state_list = []
    for onset in onsets_list:
        initial_state = df_matrix[onset-3:onset]
        initial_state = np.mean(initial_state, axis=0)
        initial_state_list.append(initial_state)
    """
    visual_context_stable_vis_1
    visual_context_vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "visual_context_stable_vis_2_onsets.npy"))
    odour_context_vis_1_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_stable_vis_1_onsets.npy"))
    odour_context_vis_2_onsets = np.load(os.path.join(base_directory, "Stimuli_Onsets", "odour_context_stable_vis_2_onsets.npy"))
    """
    return initial_state_list


def load_all_initial_states(base_directory, df_matrix):
    vis_context_vis_1_initial_states = get_initial_states(base_directory, df_matrix, "visual_context_stable_vis_1")
    vis_context_vis_2_initial_states = get_initial_states(base_directory, df_matrix, "visual_context_stable_vis_2")
    odr_context_vis_1_initial_states = get_initial_states(base_directory, df_matrix, "odour_context_stable_vis_1")
    odr_context_vis_2_initial_states = get_initial_states(base_directory, df_matrix, "odour_context_stable_vis_2")

    visual_context_states = vis_context_vis_1_initial_states + vis_context_vis_2_initial_states
    odour_context_states = odr_context_vis_1_initial_states + odr_context_vis_2_initial_states

    return vis_context_vis_1_initial_states, odr_context_vis_1_initial_states


def compare_initial_states(data_directory, session, output_directory):

    # Get Lick CD
    lick_cd_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"
    lick_cd = np.load(os.path.join(lick_cd_directory, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

    # load DF Matrix
    df_matrix = load_df_matrix(os.path.join(data_directory, session))

    # Get Initial States
    vis_context_vis_1_initial_states, odr_context_vis_1_initial_states = load_all_initial_states(os.path.join(data_directory, session), df_matrix)

    # Load Model Dict
    model_dict = np.load(os.path.join(output_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
    Nbehav = model_dict["Nbehav"]
    Nt = model_dict["Nt"]
    model_params = model_dict["MVAR_Parameters"]
    preceeding_window = int(Nt / 2)

    # Load Recurrent Weights
    n_neurons = np.shape(model_params)[0]
    recurrent_weights = model_params[:, 0:n_neurons]
    np.fill_diagonal(recurrent_weights, 0)
    print("initial recurrent_weights", np.shape(recurrent_weights))

    # Load Stimuli Weights
    stimulus_weight_list = []
    for stimulus_index in range(6):
        stimulus_weight_start = n_neurons + (stimulus_index * Nt)
        stimulus_weight_stop = stimulus_weight_start + Nt
        stimulus_weight_list.append(model_params[:, stimulus_weight_start:stimulus_weight_stop])

    # Extract Vis 1 and 2 Stimuli
    visual_context_vis_1 = stimulus_weight_list[0]
    visual_context_vis_2 = stimulus_weight_list[1]
    odour_context_vis_1 = stimulus_weight_list[2]
    odour_context_vis_2 = stimulus_weight_list[3]
    print("visual_context_vis_1", np.shape(visual_context_vis_1))

    # Get Mean Stimuli Response
    response_window_size = 6
    visual_context_vis_1 = np.mean(visual_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    visual_context_vis_2 = np.mean(visual_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_1 = np.mean(odour_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_2 = np.mean(odour_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    print("visual_context_vis_1", np.shape(visual_context_vis_1))


    vis_context_vis_1_projection = run_trajectory(vis_context_vis_1_initial_states, recurrent_weights, visual_context_vis_1, lick_cd)
    odr_context_vis_1_projection = run_trajectory(odr_context_vis_1_initial_states, recurrent_weights, visual_context_vis_1, lick_cd)

    vis_context_vis_2_projection = run_trajectory(vis_context_vis_1_initial_states, recurrent_weights, visual_context_vis_2, lick_cd)
    odr_context_vis_2_projection = run_trajectory(odr_context_vis_1_initial_states, recurrent_weights, visual_context_vis_2, lick_cd)


    plt.title("each stimuli")
    plt.plot(vis_context_vis_1_projection, c='b')
    plt.plot(odr_context_vis_1_projection, c='g')
    plt.plot(vis_context_vis_2_projection, c='r')
    plt.plot(odr_context_vis_2_projection, c='m')
    plt.show()


    visual_state_diff = np.subtract(vis_context_vis_1_projection, vis_context_vis_2_projection)
    odour_state_diff = np.subtract(odr_context_vis_1_projection, odr_context_vis_2_projection)
    plt.title("diff")
    plt.plot(visual_state_diff, c='b')
    plt.plot(odour_state_diff, c='g')
    plt.show()




# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_No_Preceeding_Lick_Regressor"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

visual_list = []
odour_list = []

for session  in control_session_list:
    compare_initial_states(data_root, session, mvar_output_root)


visual_mean = np.mean(np.array(visual_list), axis=0)
odour_mean = np.mean(np.array(odour_list), axis=0)

plt.plot(visual_mean, c='b')
plt.plot(odour_mean, c='g')
plt.show()


## Just Have Stimuli