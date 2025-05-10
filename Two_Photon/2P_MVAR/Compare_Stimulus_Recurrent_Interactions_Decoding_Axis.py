import numpy as np
import os
import matplotlib.pyplot as plt
import pickle
from scipy import stats


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


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


def view_psth(mean_activity):

    # Plot Raster
    magnitude = np.percentile(np.abs(mean_activity), q=99.5)

    figure_1 = plt.figure(figsize=(10,5))
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.imshow(np.transpose(mean_activity), vmin=-magnitude, vmax=magnitude, cmap='bwr')
    axis_1.set_xlabel("Time (S)")
    axis_1.set_ylabel("Neurons")
    forceAspect(axis_1)

    plt.show()



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



def get_stim_cd(output_directory, session, start_window, response_window_size=6):

    # Load Activity Tensors
    vis_context_vis_1_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_1"))
    vis_context_vis_2_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_2"))
    odr_context_vis_1_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_stable_vis_1"))
    odr_context_vis_2_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_stable_vis_2"))
    print("vis_context_vis_1_tensor", np.shape(vis_context_vis_1_tensor))

    # Get Mean Trial Activity
    mean_vis_context_vis_1 = np.mean(vis_context_vis_1_tensor, axis=0)
    mean_vis_context_vis_2 = np.mean(vis_context_vis_2_tensor, axis=0)
    mean_odr_context_vis_1 = np.mean(odr_context_vis_1_tensor, axis=0)
    mean_odr_context_vis_2 = np.mean(odr_context_vis_2_tensor, axis=0)
    print("mean_vis_context_vis_1", np.shape(mean_vis_context_vis_1))

    # Get Mean Stimuli Response
    mean_vis_context_vis_1_response = np.mean(mean_vis_context_vis_1[start_window:start_window + response_window_size], axis=0)
    mean_vis_context_vis_2_response = np.mean(mean_vis_context_vis_2[start_window:start_window + response_window_size], axis=0)
    mean_odr_context_vis_1_response = np.mean(mean_odr_context_vis_1[start_window:start_window + response_window_size], axis=0)
    mean_odr_context_vis_2_response = np.mean(mean_odr_context_vis_2[start_window:start_window + response_window_size], axis=0)
    print("mean_vis_context_vis_1_response", np.shape(mean_vis_context_vis_1_response))


    # Get Stim CDs
    visual_stim_cd = np.subtract(mean_vis_context_vis_1_response, mean_vis_context_vis_2_response)
    visual_stim_cd = visual_stim_cd / np.linalg.norm(visual_stim_cd)


    # Visualise As Sanity Check
    vis_context_vis_1_projection = np.dot(mean_vis_context_vis_1, visual_stim_cd)
    vis_context_vis_2_projection = np.dot(mean_vis_context_vis_2, visual_stim_cd)
    odr_context_vis_1_projection = np.dot(mean_odr_context_vis_1, visual_stim_cd)
    odr_context_vis_2_projection = np.dot(mean_odr_context_vis_2, visual_stim_cd)

    plt.plot(vis_context_vis_1_projection, c='b')
    plt.plot(vis_context_vis_2_projection, c='r')
    plt.plot(odr_context_vis_1_projection, c='g')
    plt.plot(odr_context_vis_2_projection, c='m')
    plt.show()

    return visual_stim_cd




def get_initial_states(base_directory, df_matrix, stimulus_type):

    onsets_list = np.load(os.path.join(base_directory, "Stimuli_Onsets", stimulus_type + "_onsets.npy"))
    initial_state_list = []
    for onset in onsets_list:
        initial_state = df_matrix[onset-3:onset]
        initial_state = np.mean(initial_state, axis=0)
        initial_state_list.append(initial_state)

    return initial_state_list


def load_all_initial_states(base_directory, df_matrix):
    vis_context_vis_1_initial_states = get_initial_states(base_directory, df_matrix, "visual_context_stable_vis_1")
    vis_context_vis_2_initial_states = get_initial_states(base_directory, df_matrix, "visual_context_stable_vis_2")
    odr_context_vis_1_initial_states = get_initial_states(base_directory, df_matrix, "odour_context_stable_vis_1")
    odr_context_vis_2_initial_states = get_initial_states(base_directory, df_matrix, "odour_context_stable_vis_2")

    visual_context_states = vis_context_vis_1_initial_states + vis_context_vis_2_initial_states
    odour_context_states = odr_context_vis_1_initial_states + odr_context_vis_2_initial_states

    return vis_context_vis_1_initial_states, odr_context_vis_1_initial_states




def get_integrated_interaction(stimulus_vector, recurrent_weights, initial_state):

    print("stimulus_vector", np.shape(stimulus_vector))
    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_value = initial_state
    for x in range(9):
        trial_vector.append(current_value)
        result = np.matmul(recurrent_weights, current_value) + stimulus_vector
        print("result", np.shape(result))
        current_value = result


    trial_vector = np.array(trial_vector)

    #view_psth(trial_vector)

    return trial_vector


def load_stimuli_vectors(model_params, n_neurons, Nt, preceeding_window):

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

    return [visual_context_vis_1, visual_context_vis_2, odour_context_vis_1, odour_context_vis_2]



def compare_trajectories(stimuli_vector_list, recurrent_weights, initial_state, lick_cd):

    # View Interaction Between Stimulus Vector and Recurrent Weights
    vis_context_vis_1_interaction_vector = get_integrated_interaction(stimuli_vector_list[0], recurrent_weights, initial_state)
    vis_context_vis_2_interaction_vector = get_integrated_interaction(stimuli_vector_list[1], recurrent_weights, initial_state)
    odr_context_vis_1_interaction_vector = get_integrated_interaction(stimuli_vector_list[2], recurrent_weights, initial_state)
    odr_context_vis_2_interaction_vector = get_integrated_interaction(stimuli_vector_list[3], recurrent_weights, initial_state)

    # Project Onto Lick CD
    vis_1_projection = np.dot(vis_context_vis_1_interaction_vector, lick_cd)
    vis_2_projection = np.dot(vis_context_vis_2_interaction_vector, lick_cd)
    odr_1_projection = np.dot(odr_context_vis_1_interaction_vector, lick_cd)
    odr_2_projection = np.dot(odr_context_vis_2_interaction_vector, lick_cd)

    """
    plt.plot(vis_1_projection, c='b')
    plt.plot(vis_2_projection, c='r')
    plt.plot(odr_1_projection, c='g')
    plt.plot(odr_2_projection, c='m')
    plt.show()
    """

    return vis_1_projection


def compare_stimulus_recurrent_interaction(data_root, session, output_directory):

    # load DF Matrix
    df_matrix = load_df_matrix(os.path.join(data_root, session))

    # Load Decoding Axis
    decoding_coefs = np.load(os.path.join(data_root, session, "Context_Decoding", "Decoding_Coefs.npy"))
    print("decoding coefs", np.shape(decoding_coefs))
    decoding_coefs = np.mean(decoding_coefs[0:18], axis=0)
    decoding_coefs = np.squeeze(decoding_coefs)
    print("decoding coefs", np.shape(decoding_coefs))

    # Get Lick CD
    lick_cd_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"
    lick_cd = np.load(os.path.join(lick_cd_directory, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

    # Load Model Dict
    model_dict = np.load(os.path.join(output_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
    model_params = model_dict["MVAR_Parameters"]
    Nt = model_dict["Nt"]
    preceeding_window = int(Nt/2)

    # Load Recurrent Weights
    n_neurons = np.shape(model_params)[0]
    recurrent_weights = model_params[:, 0:n_neurons]
    np.fill_diagonal(recurrent_weights, 0)

    # Load Stimuli Vectors
    stimuli_vector_list = load_stimuli_vectors(model_params, n_neurons, Nt, preceeding_window)

    # Compare Trajectories
    initial_state = np.zeros(n_neurons)
    zero_vis_1_proj = compare_trajectories(stimuli_vector_list, recurrent_weights, initial_state, lick_cd)

    initial_state = decoding_coefs
    visual_vis_1_proj = compare_trajectories(stimuli_vector_list, recurrent_weights, initial_state, lick_cd)

    initial_state = -1 * decoding_coefs
    odour_vis_1_proj = compare_trajectories(stimuli_vector_list, recurrent_weights, initial_state, lick_cd)

    # Zero All
    print("zero_vis_1_proj", zero_vis_1_proj)
    zero_vis_1_proj = np.subtract(zero_vis_1_proj, zero_vis_1_proj[0])
    visual_vis_1_proj = np.subtract(visual_vis_1_proj, visual_vis_1_proj[0])
    odour_vis_1_proj = np.subtract(odour_vis_1_proj, odour_vis_1_proj[0])

    print("zero_vis_1_proj", zero_vis_1_proj)
    plt.plot(zero_vis_1_proj, c='k')
    plt.plot(visual_vis_1_proj, c='b')
    plt.plot(odour_vis_1_proj, c='g')

    plt.xlabel("Timestep")
    plt.ylabel("Visual lick CD")
    plt.show()

    """

    """
    return zero_vis_1_proj, visual_vis_1_proj, odour_vis_1_proj


# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_Odours"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_No_Preceeding_Lick_Regressor"




control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

zero_vis_1_proj_list = []
visual_vis_1_proj_list = []
odour_vis_1_proj_list = []

for session in control_session_list:
    zero_vis_1_proj, visual_vis_1_proj, odour_vis_1_proj = compare_stimulus_recurrent_interaction(data_root, session, mvar_output_root)
    zero_vis_1_proj_list.append(zero_vis_1_proj)
    visual_vis_1_proj_list.append(visual_vis_1_proj)
    odour_vis_1_proj_list.append(odour_vis_1_proj)


mean_zero_vis_1_proj = np.mean(zero_vis_1_proj_list, axis=0)
mean_visual_vis_1_proj = np.mean(visual_vis_1_proj_list, axis=0)
mean_odour_vis_1_proj = np.mean(odour_vis_1_proj_list, axis=0)

t_stats, p_values = stats.ttest_rel(visual_vis_1_proj_list, odour_vis_1_proj_list, axis=0)
print("p values", p_values)

plt.title("Grouup mean")
plt.plot(mean_zero_vis_1_proj, c='k')
plt.plot(mean_visual_vis_1_proj, c='b')
plt.plot(mean_odour_vis_1_proj, c='g')
plt.show()
