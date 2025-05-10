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



def get_integrated_interaction_old(stimulus_vector, recurrent_weights):

    print("stimulus_vector", np.shape(stimulus_vector))
    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_value = np.zeros(n_neurons)
    for x in range(9):
        result = np.matmul(recurrent_weights, stimulus_vector)
        current_value = np.add(current_value, result)
        print("result", np.shape(result))
        trial_vector.append(current_value)

    trial_vector = np.array(trial_vector)

    #view_psth(trial_vector)

    return trial_vector



def get_integrated_interaction(stimulus_vector, recurrent_weights):

    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_state = np.zeros(n_neurons)
    for x in range(9):
        trial_vector.append(current_state)

        current_state = np.add(current_state, stimulus_vector)
        new_state = np.matmul(recurrent_weights, current_state)
        current_state = new_state

    trial_vector = np.array(trial_vector)
    return trial_vector



def get_integrated_interaction_initial_state(stimulus_vector, recurrent_weights):

    print("stimulus_vector", np.shape(stimulus_vector))
    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_state = stimulus_vector
    for x in range(9):
        trial_vector.append(current_state)
        new_state = np.matmul(recurrent_weights, current_state)
        current_state = new_state
    trial_vector = np.array(trial_vector)

    #view_psth(trial_vector)

    return trial_vector




"""
def get_integrated_interaction(stimulus_vector, recurrent_weights):

    print("stimulus_vector", np.shape(stimulus_vector))
    trial_vector = []
    n_neurons = np.shape(stimulus_vector)[0]
    current_value = np.zeros(n_neurons)
    for x in range(9):
        trial_vector.append(current_value)
        result = np.matmul(recurrent_weights, current_value) + stimulus_vector
        print("result", np.shape(result))
        current_value = result


    trial_vector = np.array(trial_vector)

    #view_psth(trial_vector)

    return trial_vector
"""


def test_signficiance(vis_context_vis_1_projection_list, vis_context_vis_2_projection_list, odour_context_vis_1_projection_list, odour_context_vis_2_projection_list):

    vis_diff_list = []
    odour_diff_list = []

    n_mice = len(vis_context_vis_1_projection_list)

    for mouse in range(n_mice):
        vis_diff = np.subtract(vis_context_vis_1_projection_list[mouse], vis_context_vis_2_projection_list[mouse])
        odr_diff = np.subtract(odour_context_vis_1_projection_list[mouse], odour_context_vis_2_projection_list[mouse])

        vis_diff_list.append(vis_diff)
        odour_diff_list.append(odr_diff)

    t_stats, p_values = stats.ttest_rel(vis_diff_list, odour_diff_list)
    print("t_stats", t_stats)
    print("p_values", p_values)


def get_means_and_bounds(data_list):

    data_list = np.array(data_list)
    data_mean = np.mean(data_list, axis=0)

    data_sem = stats.sem(data_list, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound






def compare_stimulus_recurrent_interaction(data_root, session, output_directory):

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

    weight_magnitude = np.percentile(np.abs(recurrent_weights), q=95)
    plt.imshow(recurrent_weights, cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.show()


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


    # View Interaction Between Stimulus Vector and Recurrent Weights
    vis_context_vis_1_interaction_vector = get_integrated_interaction(visual_context_vis_1, recurrent_weights)
    vis_context_vis_2_interaction_vector = get_integrated_interaction(visual_context_vis_2, recurrent_weights)
    odr_context_vis_1_interaction_vector = get_integrated_interaction(odour_context_vis_1, recurrent_weights)
    odr_context_vis_2_interaction_vector = get_integrated_interaction(odour_context_vis_2, recurrent_weights)


    # Project Onto Lick CD
    vis_1_projection = np.dot(vis_context_vis_1_interaction_vector, lick_cd)
    vis_2_projection = np.dot(vis_context_vis_2_interaction_vector, lick_cd)

    odr_1_projection = np.dot(odr_context_vis_1_interaction_vector, lick_cd)
    odr_2_projection = np.dot(odr_context_vis_2_interaction_vector, lick_cd)


    plt.plot(vis_1_projection, c='b')
    plt.plot(vis_2_projection, c='r')
    plt.plot(odr_1_projection, c='g')
    plt.plot(odr_2_projection, c='m')
    plt.show()

    return vis_1_projection, vis_2_projection, odr_1_projection, odr_2_projection



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


vis_context_vis_1_projection_list = []
vis_context_vis_2_projection_list = []
odour_context_vis_1_projection_list = []
odour_context_vis_2_projection_list = []


for session in control_session_list:
    vis_1_projection, vis_2_projection, odr_1_projection, odr_2_projection = compare_stimulus_recurrent_interaction(data_root, session, mvar_output_root)

    vis_context_vis_1_projection_list.append(vis_1_projection)
    vis_context_vis_2_projection_list.append(vis_2_projection)
    odour_context_vis_1_projection_list.append(odr_1_projection)
    odour_context_vis_2_projection_list.append(odr_2_projection)

test_signficiance(vis_context_vis_1_projection_list, vis_context_vis_2_projection_list, odour_context_vis_1_projection_list, odour_context_vis_2_projection_list)


mean_vis_context_vis_1_projection_list = np.mean(vis_context_vis_1_projection_list, axis=0)
mean_vis_context_vis_2_projection_list = np.mean(vis_context_vis_2_projection_list, axis=0)
mean_odour_context_vis_1_projection_list = np.mean(odour_context_vis_1_projection_list, axis=0)
mean_odour_context_vis_2_projection_list = np.mean(odour_context_vis_2_projection_list, axis=0)

vis_context_vis_1_mean, vis_context_vis_1_lower_bound, vis_context_vis_1_upper_bound = get_means_and_bounds(vis_context_vis_1_projection_list)
vis_context_vis_2_mean, vis_context_vis_2_lower_bound, vis_context_vis_2_upper_bound = get_means_and_bounds(vis_context_vis_2_projection_list)
odour_context_vis_1_mean, odour_context_vis_1_lower_bound, odour_context_vis_1_upper_bound = get_means_and_bounds(odour_context_vis_1_projection_list)
odour_context_vis_2_mean, odour_context_vis_2_lower_bound, odour_context_vis_2_upper_bound = get_means_and_bounds(odour_context_vis_2_projection_list)

figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)

x_values = list(range(len(vis_context_vis_1_mean)))


plt.title("MEan ")
axis_1.plot(vis_context_vis_1_mean, c='b')
axis_1.fill_between(x_values, vis_context_vis_1_lower_bound, vis_context_vis_1_upper_bound, color='b', alpha=0.5)

axis_1.plot(vis_context_vis_2_mean, c='r')
axis_1.fill_between(x_values, vis_context_vis_2_lower_bound, vis_context_vis_2_upper_bound, color='r', alpha=0.5)

axis_1.plot(odour_context_vis_1_mean, c='g')
axis_1.fill_between(x_values, odour_context_vis_1_lower_bound, odour_context_vis_1_upper_bound, color='g', alpha=0.5)

axis_1.plot(odour_context_vis_2_mean, c='m')
axis_1.fill_between(x_values, odour_context_vis_2_lower_bound, odour_context_vis_2_upper_bound, color='m', alpha=0.5)



plt.show()
