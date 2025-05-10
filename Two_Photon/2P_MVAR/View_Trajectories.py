import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

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


def get_trial_timing_data(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        start_window = session_trial_tensor_dict["start_window"]
        stop_window = session_trial_tensor_dict["stop_window"]
    return start_window, stop_window



def plot_vector_field(axis, lick_cd, context_cd, recurrent_weights, x_range, y_range, density=20):

    x = []
    y = []
    u = []
    v = []

    # np.matmul(mvar_parameters [n_neurons, n_regressors], design_matrix [n_regressors, n_timepoints])
    for x_value in np.linspace(start=x_range[0], stop=x_range[1], num=density):
        for y_value in np.linspace(start=y_range[0], stop=y_range[1], num=density):

            point_projection = np.add((lick_cd * x_value), (context_cd * y_value))
            point_derivative = np.matmul(recurrent_weights, point_projection)

            x.append(x_value)
            y.append(y_value)
            u.append(point_derivative[0])
            v.append(point_derivative[1])

    axis.quiver(x, y, u, v, angles='xy')



def plot_stimuli_vectors(initial_value, vector, lick_cd, context_cd, axis, start_window, colour, window_size=6):

    # Get Stimuli Vector Mean
    start_window = np.abs(start_window)
    mean_vector = np.mean(vector[:, start_window:start_window + window_size], axis=1)

    # Project Vector Onto Lick and Context Dimensions
    vector_x = np.dot(mean_vector, lick_cd)
    vector_y = np.dot(mean_vector, context_cd)

    # Project Start Value Onto Lick and Context Dimensions
    start_x = np.dot(initial_value, lick_cd)
    start_y = np.dot(initial_value, context_cd)


    x = [start_x]
    y = [start_y]
    u = [vector_x]
    v = [vector_y]

    axis.quiver(x, y, u, v, angles='xy', color=colour)




def get_space_range(vis_context_vis_1_projection, vis_context_vis_2_projection, odr_context_vis_1_projection, odr_context_vis_2_projection):

    combined_trajectories = np.vstack([
        vis_context_vis_1_projection,
        vis_context_vis_2_projection,
        odr_context_vis_1_projection,
        odr_context_vis_2_projection])

    projection_min = np.min(combined_trajectories)
    projection_max = np.max(combined_trajectories)

    return projection_min, projection_max


def plot_trajectories(data_directory, session, output_directory):

    # Load Activity Tensors
    vis_context_vis_1_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_1"))
    vis_context_vis_2_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_2"))
    odr_context_vis_1_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_stable_vis_1"))
    odr_context_vis_2_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_stable_vis_2"))

    # Get Means
    mean_vis_context_vis_1 = np.mean(vis_context_vis_1_tensor, axis=0)
    mean_vis_context_vis_2 = np.mean(vis_context_vis_2_tensor, axis=0)
    mean_odr_context_vis_1 = np.mean(odr_context_vis_1_tensor, axis=0)
    mean_odr_context_vis_2 = np.mean(odr_context_vis_2_tensor, axis=0)

    # Load Model Dict
    model_dict = np.load(os.path.join(output_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
    print("model dict", model_dict.keys())
    Nbehav = model_dict["Nbehav"]
    Nt = model_dict["Nt"]
    model_params = model_dict["MVAR_Parameters"]
    print("Nt", Nt)
    preceeding_window = int(Nt/2)
    print("preceeding_window", preceeding_window)

    # Load Recurrent Weights
    print("model params", np.shape(model_params))
    n_neurons = np.shape(model_params)[0]
    recurrent_weights = model_params[:, 0:n_neurons]
    np.fill_diagonal(recurrent_weights, 0)


    # Get Start and Stop WIndows
    #start_window, stop_window = get_start_stop_windows(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_1"))

    # Load Stimuli Weights
    stimulus_weight_list = []
    for stimulus_index in range(6):
        stimulus_weight_start = n_neurons + (stimulus_index * Nt)
        stimulus_weight_stop = stimulus_weight_start + Nt
        stimulus_weight_list.append(model_params[:, stimulus_weight_start:stimulus_weight_stop])


    # Get Context Dimension
    mean_vis_context = np.mean(np.array([mean_vis_context_vis_1, mean_vis_context_vis_2]), axis=0)
    vis_context_preceeding = mean_vis_context[0:preceeding_window]
    vis_context_vector = np.mean(vis_context_preceeding, axis=0)

    mean_odr_context = np.mean(np.array([mean_odr_context_vis_1, mean_odr_context_vis_2]), axis=0)
    odr_context_preceeding = mean_odr_context[0:preceeding_window]
    odr_context_vector = np.mean(odr_context_preceeding, axis=0)

    context_cd = np.subtract(vis_context_vector, odr_context_vector)

    print("mean_vis_context", np.shape(mean_vis_context))
    print("vis_context_preceeding", np.shape(vis_context_preceeding))


    # Get Lick CD
    behaviour_params = model_params[:, -Nbehav:]
    lick_params = behaviour_params[:, 1:] # Remove Running
    n_lick_timepoints = np.shape(lick_params)[1]
    preceeding_lick_timepoints = int((n_lick_timepoints-1)/2)
    preceeding_lick_activity = lick_params[:, 0:preceeding_lick_timepoints]
    lick_cd = np.mean(preceeding_lick_activity, axis=1)

    """
    lick_magnitude = np.percentile(np.abs(lick_params), q=95)
    plt.imshow(preceeding_lick_activity, cmap="bwr", vmin=-lick_magnitude, vmax=lick_magnitude)
    MVAR_Utils_2P.forceAspect(plt.gca())
    plt.show()
    """

    # Get Projections
    vis_context_vis_1_lick_projection = np.dot(mean_vis_context_vis_1, lick_cd)
    vis_context_vis_2_lick_projection = np.dot(mean_vis_context_vis_2, lick_cd)
    odr_context_vis_1_lick_projection = np.dot(mean_odr_context_vis_1, lick_cd)
    odr_context_vis_2_lick_projection = np.dot(mean_odr_context_vis_2, lick_cd)

    vis_context_vis_1_context_projection = np.dot(mean_vis_context_vis_1, context_cd)
    vis_context_vis_2_context_projection = np.dot(mean_vis_context_vis_2, context_cd)
    odr_context_vis_1_context_projection = np.dot(mean_odr_context_vis_1, context_cd)
    odr_context_vis_2_context_projection = np.dot(mean_odr_context_vis_2, context_cd)


    # Get Space Ranges
    lick_min, lick_max = get_space_range(vis_context_vis_1_lick_projection,
                                         vis_context_vis_2_lick_projection,
                                         odr_context_vis_1_lick_projection,
                                         odr_context_vis_2_lick_projection)

    context_min, context_max = get_space_range(vis_context_vis_1_context_projection,
                                               vis_context_vis_2_context_projection,
                                               odr_context_vis_1_context_projection,
                                               odr_context_vis_2_context_projection)


    # Plot Trajectories
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.set_title("Phase plot")
    axis_1.plot(vis_context_vis_1_lick_projection, vis_context_vis_1_context_projection, c='b')
    axis_1.scatter([vis_context_vis_1_lick_projection[0]], [vis_context_vis_1_context_projection[0]], c='k')

    axis_1.plot(vis_context_vis_2_lick_projection, vis_context_vis_2_context_projection, c='r')
    axis_1.scatter([vis_context_vis_2_lick_projection[0]], [vis_context_vis_2_context_projection[0]], c='k')

    axis_1.plot(odr_context_vis_1_lick_projection, odr_context_vis_1_context_projection, c='g')
    axis_1.scatter([odr_context_vis_1_lick_projection[0]], [odr_context_vis_1_context_projection[0]], c='k')

    axis_1.plot(odr_context_vis_2_lick_projection, odr_context_vis_2_context_projection, c='m')
    axis_1.scatter([odr_context_vis_2_lick_projection[0]], [odr_context_vis_2_context_projection[0]], c='k')

    # Plot Vector Field
    plot_vector_field(axis_1, lick_cd, context_cd, recurrent_weights, [lick_min, lick_max], [context_min, context_max])

    # Plot Stimuli Vectors
    start_window = int(Nt/2)
    plot_stimuli_vectors(mean_vis_context_vis_1[np.abs(start_window)], stimulus_weight_list[0], lick_cd, context_cd, axis_1, start_window, colour="Blue")
    plot_stimuli_vectors(mean_vis_context_vis_2[np.abs(start_window)], stimulus_weight_list[1], lick_cd, context_cd, axis_1, start_window, colour="Red")
    plot_stimuli_vectors(mean_odr_context_vis_1[np.abs(start_window)], stimulus_weight_list[2], lick_cd, context_cd, axis_1, start_window, colour="Green")
    plot_stimuli_vectors(mean_odr_context_vis_2[np.abs(start_window)], stimulus_weight_list[3], lick_cd, context_cd, axis_1, start_window, colour="Purple")

    plt.show()


    """
        plt.plot(vis_context_vis_1_lick_projection, c='b')
        plt.plot(vis_context_vis_2_lick_projection, c='r')
        plt.plot(odr_context_vis_1_lick_projection, c='g')
        plt.plot(odr_context_vis_2_lick_projection, c='m')
        plt.show()

        plt.plot(vis_context_vis_1_context_projection, c='b')
        plt.plot(vis_context_vis_2_context_projection, c='r')
        plt.plot(odr_context_vis_1_context_projection, c='g')
        plt.plot(odr_context_vis_2_context_projection, c='m')
        plt.show()
        """




# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_Odours"
#mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Combined_Model_Odours_No_Delta_1S"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

for session  in control_session_list:
    plot_trajectories(data_root, session, mvar_output_root)


