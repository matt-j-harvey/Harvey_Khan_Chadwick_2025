import numpy as np
import os
import matplotlib.pyplot as plt
import pickle

import MVAR_Utils_2P
import RNN_Model
import torch
from matplotlib.pyplot import figure





def get_space_range(vis_context_vis_1_projection, vis_context_vis_2_projection, odr_context_vis_1_projection, odr_context_vis_2_projection):

    combined_trajectories = np.vstack([
        vis_context_vis_1_projection,
        vis_context_vis_2_projection,
        odr_context_vis_1_projection,
        odr_context_vis_2_projection])

    projection_min = np.min(combined_trajectories)
    projection_max = np.max(combined_trajectories)

    return projection_min, projection_max


def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        print("tenspr dct", session_trial_tensor_dict.keys())
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor


def make_rnn_prediction(rnn, input_data, device):
    with torch.no_grad():
        input_data = torch.tensor(input_data, dtype=torch.float, device=device)
        output_prediction = rnn(input_data).detach().numpy()
        output_prediction = output_prediction[0]
    return output_prediction



def plot_vector_field_real(axis, x_range, y_range, density=20):

    x = []
    y = []
    u = []
    v = []

    for x_value in np.linspace(start=x_range[0], stop=x_range[1], num=density):
        for y_value in np.linspace(start=y_range[0], stop=y_range[1], num=density):

            point_derivative = van_der_pol_system(x_value, y_value)

            x.append(x_value)
            y.append(y_value)
            u.append(point_derivative[0])
            v.append(point_derivative[1])

    axis.quiver(x, y, u, v, angles='xy', color='g', alpha=0.5)



def van_der_pol_system(x, y, u=0.3):
    dx = y
    dy = u * (1- x**2)*y - x
    return np.array([dx, dy])


def plot_vector_field_rnn(axis, rnn, device, x_range, y_range, lick_cd, context_cd, n_inputs, n_outputs, density=20):

    x = []
    y = []
    u = []
    v = []

    for x_value in np.linspace(start=x_range[0], stop=x_range[1], num=density):
        for y_value in np.linspace(start=y_range[0], stop=y_range[1], num=density):

            # Get Activity At This Point
            lick_projection = np.multiply(x_value, lick_cd)
            context_projection = np.multiply(y_value, context_cd)


            neural_activity = np.add(lick_projection, context_projection)


            # Predict Next State
            current_state = np.zeros((1, n_inputs))
            print("current state", np.shape(current_state))
            current_state[:, 0:n_outputs] = neural_activity
            new_state = make_rnn_prediction(rnn, current_state, device)

            # Get Derivative
            point_derivative = np.subtract(new_state, neural_activity)
            #point_derivative = new_state
            derivative_lick_projection = np.dot(point_derivative, lick_cd)
            derivative_context_projection = np.dot(point_derivative, context_cd)

            x.append(x_value)
            y.append(y_value)
            u.append(derivative_lick_projection)
            v.append(derivative_context_projection)

    axis.quiver(x, y, u, v, angles='xy', color='b', alpha=0.5)




def plot_trajectories(axis, output_directory, session, lick_cd, context_cd):

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

    # Get Projections
    vis_context_vis_1_lick_projection = np.dot(mean_vis_context_vis_1, lick_cd)
    vis_context_vis_2_lick_projection = np.dot(mean_vis_context_vis_2, lick_cd)
    odr_context_vis_1_lick_projection = np.dot(mean_odr_context_vis_1, lick_cd)
    odr_context_vis_2_lick_projection = np.dot(mean_odr_context_vis_2, lick_cd)

    vis_context_vis_1_context_projection = np.dot(mean_vis_context_vis_1, context_cd)
    vis_context_vis_2_context_projection = np.dot(mean_vis_context_vis_2, context_cd)
    odr_context_vis_1_context_projection = np.dot(mean_odr_context_vis_1, context_cd)
    odr_context_vis_2_context_projection = np.dot(mean_odr_context_vis_2, context_cd)

    # Plot Data
    axis.plot(vis_context_vis_1_lick_projection, vis_context_vis_1_context_projection, c='b')
    axis.scatter([vis_context_vis_1_lick_projection[0]], [vis_context_vis_1_context_projection[0]], c='k')

    axis.plot(vis_context_vis_2_lick_projection, vis_context_vis_2_context_projection, c='r')
    axis.scatter([vis_context_vis_2_lick_projection[0]], [vis_context_vis_2_context_projection[0]], c='k')

    axis.plot(odr_context_vis_1_lick_projection, odr_context_vis_1_context_projection, c='g')
    axis.scatter([odr_context_vis_1_lick_projection[0]], [odr_context_vis_1_context_projection[0]], c='k')

    axis.plot(odr_context_vis_2_lick_projection, odr_context_vis_2_context_projection, c='m')
    axis.scatter([odr_context_vis_2_lick_projection[0]], [odr_context_vis_2_context_projection[0]], c='k')

    # Get Space Ranges
    lick_min, lick_max = get_space_range(vis_context_vis_1_lick_projection,
                                         vis_context_vis_2_lick_projection,
                                         odr_context_vis_1_lick_projection,
                                         odr_context_vis_2_lick_projection)

    context_min, context_max = get_space_range(vis_context_vis_1_context_projection,
                                               vis_context_vis_2_context_projection,
                                               odr_context_vis_1_context_projection,
                                               odr_context_vis_2_context_projection)

    return [lick_min, lick_max], [context_min, context_max]


def evaluate_rnn(data_directory, session, output_directory, selected_iteration=5000):

    # Get Weight Directory
    weight_directory = os.path.join(output_directory, session, "RNN_Weights")

    # Load Data
    design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, output_directory, "combined")
    print("design_matrix", np.shape(design_matrix))
    print("delta_f_matrix", np.shape(delta_f_matrix))#

    # Load RNN
    n_inputs = np.shape(design_matrix)[1]
    n_outputs = np.shape(delta_f_matrix)[0]
    n_neurons = 250
    device = torch.device('cpu')
    rnn = RNN_Model.custom_rnn_model_2(n_inputs, n_neurons, n_outputs, device)
    rnn.load_state_dict(torch.load(os.path.join(weight_directory, str(selected_iteration).zfill(6) + '_model.pth')))

    # Load Coding Dimensions
    lick_cd_folder  = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"
    lick_cd = np.load(os.path.join(lick_cd_folder, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

    context_cd = np.load(os.path.join(data_directory, session, "Context_Decoding", "Decoding_Coefs.npy"))
    print("context cd", np.shape(context_cd))
    context_cd = np.mean(context_cd[0:18], axis=0)
    context_cd = np.squeeze(context_cd)
    print("context cd", np.shape(context_cd))

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)

    # Plot Trajectories
    lick_range, context_range = plot_trajectories(axis_1, output_directory, session, lick_cd, context_cd)

    # Plot Vector Field
    extent = 3
    plot_vector_field_rnn(axis_1, rnn, device, lick_range, context_range, lick_cd, context_cd, n_inputs, n_outputs, density=20)


    plt.show()

# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_RNN_Results"

session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


evaluate_rnn(data_root, session_list[1], output_root)


"""

# View Vector Field
plot_vector_field_real(axis_1,[-3,3], [-3,3], density=20)
plot_vector_field_rnn(axis_1, rnn, device, [-3,3], [-3,3], density=20)
plt.show()
"""



