import numpy as np
import os
import matplotlib.pyplot as plt

import RNN_Model
import torch
from matplotlib.pyplot import figure


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


def plot_vector_field_rnn(axis, rnn, device, x_range, y_range, density=20):

    x = []
    y = []
    u = []
    v = []

    for x_value in np.linspace(start=x_range[0], stop=x_range[1], num=density):
        for y_value in np.linspace(start=y_range[0], stop=y_range[1], num=density):

            current_state = np.array([x_value, y_value])
            current_state = np.reshape(current_state, (1,2))
            new_state  = make_rnn_prediction(rnn, current_state, device)
            print("new_state", new_state)
            point_derivative = np.subtract(new_state, current_state)[0]
            print("point derivative", point_derivative)
            x.append(x_value)
            y.append(y_value)
            u.append(point_derivative[0])
            v.append(point_derivative[1])

    axis.quiver(x, y, u, v, angles='xy', color='b', alpha=0.5)
    plt.show()




# Load RNN
weight_directory = r"C:\Users\matth\Documents\RNN_Play\RNN_Weights"
n_inputs = 2
n_outputs = 2
n_neurons = 250
device = torch.device('cpu')
rnn = RNN_Model.custom_rnn_model(n_inputs, n_neurons, n_outputs, device)
rnn.load_state_dict(torch.load(os.path.join(weight_directory, 'model.pth')))


figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)

# View Vector Field
plot_vector_field_real(axis_1,[-3,3], [-3,3], density=20)
plot_vector_field_rnn(axis_1, rnn, device, [-3,3], [-3,3], density=20)
plt.show()




