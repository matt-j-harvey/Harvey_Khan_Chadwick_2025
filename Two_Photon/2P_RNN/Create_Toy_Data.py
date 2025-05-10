import numpy as np
import matplotlib.pyplot as plt
import os

def flatten_tensor(data_tensor):
    n_trials, n_timepoints, n_dim = np.shape(data_tensor)
    data_tensor = np.reshape(data_tensor, (n_trials * n_timepoints, n_dim))
    return data_tensor

def plot_vector_field(axis, x_range, y_range, density=20):

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

    axis.quiver(x, y, u, v, angles='xy')


def van_der_pol_system(x, y, u=0.3):
    dx = y
    dy = u * (1- x**2)*y - x
    return np.array([dx, dy])


def generate_trajectory(initial_state, dt=0.1, n_timepoints=20):
    trajectory = []
    current_state = initial_state
    for timepoint_index in range(n_timepoints):
        trajectory.append(current_state)
        derivative = van_der_pol_system(current_state[0], current_state[1])
        derivative = np.multiply(derivative, dt)
        new_state = np.add(current_state, derivative)
        current_state = new_state

    trajectory = np.array(trajectory)
    return trajectory


def generate_dataset(initial_point_magnitude=3, density=20):

    data_tensor = []
    for x in np.linspace(-initial_point_magnitude, initial_point_magnitude, num=density):
        for y in np.linspace(-initial_point_magnitude, initial_point_magnitude, num=density):
            trajectory = generate_trajectory(np.array([x, y]))
            data_tensor.append(trajectory)

            plt.plot(trajectory[:, 0], trajectory[:, 1], alpha=0.5)
    plt.show()
    data_tensor = np.array(data_tensor)
    return data_tensor




figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)
plot_vector_field(axis_1, [-3,3], [-3,3])
plt.show()

save_directory = r"C:\Users\matth\Documents\RNN_Play\Test_Data"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

data_tensor = generate_dataset()
print("data tensor shape", np.shape(data_tensor))
input_data = data_tensor[:, 0:-1]
output_data = data_tensor[:, 1:]


# Flatten Tensors
input_data = flatten_tensor(input_data)
output_data = flatten_tensor(output_data)
print("Input Data Shape", np.shape(input_data))

np.save(os.path.join(save_directory, "Input_Data.npy"), input_data)
np.save(os.path.join(save_directory, "Output_Data.npy"), output_data)





