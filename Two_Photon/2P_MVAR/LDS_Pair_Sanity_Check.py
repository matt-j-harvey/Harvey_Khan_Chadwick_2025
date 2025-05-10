import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean

def get_distance(trajectory_1, trajectory_2):
    distance_list = []

    n_timepoints = np.shape(trajectory_1)[0]
    for timepoint_index in range(n_timepoints):
        distance = euclidean(trajectory_1[timepoint_index], trajectory_2[timepoint_index])
        distance_list.append(distance)

    return distance_list


def run_trajectory(initial_state, recurrent_weights, stimulus_vector, dt=0.5):

    trial_vector = []
    current_value = initial_state
    trial_vector.append(initial_state)

    for x in range(9):

        derivative = np.matmul(recurrent_weights, current_value) + stimulus_vector

        new_state = current_value + (derivative * dt) #+ stimulus_vector

        current_value = new_state
        trial_vector.append(current_value)

    trial_vector = np.array(trial_vector)
    return trial_vector


def plot_stimuli_vector(axis, initial_state, vector, colour):

    x = [initial_state[0]]
    y = [initial_state[1]]
    u = [vector[0]]
    v = [vector[1]]

    axis.quiver(x, y, u, v, angles='xy', color=colour, alpha=0.5)


def plot_vector_field(axis, transformation_matrix, x_range, y_range, density=20):

    x = []
    y = []
    u = []
    v = []

    for x_value in np.linspace(start=x_range[0], stop=x_range[1], num=density):
        for y_value in np.linspace(start=y_range[0], stop=y_range[1], num=density):

            point_derivative = np.matmul(transformation_matrix, np.array([x_value,y_value]))

            x.append(x_value)
            y.append(y_value)
            u.append(point_derivative[0])
            v.append(point_derivative[1])

    axis.quiver(x, y, u, v, angles='xy')




transformation_matrix = np.array([[0,0.2],
                                  [0.2,0]])


figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)

vis_context_stim_1_vector = np.array([1, 1])
vis_context_stim_2_vector = np.array([-1,1])
odr_context_stim_1_vector = np.array([0.8, 0.1])
odr_context_stim_2_vector = np.array([-2,-0.5])

#visual_initial_state = [8, -8]
visual_initial_state = [0, -8]
odour_initial_state = [0, 0]

# Get Trajectories
vis_context_vis_stim_1_trajectory = run_trajectory(visual_initial_state, transformation_matrix, vis_context_stim_1_vector)
vis_context_vis_stim_2_trajectory = run_trajectory(visual_initial_state, transformation_matrix, vis_context_stim_2_vector)
vis_context_odr_stim_1_trajectory = run_trajectory(visual_initial_state, transformation_matrix, odr_context_stim_1_vector)
vis_context_odr_stim_2_trajectory = run_trajectory(visual_initial_state, transformation_matrix, odr_context_stim_2_vector)

odr_context_vis_stim_1_trajectory = run_trajectory(odour_initial_state, transformation_matrix, vis_context_stim_1_vector)
odr_context_vis_stim_2_trajectory = run_trajectory(odour_initial_state, transformation_matrix, vis_context_stim_2_vector)
odr_context_odr_stim_1_trajectory = run_trajectory(odour_initial_state, transformation_matrix, odr_context_stim_1_vector)
odr_context_odr_stim_2_trajectory = run_trajectory(odour_initial_state, transformation_matrix, odr_context_stim_2_vector)


# Plot Vector Field
combined_trajectories = np.vstack([
    
    vis_context_vis_stim_1_trajectory,
    vis_context_vis_stim_2_trajectory,
    vis_context_odr_stim_1_trajectory,
    vis_context_odr_stim_2_trajectory,
        
    odr_context_vis_stim_1_trajectory,
    odr_context_vis_stim_2_trajectory,
    odr_context_odr_stim_1_trajectory,
    odr_context_odr_stim_2_trajectory,

])

x_min = np.min(combined_trajectories[:, 0])
x_max = np.max(combined_trajectories[:, 0])
y_min = np.min(combined_trajectories[:, 1])
y_max = np.max(combined_trajectories[:, 1])

print("combined_trajectories", np.shape(combined_trajectories))
plot_vector_field(axis_1, transformation_matrix, [x_min, x_max], [y_min, y_max], density=20)


# Scatter Initial Points
axis_1.scatter([visual_initial_state[0]], [visual_initial_state[1]], c='b')
axis_1.scatter([odour_initial_state[0]], [odour_initial_state[1]], c='m')

# Plot Stimuli Vectors
plot_stimuli_vector(axis_1, visual_initial_state, vis_context_stim_1_vector, "b")
plot_stimuli_vector(axis_1, visual_initial_state, vis_context_stim_2_vector, "r")
plot_stimuli_vector(axis_1, visual_initial_state, odr_context_stim_1_vector, "g")
plot_stimuli_vector(axis_1, visual_initial_state, odr_context_stim_2_vector, "m")

plot_stimuli_vector(axis_1, odour_initial_state, vis_context_stim_1_vector, "b")
plot_stimuli_vector(axis_1, odour_initial_state, vis_context_stim_2_vector, "r")
plot_stimuli_vector(axis_1, odour_initial_state, odr_context_stim_1_vector, "g")
plot_stimuli_vector(axis_1, odour_initial_state, odr_context_stim_2_vector, "m")


# Plot Trajectories

vis_context_vis_stim_1_trajectory,
vis_context_vis_stim_2_trajectory,
vis_context_odr_stim_1_trajectory,
vis_context_odr_stim_2_trajectory,

odr_context_vis_stim_1_trajectory,
odr_context_vis_stim_2_trajectory,
odr_context_odr_stim_1_trajectory,
odr_context_odr_stim_2_trajectory,


plt.plot(vis_context_vis_stim_1_trajectory[:, 0], vis_context_vis_stim_1_trajectory[:, 1], c='b')
plt.plot(vis_context_vis_stim_2_trajectory[:, 0], vis_context_vis_stim_2_trajectory[:, 1], c='r')
plt.plot(vis_context_odr_stim_1_trajectory[:, 0], vis_context_odr_stim_1_trajectory[:, 1], c='g')
plt.plot(vis_context_odr_stim_2_trajectory[:, 0], vis_context_odr_stim_2_trajectory[:, 1], c='m')
plt.scatter(vis_context_vis_stim_1_trajectory[:, 0], vis_context_vis_stim_1_trajectory[:, 1], c='b')
plt.scatter(vis_context_vis_stim_2_trajectory[:, 0], vis_context_vis_stim_2_trajectory[:, 1], c='r')
plt.scatter(vis_context_odr_stim_1_trajectory[:, 0], vis_context_odr_stim_1_trajectory[:, 1], c='g')
plt.scatter(vis_context_odr_stim_2_trajectory[:, 0], vis_context_odr_stim_2_trajectory[:, 1], c='m')


plt.plot(odr_context_vis_stim_1_trajectory[:, 0], odr_context_vis_stim_1_trajectory[:, 1], c='b')
plt.plot(odr_context_vis_stim_2_trajectory[:, 0], odr_context_vis_stim_2_trajectory[:, 1], c='r')
plt.plot(odr_context_odr_stim_1_trajectory[:, 0], odr_context_odr_stim_1_trajectory[:, 1], c='g')
plt.plot(odr_context_odr_stim_2_trajectory[:, 0], odr_context_odr_stim_2_trajectory[:, 1], c='m')
plt.scatter(odr_context_vis_stim_1_trajectory[:, 0], odr_context_vis_stim_1_trajectory[:, 1], c='b')
plt.scatter(odr_context_vis_stim_2_trajectory[:, 0], odr_context_vis_stim_2_trajectory[:, 1], c='r')
plt.scatter(odr_context_odr_stim_1_trajectory[:, 0], odr_context_odr_stim_1_trajectory[:, 1], c='g')
plt.scatter(odr_context_odr_stim_2_trajectory[:, 0], odr_context_odr_stim_2_trajectory[:, 1], c='m')




# Get Diffs
vis_context_vis_distance = get_distance(vis_context_vis_stim_1_trajectory, vis_context_vis_stim_2_trajectory)
vis_context_odr_distance = get_distance(vis_context_odr_stim_1_trajectory, vis_context_odr_stim_2_trajectory)

odr_context_vis_distance = get_distance(odr_context_vis_stim_1_trajectory, odr_context_vis_stim_2_trajectory)
odr_context_odr_distance = get_distance(odr_context_odr_stim_1_trajectory, odr_context_odr_stim_2_trajectory)

plt.show()

plt.title("Just vis")
plt.plot(vis_context_vis_distance, c='b')
plt.plot(odr_context_vis_distance, c='g')
plt.show()

vis_initial_state_diff = np.subtract(vis_context_vis_distance, vis_context_odr_distance)
odr_initial_state_diff = np.subtract(odr_context_vis_distance, odr_context_odr_distance)

plt.title("Both")
plt.plot(vis_initial_state_diff, c='b')
plt.plot(odr_initial_state_diff, c='g')
plt.show()
