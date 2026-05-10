import os

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import Session_List
import GLM_Utils
import Mixed_Effects_Modelling

def get_group_jaw_motion_energy(data_root, session_list):

    group_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Get Jaw Motion Energy
            jaw_motion_energy = np.load(os.path.join(data_root, session, "Mousecam_Analysis", "Mean_Jaw_Motion_Energy.npy"))
            jaw_motion_energy = stats.zscore(jaw_motion_energy)

            # Get Onsets
            onsets = GLM_Utils.get_cr_onsets(behaviour_matrix)

            # Get Data Tensor
            tensor = GLM_Utils.get_data_tensor(jaw_motion_energy, onsets, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000, return_onsets=False)

            # Get Mean
            mean = np.mean(tensor, axis=0)
            print("mean", np.shape(mean))

            mouse_list.append(mean)

        group_list.append(mouse_list)

    return group_list



def get_window_mean(nested_list, window_start):

    window_values_list = []
    for mouse in nested_list:
        mouse_list = []
        for session in mouse:
            session_mean = np.mean(session[window_start:])
            mouse_list.append(session_mean)
        window_values_list.append(mouse_list)

    return window_values_list


def plot_session_scatter(control_values, hom_values):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    for mouse in control_values:
        n_values = len(mouse)
        x_values = np.zeros(n_values)
        jitter = np.random.uniform(low=-0.1, high=0.1, size=n_values)
        x_values = np.add(x_values, jitter)
        axis_1.scatter(x_values, mouse, alpha=0.5)

    for mouse in hom_values:
        n_values = len(mouse)
        x_values = np.ones(n_values)
        jitter = np.random.uniform(low=-0.1, high=0.1, size=n_values)
        x_values = np.add(x_values, jitter)
        axis_1.scatter(x_values, mouse)

    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_xlim(-0.5, 1.5)
    axis_1.set_xticks([0, 1], labels=["Wildtype", "Neurexin"])



    plt.show()


frame_period = 36
start_window_ms = -1000
stop_window_ms = 1000
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

control_session_list = Session_List.control_switching
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"

hom_session_list = Session_List.neurexin_switching
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"


# Get Group Results
control_energy = get_group_jaw_motion_energy(control_data_root, control_session_list)
hom_energy = get_group_jaw_motion_energy(hom_data_root, hom_session_list)

# Get WIndow Means
control_window_mean = get_window_mean(control_energy, np.abs(start_window))
hom_window_mean = get_window_mean(hom_energy, np.abs(start_window))

print("control_window_mean", control_window_mean)
print("hom_window_mean", hom_window_mean)

Mixed_Effects_Modelling.fit_jaw_mixedlm(control_window_mean, hom_window_mean, group_names=("A", "B"), reml=True)


plot_session_scatter(control_window_mean, hom_window_mean)