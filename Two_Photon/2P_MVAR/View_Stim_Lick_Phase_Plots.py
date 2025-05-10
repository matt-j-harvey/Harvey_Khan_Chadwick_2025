import numpy as np
import os
import matplotlib.pyplot as plt
import pickle




def get_stim_cd(model_params, n_neurons, Nt, preceeding_window):

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

    vis_response_diff = np.subtract(visual_context_vis_1, visual_context_vis_2)
    vis_response_cd = vis_response_diff / np.sqrt(np.sum(vis_response_diff ** 2))

    odr_response_diff = np.subtract(odour_context_vis_1, odour_context_vis_2)
    odr_response_cd = odr_response_diff / np.sqrt(np.sum(odr_response_diff ** 2))


    return vis_response_cd, odr_response_cd




def plot_vector_field(axis, lick_cd, context_cd, recurrent_weights, x_range, y_range, density=20):

    x = []
    y = []
    u = []
    v = []

    # np.matmul(mvar_parameters [n_neurons, n_regressors], design_matrix [n_regressors, n_timepoints])
    for x_value in np.linspace(start=x_range[0], stop=x_range[1], num=density):
        for y_value in np.linspace(start=y_range[0], stop=y_range[1], num=density):

            point_projection = np.add((lick_cd * x_value), (context_cd * y_value))
            new_point = np.matmul(recurrent_weights, point_projection)
            #point_derivative = np.subtract(new_point, point_projection)
            point_derivative = new_point

            x.append(x_value)
            y.append(y_value)
            u.append(point_derivative[0])
            v.append(point_derivative[1])

    axis.quiver(x, y, u, v, angles='xy')




def view_stim_lick_phase_plot(data_root, session, output_directory):

    # Load Model Dict
    model_dict = np.load(os.path.join(output_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
    model_params = model_dict["MVAR_Parameters"]
    Nt = model_dict["Nt"]
    preceeding_window = int(Nt/2)

    # Load Recurrent Weights
    n_neurons = np.shape(model_params)[0]
    recurrent_weights = model_params[:, 0:n_neurons]
    np.fill_diagonal(recurrent_weights, 0)

    # Get Lick CD
    lick_cd_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"
    lick_cd = np.load(os.path.join(lick_cd_directory, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

    # Get Vis Response CD
    vis_response_cd, odour_response_cd = get_stim_cd(model_params, n_neurons, Nt, preceeding_window)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,2,1)
    axis_2 = figure_1.add_subplot(1,2,2)

    x_range = [-1,1]
    y_range = [-1, 1]
    plot_vector_field(axis_1, lick_cd, vis_response_cd, recurrent_weights, x_range, y_range, density=20)
    plot_vector_field(axis_2, lick_cd, odour_response_cd, recurrent_weights, x_range, y_range, density=20)

    plt.show()


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

for session in control_session_list:
    view_stim_lick_phase_plot(data_root, session, mvar_output_root)