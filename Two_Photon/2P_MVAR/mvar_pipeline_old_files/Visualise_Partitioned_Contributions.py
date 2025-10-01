import os
import numpy as np
import matplotlib.pyplot as plt#
from scipy import stats

import MVAR_Utils_2P
import Plotting_Functions

def project_onto_lick_cd(neural_activity, lick_cd):
    n_timepoints, n_neurons = np.shape(neural_activity)

    lick_cd_projection = []
    for timepoint_index in range(n_timepoints):
        timepoint_data = neural_activity[timepoint_index]
        timepoint_projection = np.dot(timepoint_data, lick_cd)
        lick_cd_projection.append(timepoint_projection)

    lick_cd_projection = np.array(lick_cd_projection)
    print("lick_cd_projection", np.shape(lick_cd_projection))
    return lick_cd_projection



def view_mean_weights(data_root, session_list, mvar_output_root, start_window, stop_window, frame_rate):

    for session in session_list:

        # Load Lick CD
        lick_cd = np.load(os.path.join(data_root, session, "Coding_Dimensions", "Lick_CD.npy"))
        print("lick cd", np.shape(lick_cd))

        # Load MVAR Parameters - The Structure is:     DesignMatrix = np.concatenate((dFtot_negshift.T, stimblocks.T, behaviourtot.T), axis=1)  # design matrix
        model_dict = np.load(os.path.join(mvar_output_root, session, "Full_Model", "visual" + "_Model_Dict.npy"), allow_pickle=True)[()]
        mvar_parameters = model_dict["MVAR_Parameters"]  # Structure (N_Neurons, N_Regressors)
        Nt = model_dict["Nt"]
        Nstim = model_dict["N_stim"]
        Nvar = model_dict["Nvar"]
        print("mvar_parameters", np.shape(mvar_parameters))
        stim_weights = mvar_parameters[:, Nvar:Nvar+Nt]

        plt.title("MVAR Parameters")
        plt.imshow(mvar_parameters)
        plt.show()

        plt.imshow(stim_weights)
        MVAR_Utils_2P.forceAspect(plt.gca())
        plt.show()

        lick_cd_projection = project_onto_lick_cd(np.transpose(stim_weights), lick_cd)
        plt.plot(lick_cd_projection)
        plt.show()


def visualise_component_contribution(data_root, session_list, mvar_output_root, start_window, stop_window, frame_rate, model_component):

    visual_group_list = []
    odour_group_list = []


    for session in session_list:

        # Load Lick CD
        lick_cd = np.load(os.path.join(data_root, session,"Coding_Dimensions", "Lick_CD.npy"))
        print("lick cd", np.shape(lick_cd))

        # Load Regression Matricies
        DesignMatrix, dFtot, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, mvar_output_root, "visual")

        # Load Contributions
        visual_contribution = np.load(os.path.join(mvar_output_root, session, "Partitioned_Contribution", "visual", "vis_1_" + model_component + "_contribution.npy"))
        odour_contribution = np.load(os.path.join(mvar_output_root, session, "Partitioned_Contribution", "odour", "vis_1_" + model_component + "_contribution.npy"))
        print("visual_contribution", np.shape(visual_contribution))

        # View Weights
        """
        plt.imshow(np.transpose(visual_contribution))
        MVAR_Utils_2P.forceAspect(plt.gca(), aspect=1)
        plt.show()
        """
        # Get CD Projections
        visual_contribution_projection = project_onto_lick_cd(visual_contribution, lick_cd)
        odour_contribution_projection = project_onto_lick_cd(odour_contribution, lick_cd)

        visual_group_list.append(visual_contribution_projection)
        odour_group_list.append(odour_contribution_projection)

        x_values = list(range(start_window, stop_window))
        frame_period = 1.0 / frame_rate
        x_values = np.multiply(x_values, frame_period)
        x_values = x_values[timewindow]

        plt.plot(x_values, visual_contribution_projection, c='b')
        plt.plot(x_values, odour_contribution_projection, c='g')



    Plotting_Functions.plot_line_graph(visual_group_list, odour_group_list, x_values)
    plt.show()

    """
    # Visualise
    mean_visual_group = np.mean(np.array(visual_group_list), axis=0)
    mean_odour_group = np.mean(np.array(odour_group_list), axis=0)


    print("p values", p_values)

    plt.title(model_component)
    plt.plot(x_values, mean_visual_group, c='b')
    plt.plot(x_values, mean_odour_group, c='g')
    plt.show()
    """


def check_alignment(data_root, session_list, mvar_output_root, start_window, stop_window, frame_rate):
    visual_group_list = []
    odour_group_list = []


    figure_1 = plt.figure(figsize=(4,6))
    axis_1 = figure_1.add_subplot(1,1,1)

    for session in session_list:

        # Load Lick CD
        lick_cd = np.load(os.path.join(data_root, session,"Coding_Dimensions", "Lick_CD.npy"))
        print("lick cd", np.shape(lick_cd))

        # Load Models
        visual_model_dict = np.load(os.path.join(mvar_output_root, session, "Full_Model", "visual" + "_Model_Dict.npy"), allow_pickle=True)[()]
        visual_mvar_parameters = visual_model_dict["MVAR_Parameters"]  # Structure (N_Neurons, N_Regressors)
        print("visual_mvar_parameters", np.shape(visual_mvar_parameters))

        odour_model_dict = np.load(os.path.join(mvar_output_root, session, "Full_Model", "odour" + "_Model_Dict.npy"), allow_pickle=True)[()]
        odour_mvar_parameters = odour_model_dict["MVAR_Parameters"]  # Structure (N_Neurons, N_Regressors)
        print("odour mvar parameters")

        # Get Session Details
        DesignMatrix, dFtot, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, mvar_output_root, "visual")
        x_values = list(range(start_window, stop_window))
        frame_period = 1.0 / frame_rate
        x_values = np.multiply(x_values, frame_period)
        x_values = x_values[timewindow]
        print("x values", x_values)

        response_window = [0, 0.5]
        response_indicies = []
        window_size = len(x_values)
        for timepoint_index in range(window_size):
            timepoint_time = x_values[timepoint_index]
            if timepoint_time > response_window[0] and timepoint_time < response_window[1]:
                response_indicies.append(timepoint_index)


        print("response indicies", response_indicies)

        # Extract Stimuli Parameters
        visual_stimulus_weights = visual_mvar_parameters[:, Nvar:Nvar+Nt]
        odour_stimulus_weights = odour_mvar_parameters[:, Nvar:Nvar+Nt]

        visual_stimulus_weights = visual_stimulus_weights[:, response_indicies]
        odour_stimulus_weights = odour_stimulus_weights[:, response_indicies]

        visual_stimulus_weights = np.mean(visual_stimulus_weights, axis=1)
        odour_stimulus_weights = np.mean(odour_stimulus_weights, axis=1)




        # Extract Recurrent Weights
        visual_recurrent_weights = visual_mvar_parameters[:, 0:Nvar]
        odour_recurrent_weights = odour_mvar_parameters[:, 0:Nvar]

        np.fill_diagonal(visual_recurrent_weights, 0)
        np.fill_diagonal(odour_recurrent_weights, 0)

        print("visual_stimulus_weights", np.shape(visual_stimulus_weights))
        print("odour_stimulus_weights", np.shape(odour_stimulus_weights))

        print("visual_recurrent_weights", np.shape(visual_recurrent_weights))
        print("odour_recurrent_weights", np.shape(odour_recurrent_weights))

        #visual_effect = np.dot(visual_recurrent_weights, visual_stimulus_weights)
        #odour_effect = np.dot(odour_recurrent_weights, visual_stimulus_weights)

        #visual_effect = np.dot(visual_stimulus_weights, visual_recurrent_weights)
        #odour_effect = np.dot(visual_stimulus_weights, odour_recurrent_weights)
        visual_effect = np.matmul(visual_recurrent_weights, visual_stimulus_weights)
        odour_effect = np.matmul(odour_recurrent_weights, visual_stimulus_weights)



        visual_effect_projection = np.dot(visual_effect, lick_cd)
        odour_effect_projection = np.dot(odour_effect, lick_cd)

        print("visual_effect_projection", visual_effect_projection)
        print("odour_effect_projection", odour_effect_projection)


        visual_group_list.append(visual_effect_projection)
        odour_group_list.append(odour_effect_projection)
        axis_1.scatter([0, 1], [visual_effect_projection, odour_effect_projection], c=['b', 'g'], zorder=2)
        axis_1.plot([0, 1], [visual_effect_projection, odour_effect_projection], c='Grey', zorder=1)

    axis_1.set_xticks(ticks=[0,1], labels=["Visual", "Odour"])
    axis_1.set_xlim(-0.2, 1.2)
    axis_1.spines[['right', 'top']].set_visible(False)
    axis_1.set_xlabel("Context")
    axis_1.set_ylabel("Lick CD (A.U.)")

    t_stat, p_value = stats.ttest_rel(visual_group_list, odour_group_list)
    print("t_stat", t_stat)
    print("p_value", p_value)
    plt.show()