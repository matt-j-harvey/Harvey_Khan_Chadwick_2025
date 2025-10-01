import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from matplotlib.pyplot import GridSpec

import Visualise_Results_Utils



def get_sorting_indicies(raster, sorting_window):
        response_activity = raster[sorting_window[0]:sorting_window[1]]
        response_activity = np.mean(response_activity, axis=0)
        sorted_indicies = np.argsort(response_activity)
        sorted_indicies = np.flip(sorted_indicies)
        return sorted_indicies


def plot_full_raster(delta_f_matrix, save_directory):

    # Load Full Prediction
    prediction = np.load(os.path.join(save_directory, "Full_Prediction.npy"))

    # Get Data Magnitude
    vmin = np.percentile(np.abs(delta_f_matrix), q=5)
    vmax = np.percentile(np.abs(delta_f_matrix), q=95)
    print("real data", np.shape(delta_f_matrix))

    # Create Figure
    figure_1 = plt.figure(figsize=(15, 10))
    real_axis = figure_1.add_subplot(2, 1, 1)
    prediction_axis = figure_1.add_subplot(2, 1, 2)

    # Plot Rasters
    prediction_axis.imshow(np.transpose(prediction), vmin=vmin, vmax=vmax)
    real_axis.imshow(delta_f_matrix, vmin=vmin, vmax=vmax)

    # Set Titles
    prediction_axis.set_title("Predicted")
    real_axis.set_title("Real")

    # Force ASpect Ratios
    Visualise_Results_Utils.forceAspect(real_axis, aspect=3)
    Visualise_Results_Utils.forceAspect(prediction_axis, aspect=3)

    # Save Figure
    plt.savefig(os.path.join(save_directory, "Full_Raster.png"))
    plt.close()



def plot_psth(data, axis, x_values, data_magnitude=None, title=None, onset=None):

    # get Data Magnitde
    if data_magnitude == None:
        data_magnitude = np.percentile(np.abs(data), q=95)

    n_neurons = np.shape(data)[0]
    extent = [x_values[0], x_values[-1], 0, n_neurons]
    handle_1 = axis.imshow(data, cmap="bwr", vmin=-data_magnitude, vmax=data_magnitude, extent=extent)
    plt.colorbar(handle_1)

    # Set Title
    axis.set_title(title)

    # Force Aspect
    Visualise_Results_Utils.forceAspect(axis)

    if onset != None:
        axis.axvline(0, c='k', linestyle='dashed')




def plot_stim_prediction(stimulus_list, output_directory, x_values):

    # Load Data
    mean_prediction_list = np.load(os.path.join(output_directory, "Stim_Predictions.npy"))
    mean_real_list = np.load(os.path.join(output_directory, "Stim_Actual.npy"))

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Predicted_PSTHs")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Iterate Through Each Stimulus
    n_stimuli = len(stimulus_list)
    for stimulus_index in range(n_stimuli):

        # Extract Data
        stimulus_name = stimulus_list[stimulus_index]
        stimulus_real = mean_real_list[stimulus_index]
        stimulus_predicted = mean_prediction_list[stimulus_index]

        # Sort Raster
        n_timepoints, n_neurons = np.shape(stimulus_real)
        sorting_window = int(n_timepoints / 2)
        sorted_indicies = get_sorting_indicies(stimulus_real, [sorting_window, n_timepoints])
        stimulus_real = stimulus_real[:, sorted_indicies]
        stimulus_predicted = stimulus_predicted[:, sorted_indicies]

        # Calculate Diff
        diff = np.subtract(stimulus_real, stimulus_predicted)

        # Create Figure
        figure_1 = plt.figure(figsize=(15,5))
        real_axis = figure_1.add_subplot(1, 3, 1)
        prediction_axis = figure_1.add_subplot(1, 3, 2)
        diff_axis = figure_1.add_subplot(1, 3, 3)

        # Get Data Magnitude
        vmin = np.percentile(stimulus_real, q=5)
        vmax = np.percentile(stimulus_real, q=95)



        # Plot Data
        prediction_axis.imshow(np.transpose(stimulus_predicted), vmin=vmin,      vmax=vmax,                 extent=[x_values[0], x_values[-1], 0, n_neurons])
        real_axis.imshow(np.transpose(stimulus_real),            vmin=vmin,      vmax=vmax,                 extent=[x_values[0], x_values[-1], 0, n_neurons])
        diff_axis.imshow(np.transpose(diff),                     vmin=-vmax*0.5, vmax=vmax*0.5, cmap="bwr", extent=[x_values[0], x_values[-1], 0, n_neurons])

        # Set Axis Labels
        real_axis.set_ylabel("Neurons")
        real_axis.set_xlabel("Time (S)")

        prediction_axis.set_ylabel("Neurons")
        prediction_axis.set_xlabel("Time (S)")

        diff_axis.set_ylabel("Neurons")
        diff_axis.set_xlabel("Time (S)")

        # Indicate Stim Onset
        real_axis.axvline(0, c='w', linestyle='dashed')
        prediction_axis.axvline(0, c='w', linestyle='dashed')
        diff_axis.axvline(0, c='k', linestyle='dashed')

        # Force Aspects
        Visualise_Results_Utils.forceAspect(prediction_axis)
        Visualise_Results_Utils.forceAspect(real_axis)
        Visualise_Results_Utils.forceAspect(diff_axis)

        # Set Titles
        prediction_axis.set_title("Predicted")
        real_axis.set_title("Real")
        diff_axis.set_title("Diff")
        figure_1.suptitle(stimulus_name)

        # Save Figure
        plt.savefig(os.path.join(save_directory, stimulus_list[stimulus_index] + ".png"))
        plt.close()




def plot_lick_cd_projections(output_directory, stimulus_list, x_values, colour_list=["b", "r", "g", "m", "orange", "brown"]):

    # Load Lick CD Projections
    real_list = np.load(os.path.join(output_directory, "real_lick_cd_list.npy"))
    prediction_list = np.load(os.path.join(output_directory, "predicted_lick_cd_list.npy"))

    # Create Plot Save Directory
    save_directory = os.path.join(output_directory, "Lick_CD_Projections")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create Figure
    figure_1 = plt.figure(figsize=(12, 5))
    real_axis = figure_1.add_subplot(1,2,1)
    predicted_axis = figure_1.add_subplot(1,2,2)

    # Iterate Through Each Stimulus
    n_stimului = len(stimulus_list)
    combined_projection_list = []

    for stimulus_index in range(n_stimului):

        # Project onto lick CD
        real_lick_cd_projection = real_list[stimulus_index]
        predicted_lick_cd_projection = prediction_list[stimulus_index]

        # Plot These
        real_axis.plot(x_values, real_lick_cd_projection, colour_list[stimulus_index])
        predicted_axis.plot(x_values, predicted_lick_cd_projection, colour_list[stimulus_index])

        # Add To Combined projection To Later Get Y lims
        combined_projection_list.append(real_lick_cd_projection)
        combined_projection_list.append(predicted_lick_cd_projection)

    # Get YLim
    combined_data = np.concatenate(combined_projection_list)
    ymin = np.min(combined_data)
    ymax = np.max(combined_data)
    real_axis.set_ylim([ymin, ymax])
    predicted_axis.set_ylim([ymin, ymax])

    # Remove Splines
    real_axis.spines[['right', 'top']].set_visible(False)
    predicted_axis.spines[['right', 'top']].set_visible(False)

    # Add Axis Labels
    real_axis.set_ylabel("Lick CD")
    predicted_axis.set_ylabel("Lick CD")
    real_axis.set_xlabel("Time (S)")
    predicted_axis.set_xlabel("Time (S)")

    # Add Stimulus Onset Dotted Line
    real_axis.axvline(0, c='k', linestyle='dashed')
    predicted_axis.axvline(0, c='k', linestyle='dashed')

    # Add Titles
    real_axis.set_title("Real")
    predicted_axis.set_title("Predicted")

    # Save Fig
    plt.savefig(os.path.join(save_directory, "Lick_CD_Predictions.png"))
    plt.close()


def plot_group_lick_cd_projections(output_root, session_list, model_type, stimulus_list, x_values, colour_list=["b", "r", "g", "m", "orange", "brown"]):

    # Get Group Lick CD Projections
    group_real_list = []
    group_predicted_list = []

    for session in session_list:

        session_real = np.load(os.path.join(output_root, session, "MVAR_Results", model_type, "real_lick_cd_list.npy"))
        session_predicted = np.load(os.path.join(output_root, session, "MVAR_Results", model_type, "predicted_lick_cd_list.npy"))

        group_real_list.append(session_real)
        group_predicted_list.append(session_predicted)

    # Convert To Numpy Array
    group_real_list = np.array(group_real_list)
    group_predicted_list = np.array(group_predicted_list)
    group_real_list = np.squeeze(group_real_list)
    group_predicted_list = np.squeeze(group_predicted_list)
    print("group_real_list", np.shape(group_real_list))
    print("group_predicted_list", np.shape(group_predicted_list))


    # Create Plot Save Directory
    save_directory = os.path.join(output_root, "Group_Lick_CD_Projections", model_type)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create Figure
    figure_1 = plt.figure(figsize=(12, 5))
    real_axis = figure_1.add_subplot(1, 2, 1)
    predicted_axis = figure_1.add_subplot(1, 2, 2)

    # Iterate Through Each Stimulus
    n_stimului = len(stimulus_list)
    combined_projection_list = []

    for stimulus_index in range(n_stimului):

        # Extract Stimulus Data
        stimulus_real = group_real_list[:, stimulus_index]
        stimulus_predicted = group_predicted_list[:, stimulus_index]

        # Get Mean
        stimulus_real_mean = np.mean(stimulus_real, axis=0)
        stimulus_predicted_mean = np.mean(stimulus_predicted, axis=0)

        # Get SEMS
        real_sem = stats.sem(stimulus_real, axis=0)
        predicted_sem = stats.sem(stimulus_predicted, axis=0)

        # Plot These
        real_axis.plot(x_values, stimulus_real_mean, colour_list[stimulus_index])
        predicted_axis.plot(x_values, stimulus_predicted_mean, colour_list[stimulus_index])

        # Add SEM Shading
        real_axis.fill_between(x_values, y1=np.subtract(stimulus_real_mean, real_sem),
                                         y2=np.add(stimulus_real_mean, real_sem),
                                         color=colour_list[stimulus_index],
                                         alpha=0.1)

        predicted_axis.fill_between(x_values, y1=np.subtract(stimulus_predicted_mean, predicted_sem),
                                              y2=np.add(stimulus_predicted_mean, predicted_sem),
                                              color=colour_list[stimulus_index],
                                              alpha=0.1)

        # Add To Combined projection To Later Get Y lims
        combined_projection_list.append(stimulus_real_mean)
        combined_projection_list.append(stimulus_predicted_mean)

    # Get YLim
    combined_data = np.concatenate(combined_projection_list)
    ymin = np.min(combined_data)
    ymax = np.max(combined_data)
    real_axis.set_ylim([ymin, ymax])
    predicted_axis.set_ylim([ymin, ymax])

    # Remove Splines
    real_axis.spines[['right', 'top']].set_visible(False)
    predicted_axis.spines[['right', 'top']].set_visible(False)

    # Add Axis Labels
    real_axis.set_ylabel("Lick CD")
    predicted_axis.set_ylabel("Lick CD")
    real_axis.set_xlabel("Time (S)")
    predicted_axis.set_xlabel("Time (S)")

    # Add Stimulus Onset Dotted Line
    real_axis.axvline(0, c='k', linestyle='dashed')
    predicted_axis.axvline(0, c='k', linestyle='dashed')

    # Add Titles
    real_axis.set_title("Real")
    predicted_axis.set_title("Predicted")

    # Save Fig
    plt.savefig(os.path.join(save_directory, "Lick_CD_Predictions.png"))
    plt.close()



def plot_recurrent_weights(model_dict, save_directory):

    model_params = model_dict["MVAR_Parameters"]
    n_neurons = np.shape(model_params)[0]
    recurrent_weights = model_params[:, 0:n_neurons]

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    weight_magnitude = np.percentile(np.abs(recurrent_weights), q=95)
    axis_1.imshow(recurrent_weights, cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.savefig(os.path.join(save_directory, "Recurrent_Weights.png"))
    plt.close()



def plot_stim_weights(model_dict, frame_rate, save_directory):

    # Extract Stim Weights
    stim_weight_list = Visualise_Results_Utils.extract_stim_weights(model_dict)
    visual_context_vis_1 = stim_weight_list[0]
    visual_context_vis_2 = stim_weight_list[1]
    odour_context_vis_1 = stim_weight_list[2]
    odour_context_vis_2 = stim_weight_list[3]
    odour_1 = stim_weight_list[4]
    odour_2 = stim_weight_list[5]

    print("visual_context_vis_1", np.shape(visual_context_vis_1))

    # Sort By Vis context Vis 1
    n_neurons, n_timepoints = np.shape(visual_context_vis_1)
    sorting_window = int(n_timepoints / 2)
    sorted_indicies = get_sorting_indicies(np.transpose(visual_context_vis_1), [sorting_window, n_timepoints])
    visual_context_vis_1 = visual_context_vis_1[sorted_indicies]
    visual_context_vis_2 = visual_context_vis_2[sorted_indicies]
    odour_context_vis_1 = odour_context_vis_1[sorted_indicies]
    odour_context_vis_2 = odour_context_vis_2[sorted_indicies]
    odour_1 = odour_1[sorted_indicies]
    odour_2 = odour_2[sorted_indicies]


    # Save Stim Weights
    weight_save_directory = os.path.join(save_directory, "Stimuli_Weights")
    if not os.path.exists(weight_save_directory):
        os.makedirs(weight_save_directory)

    np.save(os.path.join(weight_save_directory, "visual_context_vis_1_weights.npy"), visual_context_vis_1)
    np.save(os.path.join(weight_save_directory, "visual_context_vis_2_weights.npy"), visual_context_vis_2)
    np.save(os.path.join(weight_save_directory, "odour_context_vis_1_weights.npy"), odour_context_vis_1)
    np.save(os.path.join(weight_save_directory, "odour_context_vis_2_weights.npy"), odour_context_vis_2)
    np.save(os.path.join(weight_save_directory, "odour_1_weights.npy"), odour_1)
    np.save(os.path.join(weight_save_directory, "odour_2_weights.npy"), odour_2)

    # Create Figure
    figure_1 = plt.figure(figsize=(15, 5))
    gridspec_1 = GridSpec(nrows=2, ncols=4)

    # Create Axes
    visual_context_vis_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
    visual_context_vis_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
    odour_context_vis_1_axis =  figure_1.add_subplot(gridspec_1[1, 0])
    odour_context_vis_2_axis =  figure_1.add_subplot(gridspec_1[1, 1])
    odour_1_axis =  figure_1.add_subplot(gridspec_1[1, 2])
    odour_2_axis =  figure_1.add_subplot(gridspec_1[1, 3])

    # Plot Onset
    Nt = model_dict["Nt"]
    onset = int(Nt/2)
    x_values = list(range(-onset, onset))
    frame_period = 1.0 / frame_rate
    x_values = np.multiply(x_values, frame_period)

    # Plot Data
    plot_psth(visual_context_vis_1, visual_context_vis_1_axis, x_values, title="visual_context_vis_1", onset=onset)
    plot_psth(visual_context_vis_2, visual_context_vis_2_axis, x_values, title="visual_context_vis_2", onset=onset)
    plot_psth(odour_context_vis_1, odour_context_vis_1_axis, x_values, title="odour_context_vis_1", onset=onset)
    plot_psth(odour_context_vis_2, odour_context_vis_2_axis, x_values, title="odour_context_vis_2", onset=onset)
    plot_psth(odour_1, odour_1_axis, x_values, title="odour_1", onset=onset)
    plot_psth(odour_2, odour_2_axis, x_values, title="odour_2", onset=onset)

    plt.savefig(os.path.join(save_directory, "Stim_Weights.png"))
    plt.close()



def plot_partitioned_contributions(output_directory, stimulus_list, frame_rate):

    # Load Partitioned Contributions
    recurrent_contributions = np.load(os.path.join(output_directory, "Partitioned_Contributions", "recurrent_contribution.npy"))
    stimuli_contributions = np.load(os.path.join(output_directory, "Partitioned_Contributions", "stimulus_contribution.npy"))
    print("recurrent_contributions", np.shape(recurrent_contributions))

    # Create X Values
    Nt = np.shape(recurrent_contributions[0])[0]
    onset = int(Nt/2)
    x_values = list(range(-onset, onset))
    frame_period = 1.0 / frame_rate
    x_values = np.multiply(x_values, frame_period)

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Partitioned Contributions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Iterate Through Each Stimulus
    n_stimuli = len(stimulus_list)
    for stimulus_index in range(n_stimuli):

        # Get Stimulus Data
        stimuli_data = np.transpose(stimuli_contributions[stimulus_index])
        recurrence_data = np.transpose(recurrent_contributions[stimulus_index])
        difference = np.subtract(recurrence_data, stimuli_data)
        print("recurrent data", np.shape(recurrence_data))

        # Sort By Stimuli Data
        n_neurons, n_timepoints = np.shape(stimuli_data)
        sorting_window = int(n_timepoints / 2)
        sorted_indicies = get_sorting_indicies(np.transpose(stimuli_data), [sorting_window, n_timepoints])
        stimuli_data = stimuli_data[sorted_indicies]
        recurrence_data = recurrence_data[sorted_indicies]
        difference = difference[sorted_indicies]

        # Create Figure
        figure_1 = plt.figure(figsize=(15, 5))
        gridspec_1 = GridSpec(nrows=1, ncols=3)

        # Create Axes
        stimulus_axis = figure_1.add_subplot(gridspec_1[0, 0])
        recurrence_axis = figure_1.add_subplot(gridspec_1[0, 1])
        difference_axis = figure_1.add_subplot(gridspec_1[0, 2])

        # Plot Data
        plot_psth(stimuli_data, stimulus_axis, x_values, title="Stimulus_Contribution", onset=onset)
        plot_psth(recurrence_data, recurrence_axis, x_values, title="Recurrent_Contribution", onset=onset)
        plot_psth(difference, difference_axis, x_values, title="Difference", onset=onset)

        # Add Title
        figure_1.suptitle(stimulus_list[stimulus_index])

        plt.savefig(os.path.join(save_directory, "stimulus_list[stimulus_index]" + ".png"))
        plt.close()




def plot_partitioned_lick_cds(output_directory, stimulus_list, frame_rate):

    # Load Partitioned Contributions
    recurrent_contributions = np.load(os.path.join(output_directory, "Partitioned_Contributions", "recurrent_contribution_lick_cd.npy"))
    stimuli_contributions = np.load(os.path.join(output_directory, "Partitioned_Contributions", "stimulus_contribution_lick_cd.npy"))
    print("recurrent_contributions", np.shape(recurrent_contributions))

    # Create X Values
    Nt = np.shape(recurrent_contributions[0])[0]
    onset = int(Nt / 2)
    x_values = list(range(-onset, onset))
    frame_period = 1.0 / frame_rate
    x_values = np.multiply(x_values, frame_period)

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Partitioned_Contributions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Iterate Through Each Stimulus
    n_stimuli = len(stimulus_list)
    for stimulus_index in range(n_stimuli):

        # Get Stimulus Data
        stimuli_data = stimuli_contributions[stimulus_index]
        recurrence_data = recurrent_contributions[stimulus_index]
        magnitude = np.max(np.abs(np.concatenate([stimuli_data, recurrence_data])))*1.1

        # Create Figure
        figure_1 = plt.figure(figsize=(15, 5))

        # Create Axes
        axis_1 = figure_1.add_subplot(1,1,1)

        # Plot Data
        axis_1.plot(x_values, recurrence_data, c='m')
        axis_1.plot(x_values, stimuli_data, c='b')

        # Add Title
        axis_1.set_title(stimulus_list[stimulus_index])

        # Add Onset
        axis_1.axvline(0, c='k', linestyle='dashed')

        # Set Y Lim
        axis_1.set_ylim([-magnitude, magnitude])

        plt.savefig(os.path.join(save_directory, stimulus_list[stimulus_index] + "_Partitioned.png"))
        plt.close()





def plot_group_partitioned_lick_cd_projections(output_root, session_list, model_type, stimulus_list, x_values, colour_list=["b", "r", "g", "m", "orange", "brown"]):

    # Get Group Lick CD Projections
    group_stimuli_list = []
    group_recurrent_list = []

    for session in session_list:

        session_stimuli = np.load(os.path.join(output_root, session, "MVAR_Results", model_type, "Partitioned_Contributions",  "stimulus_contribution_lick_cd.npy"))
        session_recurrent = np.load(os.path.join(output_root, session, "MVAR_Results", model_type, "Partitioned_Contributions",  "recurrent_contribution_lick_cd.npy"))

        group_stimuli_list.append(session_stimuli)
        group_recurrent_list.append(session_recurrent)


    # Convert To Numpy Array
    group_stimuli_list = np.array(group_stimuli_list)
    group_recurrent_list = np.array(group_recurrent_list)

    # Create Plot Save Directory
    save_directory = os.path.join(output_root, "Group_Lick_CD_Projections", model_type)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Create Figure
    figure_1 = plt.figure(figsize=(12, 5))
    stimuli_axis = figure_1.add_subplot(1, 2, 1)
    recurrent_axis = figure_1.add_subplot(1, 2, 2)

    # Iterate Through Each Stimulus
    n_stimului = len(stimulus_list)
    combined_projection_list = []

    for stimulus_index in range(n_stimului):

        # Extract Stimulus Data
        stimulus_only = group_stimuli_list[:, stimulus_index]
        stimulus_recurrent = group_recurrent_list[:, stimulus_index]

        # Get Mean
        stimulus_only_mean = np.mean(stimulus_only, axis=0)
        stimulus_recurrent_mean = np.mean(stimulus_recurrent, axis=0)

        # Get SEMS
        stimulus_only_sem = stats.sem(stimulus_only, axis=0)
        stimulus_recurrent_sem = stats.sem(stimulus_recurrent, axis=0)

        # Plot These
        stimuli_axis.plot(x_values, stimulus_only_mean, colour_list[stimulus_index])
        recurrent_axis.plot(x_values, stimulus_recurrent_mean, colour_list[stimulus_index])

        # Add SEM Shading
        stimuli_axis.fill_between(x_values, y1=np.subtract(stimulus_only_mean, stimulus_only_sem),
                                         y2=np.add(stimulus_only_mean, stimulus_only_sem),
                                         color=colour_list[stimulus_index],
                                         alpha=0.1)

        recurrent_axis.fill_between(x_values, y1=np.subtract(stimulus_recurrent_mean, stimulus_recurrent_sem),
                                              y2=np.add(stimulus_recurrent_mean, stimulus_recurrent_sem),
                                              color=colour_list[stimulus_index],
                                              alpha=0.1)

        # Add To Combined projection To Later Get Y lims
        combined_projection_list.append(np.add(stimulus_only_mean, stimulus_only_sem))
        combined_projection_list.append(np.add(stimulus_recurrent_mean, stimulus_recurrent_sem))
        combined_projection_list.append(np.subtract(stimulus_only_mean, stimulus_only_sem))
        combined_projection_list.append(np.subtract(stimulus_recurrent_mean, stimulus_recurrent_sem))



    # Get YLim
    combined_data = np.concatenate(combined_projection_list)
    ymin = np.min(combined_data) * 1.2
    ymax = np.max(combined_data) * 1.2
    stimuli_axis.set_ylim([ymin, ymax])
    recurrent_axis.set_ylim([ymin, ymax])

    # Remove Splines
    stimuli_axis.spines[['right', 'top']].set_visible(False)
    recurrent_axis.spines[['right', 'top']].set_visible(False)

    # Add Axis Labels
    stimuli_axis.set_ylabel("Lick CD")
    stimuli_axis.set_xlabel("Time (S)")
    recurrent_axis.set_ylabel("Lick CD")
    recurrent_axis.set_xlabel("Time (S)")

    # Add Stimulus Onset Dotted Line
    stimuli_axis.axvline(0, c='k', linestyle='dashed')
    recurrent_axis.axvline(0, c='k', linestyle='dashed')

    # Add Titles
    stimuli_axis.set_title("Stimuli")
    recurrent_axis.set_title("Recurrence")

    # Save Fig
    plt.savefig(os.path.join(save_directory, "Lick_CD_Predictions_Partitioned.png"))
    plt.close()
