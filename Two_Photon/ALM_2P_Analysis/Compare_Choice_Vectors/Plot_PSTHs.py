import matplotlib.pyplot as plt
import numpy as np
import os



def view_psths(data_directory, output_directory, lick_start_window, lick_stop_window, choice_start_window, choice_stop_window):

    # Load Data Tensors
    tensor_directory = os.path.join(output_directory, "Activity_Tensors")
    visual_lick_tensor              = np.load(os.path.join(tensor_directory, "visual_lick_tensor.npy"))
    odour_lick_tensor               = np.load(os.path.join(tensor_directory, "odour_lick_tensor.npy"))
    vis_context_stable_vis_1_tensor = np.load(os.path.join(tensor_directory, "vis_context_stable_vis_1_tensor.npy"))
    vis_context_stable_vis_2_tensor = np.load(os.path.join(tensor_directory, "vis_context_stable_vis_2_tensor.npy"))
    odour_1_tensor                  = np.load(os.path.join(tensor_directory, "odour_1_tensor.npy"))
    odour_2_tensor                  = np.load(os.path.join(tensor_directory, "odour_2_tensor.npy"))

    # Get Means
    visual_lick_mean = np.mean(visual_lick_tensor, axis=0)
    odour_lick_mean = np.mean(odour_lick_tensor, axis = 0)
    vis_context_stable_vis_1_mean = np.mean(vis_context_stable_vis_1_tensor, axis = 0)
    vis_context_stable_vis_2_mean = np.mean(vis_context_stable_vis_2_tensor, axis = 0)
    odour_1_mean = np.mean(odour_1_tensor, axis = 0)
    odour_2_mean = np.mean(odour_2_tensor, axis = 0)

    # Get Choice Dimensions
    visual_choice_dimension = np.subtract(vis_context_stable_vis_1_mean, vis_context_stable_vis_2_mean)
    odour_choice_dimension = np.subtract(odour_1_mean, odour_2_mean)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_directory, "Frame_Rate.npy"))

    # Get Choice Response Windows
    choice_start_window_frames = int(choice_start_window * frame_rate)
    choice_stop_window_frames = int(choice_stop_window * frame_rate)
    print("choice_start_window_frames", choice_start_window_frames)
    print("choice_stop_window_frames", choice_stop_window_frames)

    # Create Save Directory
    save_directory = os.path.join(output_directory, "PSTHs")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # View Each PSTH
    view_two_mean_psth(visual_lick_mean, odour_lick_mean, lick_start_window, lick_stop_window, frame_rate, 0, 8, save_directory, "Lick Dimension Comparison", ["Visual Context Licks", "Odour Context Licks"])

    view_two_mean_psth(visual_choice_dimension, odour_choice_dimension, choice_start_window, choice_stop_window, frame_rate, np.abs(choice_start_window_frames), choice_stop_window_frames, save_directory, "Choice Dimension Comparison", ["Visual Choice Dimension", "Odour Choice Dimension"])




def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def sort_raster(master_raster, other_raster_list, sorting_window_start, sorting_window_stop):

    # Get Mean Response in Sorting Window
    response = sig_activity[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)

    # Get Sorted Indicies
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    # Sort Rasters
    sorted_master_raster = master_raster[:, sorted_indicies]
    sorted_raster_list = []
    for raster in other_raster_list:
        sorted_raster = raster[:, sorted_indicies]
        sorted_raster_list.append(sorted_raster)

    return sorted_master_raster, sorted_raster_list



def view_two_mean_psth(mean_1, mean_2, start_window, stop_window, frame_rate, sorting_window_start, sorting_window_stop, save_directory, plot_title, subplot_titles, zero_line=False):

    # Get Diff
    diff = np.subtract(mean_1, mean_2)

    # Sort PSTHs
    response = mean_1[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    psth_1 = mean_1[:, sorted_indicies]
    psth_2 = mean_2[:, sorted_indicies]
    diff = diff[:, sorted_indicies]

    start_window_frames = int(start_window * frame_rate)
    stop_window_frames = int(stop_window * frame_rate)
    print("start window frames", start_window_frames)
    print("stop window frames", stop_window_frames)
    x_values = list(range(start_window_frames, stop_window_frames))

    print("mean 1", np.shape(mean_1))
    print("x values", x_values)

    x_values = np.divide(x_values, frame_rate)
    x_values = np.array(x_values)
    print("x values", x_values)
    n_neurons = np.shape(psth_1)[1]

    magnitude = np.percentile(np.abs(psth_1), q=99.5)

    figure_1 = plt.figure(figsize=(22, 5))
    axis_1 = figure_1.add_subplot(1, 3, 1)
    axis_2 = figure_1.add_subplot(1, 3, 2)
    axis_3 = figure_1.add_subplot(1, 3, 3)

    axis_1_handle = axis_1.imshow(np.transpose(psth_1), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])
    axis_2_handle = axis_2.imshow(np.transpose(psth_2), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])
    axis_3_handle = axis_3.imshow(np.transpose(diff), vmin=-magnitude, vmax=magnitude, cmap='bwr', extent=[x_values[0], x_values[-1], 0, n_neurons])

    if zero_line == True:
        axis_1.axvline(0, linestyle='dashed', c='k')
        axis_2.axvline(0, linestyle='dashed', c='k')
        axis_3.axvline(0, linestyle='dashed', c='k')

    # Set Axis Labels
    axis_1.set_xlabel("Time (S)")
    axis_2.set_xlabel("Time (S)")
    axis_3.set_xlabel("Time (S)")

    axis_1.set_ylabel("Neurons")
    axis_2.set_ylabel("Neurons")
    axis_3.set_ylabel("Neurons")

    # Force Aspects
    forceAspect(axis_1)
    forceAspect(axis_2)
    forceAspect(axis_3)

    # Set Titles
    axis_1.set_title(subplot_titles[0])
    axis_2.set_title(subplot_titles[1])
    axis_3.set_title("Difference")

    # Add Colourbars
    figure_1.colorbar(axis_1_handle, orientation='vertical', fraction=0.046, pad=0.04)
    figure_1.colorbar(axis_2_handle, orientation='vertical', fraction=0.046, pad=0.04)
    figure_1.colorbar(axis_3_handle, orientation='vertical', fraction=0.046, pad=0.04)

    figure_1.suptitle(plot_title)

    figure_1.savefig(os.path.join(save_directory, plot_title + ".png"))
    plt.close()

