import os
import numpy as np
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize

import matplotlib.pyplot as plt

import Session_List
import GLM_Utils


def z_score_regressor(data_root, session, regressor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()
    pixel_means = GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    z_scored_regressor = []
    n_timepoints = np.shape(regressor)[2]
    for timepoint_index in range(n_timepoints):
        timepoint_data = regressor[:, :, timepoint_index]
        timepoint_data = np.subtract(timepoint_data, pixel_means)
        timepoint_data = np.divide(timepoint_data, pixel_stds)
        z_scored_regressor.append(timepoint_data)

    z_scored_regressor = np.array(z_scored_regressor)
    z_scored_regressor = np.nan_to_num(z_scored_regressor)

    return z_scored_regressor


def get_reaction_time(lick_trace, vis_onset, lick_threshold, max_window):

    n_timpoints = len(lick_trace)
    for time_delta in range(max_window):
        if vis_onset + time_delta < n_timpoints:
            if lick_trace[vis_onset + time_delta] >= lick_threshold:
                return time_delta

    return None


def get_onsets_specific_rt(behaviour_matrix, rt_window, lick_trace, lick_threshold):

    rt_window_start = rt_window[0] / 37
    rt_window_stop = rt_window[1] / 37

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        onset_frame = trial[18]

        if trial_type == 1:
            if correct == 1:
                if onset_frame != None:
                    reaction_time = get_reaction_time(lick_trace, onset_frame, lick_threshold, 3000)
                    if reaction_time != None:
                        if reaction_time > rt_window_start and reaction_time <= rt_window_stop:
                            onset_list.append(onset_frame)

    return onset_list




def reconstruct_activity_into_pixel_space(data_root, session, regressor, z_score):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(data_root, session, "Local_NMF", "Spatial_Components.npy"))

    # Reconstrct
    regressor = np.dot(spatial_components, np.transpose(regressor))
    regressor = np.moveaxis(regressor, -1, 0)
    print("recontructing regressor", np.shape(regressor))

    return regressor



def linearly_interpolate(data, current_spacing, new_spacing, current_x_values, offset):

    interpolated_data = []
    n_origional_frames = np.shape(data)[0]

    for x in range(1, n_origional_frames):
        start_frame = data[x-1]
        stop_frame = data[x]
        diff = np.subtract(stop_frame, start_frame)
        derivative = np.divide(diff, current_spacing)

        for ms in range(current_spacing):
            delta = np.multiply(ms, derivative)
            interpolated_frame = np.add(start_frame, delta)
            interpolated_data.append(interpolated_frame)

    interpolated_data = np.array(interpolated_data)
    interpolated_data = interpolated_data[offset::new_spacing]
    interpolated_x_values = np.array(range(current_x_values[0], current_x_values[-1]))
    interpolated_x_values = interpolated_x_values[offset::new_spacing]

    return interpolated_data, interpolated_x_values






def plot_average_lick(data_root, session_list, output_root, start_window, stop_window, reaction_time_window):

   # Fit Models For Each Session
   mouse_activity_list = []
   for mouse in tqdm(session_list, desc="Mouse"):

       session_activity_list = []
       for session in tqdm(mouse, desc="Session"):

           print(session)

           # Load Behaviour Matrix
           behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Load Lick Threshold
           lick_threshold = np.load(os.path.join(data_root, session, "Lick_Threshold.npy"))

            # Get Lick Trace
           ai_matrix = np.load(os.path.join(data_root, session, "Downsampled_AI_Matrix_Framewise.npy"))
           channel_dict = GLM_Utils.create_stimuli_dictionary()
           lick_trace = ai_matrix[channel_dict["Lick"]]

           # Get Onsets
           onset_list = get_onsets_specific_rt(behaviour_matrix, reaction_time_window, lick_trace, lick_threshold)
           print("len onsets", len(onset_list))

           if len(onset_list) > 1:

               # Load SVT
               data_matrix = np.load(os.path.join(data_root, session, "Local_NMF", "Temporal_Components.npy"))
               data_matrix = np.transpose(data_matrix)
               #print("data matrix", np.shape(data_matrix))

               # Get Data Tensor
               data_tensor = GLM_Utils.get_data_tensor(data_matrix, onset_list, start_window, stop_window, baseline_correction=True, baseline_start=0, baseline_stop=14, early_cutoff=3000)
               #print("data tensor", np.shape(data_tensor))

               # Get Mean Activity
               mean_activity = np.mean(data_tensor, axis=0)

               # Reconstruct Into Pixel Space
               mean_activity = reconstruct_activity_into_pixel_space(data_root, session, mean_activity, z_score=False)
               #print("mean_activity", np.shape(mean_activity))

               session_activity_list.append(mean_activity)

       session_activity_list = np.array(session_activity_list)
       print("session_activity_list", np.shape(session_activity_list))
       if len(session_activity_list) > 2:
           mouse_mean_activity = np.mean(session_activity_list, axis=0)
           print("mouse_mean_activity", np.shape(mouse_mean_activity))

           mouse_activity_list.append(mouse_mean_activity)

   mouse_activity_list = np.array(mouse_activity_list)
   print("mouse_activity_list", np.shape(mouse_activity_list))
   average_response = np.mean(mouse_activity_list, axis=0)
   print("average_response", average_response.nbytes)
   print("average_response", np.shape(average_response))

   # Get X Values
   x_values = list(range(start_window, stop_window))
   x_values = np.multiply(x_values, 37)

    # Interpolate Data
   average_response, x_values = linearly_interpolate(average_response, 37, 10, x_values, offset=1)
   print("interpolated_x_values", x_values)
   print("average_response", np.shape(average_response))

   # Create Save Directory
   save_directory = os.path.join(output_root, "Mean_Activity_Example_Interpolated")
   if not os.path.exists(save_directory):
       os.makedirs(save_directory)

    # Get Colourmap
   cmap = plt.get_cmap("jet")
   magnitude = [0, 0.05]
   norm = Normalize(vmin=magnitude[0], vmax=magnitude[1])
   colourmap = ScalarMappable(cmap=cmap, norm=norm)

   # Get Atlas Outlines
   atlas_pixels = GLM_Utils.get_atlas_outline_pixels()

   # Load Mask
   indicies, image_height, image_width = GLM_Utils.load_tight_mask()

   # Get Background Pixels
   background_pixels = GLM_Utils.get_background_pixels(indicies, image_height, image_width)


   n_frames = np.shape(average_response)[0]



   for frame_index in range(n_frames):

       # Extract Frame
       data = average_response[frame_index]

       # Create Axis
       figure_1 = plt.figure()
       axis_1 = figure_1.add_subplot(1,1,1)

       # Convert To Colour
       data = colourmap.to_rgba(data)

       # Set Background To Black
       data[background_pixels] = (1, 1, 1, 1)

       # Set Atlas Outlines To White
       data[atlas_pixels] = (1, 1, 1, 1)

       # Remove Olfactory Bulb
       data= data[58:]

       # Plot Data
       im = axis_1.imshow(data, vmin=magnitude[0], vmax=magnitude[1], cmap=cmap)

       # Remove Axis
       axis_1.axis('off')

       # Add Colourbar
       colourbar_ticks = np.around(np.linspace(start=magnitude[0], stop=magnitude[1], num=5), 2)
       figure_1.colorbar(im, ax=axis_1, orientation='vertical', ticks=colourbar_ticks)

       # Set Title
       axis_1.set_title(str(x_values[frame_index]) + "ms")

       plt.savefig(os.path.join(save_directory, str(frame_index).zfill(3) + ".png"))
       plt.close()


# Set Directories
data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Widefield_Opto"
output_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Analysis_Output\Widefield_GLM"

# Select Analysis Details
frame_period = 37
start_window_ms = -500
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

# Load Session List
session_list = Session_List.nested_session_list

# Run Pipeline
plot_average_lick(data_root, session_list, output_root, start_window, stop_window, reaction_time_window=[1700,1900])

