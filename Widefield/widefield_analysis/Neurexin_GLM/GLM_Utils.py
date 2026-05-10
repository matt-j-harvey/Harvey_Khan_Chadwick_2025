import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import pickle
from skimage.morphology import binary_dilation
from skimage.segmentation import flood_fill


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor


def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary






def get_data_tensor(df_matrix, onset_list, start_window, stop_window, baseline_correction=False, baseline_start=0, baseline_stop=5, early_cutoff=3000, return_onsets=False):

    n_timepoints = np.shape(df_matrix)[0]

    tensor = []
    selected_onsets = []
    for onset in onset_list:
        trial_start = onset + start_window
        trial_stop = onset + stop_window

        if trial_start >= early_cutoff and trial_stop < n_timepoints:
            trial_data = df_matrix[trial_start:trial_stop]
            selected_onsets.append(onset)

            if baseline_correction == True:
                baseline = trial_data[baseline_start:baseline_stop]
                baseline_mean = np.mean(baseline, axis=0)
                trial_data = np.subtract(trial_data, baseline_mean)

            tensor.append(trial_data)

    tensor = np.array(tensor)

    if return_onsets == True:
        return tensor, selected_onsets
    else:
        return tensor



def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image




def create_activity_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Activity Matrix
    activity_matrix = np.load(os.path.join(data_root_directory, session, "Local_NMF", "Temporal_Components.npy"))
    activity_matrix = np.transpose(activity_matrix)
    number_of_timepoints, number_of_components = np.shape(activity_matrix)

    # Load Onsets
    onsets_list = np.load(os.path.join(mvar_output_directory, session, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensors
    activity_tensor, selected_onsets = get_data_tensor(activity_matrix, onsets_list, start_window, stop_window, return_onsets=True)

    # Convert Tensor To Array
    activity_tensor = np.array(activity_tensor)

    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "tensor":activity_tensor,
        "start_window": start_window,
        "stop_window": stop_window,
        "onsets_file": onsets_file,
        "selected_onsets":selected_onsets,
    }

    # Save Trial Tensor
    save_directory = os.path.join(mvar_output_directory, session, "Activity_Tensors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(save_directory, tensor_name)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)





def create_behaviour_tensor(data_root_directory, session, mvar_output_directory, onsets_file, start_window, stop_window):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(mvar_output_directory, session, "Behaviour", "Behavioural_Regressor_Matrix.npy"))
    number_of_timepoints, number_of_components = np.shape(behaviour_matrix)

    # Load Onsets
    onsets_list = np.load(os.path.join(mvar_output_directory, session, "Stimuli_Onsets", onsets_file))

    # Get Activity Tensors
    behaviour_tensor, selected_onsets = get_data_tensor(behaviour_matrix, onsets_list, start_window, stop_window, return_onsets=True)

    # Convert Tensor To Array
    behaviour_tensor = np.array(behaviour_tensor)

    # Create Activity Tensor Dict
    trial_tensor_dictionary = {
        "tensor":behaviour_tensor,
        "start_window": start_window,
        "stop_window": stop_window,
        "onsets_file": onsets_file,
        "selected_onsets": selected_onsets,
    }

    # Save Trial Tensor
    save_directory = os.path.join(mvar_output_directory, session, "Behaviour_Tensors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    tensor_name = onsets_file.replace("_onsets.npy", "")
    tensor_name = tensor_name.replace("_onset_frames.npy", "")
    tensor_file = os.path.join(save_directory, tensor_name)

    with open(tensor_file + ".pickle", 'wb') as handle:
        pickle.dump(trial_tensor_dictionary, handle, protocol=4)




def load_tight_mask():
    tight_mask_file = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\Tight_Mask_Dict.npy"
    tight_mask_dict = np.load(tight_mask_file, allow_pickle=True)[()]
    indicies = tight_mask_dict["indicies"]
    image_height = tight_mask_dict["image_height"]
    image_width = tight_mask_dict["image_width"]
    return indicies, image_height, image_width

def load_atlas():
    atlas = np.load(r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\M2_Three_Segments_Masked_All_Sessions.npy")
    return atlas



def get_roi_pixels(atlas, roi_list):

    # Mask Atlas
    indicies, image_height, image_width = load_tight_mask()
    atlas = np.reshape(atlas, image_height * image_width)
    atlas = atlas[indicies]

    selected_indicies = []
    for roi in roi_list:
        roi_mask = np.where(atlas==roi, 1, 0)
        roi_indicies = np.argwhere(roi_mask)

        for index in roi_indicies:
            selected_indicies.append(index[0])

    selected_indicies = np.array(selected_indicies)
    return selected_indicies




def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load(r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\churchland_outlines_aligned_single.npy")
    atlas_outline = np.roll(atlas_outline, -5, axis=1)
    atlas_outline = binary_dilation(atlas_outline)
    #atlas_outline[3:8, 115:186] = 0
    atlas_pixels = np.nonzero(atlas_outline)


    return atlas_pixels


def get_full_outlines():

    # Load Atlas
    atlas_outline = np.load(r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\churchland_outlines_aligned_single.npy")
    atlas_outline = np.roll(atlas_outline, -5, axis=1)
    atlas_outline = binary_dilation(atlas_outline)
    flood_filled_atlas_outline = flood_fill(atlas_outline, seed_point=(0,0), new_value=2)
    flood_filled_atlas_outline = np.ndarray.astype(flood_filled_atlas_outline, int)
    new_atlas_outline = np.subtract(flood_filled_atlas_outline, atlas_outline)
    outline_pixels = np.nonzero(new_atlas_outline)
    return outline_pixels


def get_background_pixels(indicies, image_height, image_width):
    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_pixels = np.nonzero(template)
    return background_pixels



def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image


def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    return cmap



def get_mean_sd(data):
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound




def get_irrel_vis_2_onsets(behaviour_matrix):
    onset_list = []

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_irrel_type = trial[6]
        ignore_irrel = trial[7]
        trial_irrel_onset = trial[20]

        if trial_correct == 1:
            if trial_type == 3 or trial_type == 4:
                if trial_irrel_type == 2:
                    if ignore_irrel == 1:
                        if trial_irrel_onset != None:
                            onset_list.append(trial_irrel_onset)

    return onset_list

def get_cr_onsets(behaviour_matrix):
    onset_list = []

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_stim_onset = trial[18]

        if trial_correct == 1:
            if trial_type == 2:
                if trial_stim_onset != None:
                    onset_list.append(trial_stim_onset)

    return onset_list


def get_hit_onsets(behaviour_matrix):
    onset_list = []

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_stim_onset = trial[18]

        if trial_correct == 1:
            if trial_type == 1:
                if trial_stim_onset != None:
                    onset_list.append(trial_stim_onset)

    return onset_list



def get_vis_1_onsets(behaviour_matrix):
    onset_list = []

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_stim_onset = trial[18]

        if trial_type == 1:
            if trial_stim_onset != None:
                onset_list.append(trial_stim_onset)

    return onset_list


def get_vis_2_onsets(behaviour_matrix):
    onset_list = []

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_stim_onset = trial[18]

        if trial_type == 2:
            if trial_stim_onset != None:
                onset_list.append(trial_stim_onset)

    return onset_list




def get_odour_2_onsets(behaviour_matrix):
    onset_list = []

    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_stim_onset = trial[18]

        if trial_correct == 1:
            if trial_type == 4:
                if trial_stim_onset != None:
                    onset_list.append(trial_stim_onset)

    return onset_list



def load_lick_trace(data_root, session):
    ai_matrix = np.load(os.path.join(data_root, session, "Downsampled_AI_Matrix_Framewise.npy"))
    channel_dict = create_stimuli_dictionary()
    lick_trace = ai_matrix[channel_dict["Lick"]]
    return lick_trace


def get_hit_onsets_rt_window(data_root, session, rt_window_start, rt_window_stop):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(data_root, session, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Load Lick Trace
    lick_trace = load_lick_trace(data_root, session)

    # Load Lick Threshold
    lick_threshold = np.load(os.path.join(data_root, session, "Lick_Threshold.npy"))

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        trial_correct = trial[3]
        trial_stim_onset = trial[18]

        if trial_correct == 1:def z_score_regressor(data_root, session, regressor):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = load_tight_mask()
    pixel_means = create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    regressor = np.subtract(regressor, pixel_means)
    regressor = np.divide(regressor, pixel_stds)

    return regressor
            if trial_type == 1:
                if trial_stim_onset != None:

                    # Get RT
                    trial_rt =  get_reaction_time(lick_trace, trial_stim_onset, lick_threshold, max_window=3000)
                    trial_rt = trial_rt * 37

                    if trial_rt >= rt_window_start and trial_rt < rt_window_stop:
                        onset_list.append(trial_stim_onset)

    return onset_list






def reconstruct_regressor_into_pixel_space(data_root, session, regressor):

    # Load Spatial Components
    spatial_components = np.load(os.path.join(data_root, session, "Local_NMF", "Spatial_Components.npy"))

    # Reconstrct
    regressor = np.dot(spatial_components, np.transpose(regressor))
    regressor = np.moveaxis(regressor, -1, 0)
    print("regressor", np.shape(regressor))

    # Z Score
    regressor = z_score_regressor(data_root, session, regressor)
    print("regressor", np.shape(regressor))

    return regressor


def get_reaction_time(lick_trace, vis_onset, lick_threshold, max_window):

    n_timpoints = len(lick_trace)
    for time_delta in range(max_window):
        if vis_onset + time_delta < n_timpoints:
            if lick_trace[vis_onset + time_delta] >= lick_threshold:
                return time_delta

    return None



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
    print("interpolated_x_values pre offset", interpolated_x_values)
    interpolated_x_values = interpolated_x_values[offset::new_spacing]

    return interpolated_data, interpolated_x_values