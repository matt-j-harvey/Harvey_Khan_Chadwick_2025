import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.feature import canny
from scipy import stats

from Widefield_Utils import widefield_utils

def get_mean_and_bounds(data):
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    upper_bound = np.add(data_mean, data_sem)
    lower_bound = np.subtract(data_mean, data_sem)
    return data_mean, upper_bound, lower_bound


def plot_roi_trace(output_directory, n_bins, bin_start_list, bin_stop_list, atlas, atlas_dict, selected_roi, bin_stop_frame_list, start_window):

    # Get ROI Pixels
    roi_label = atlas_dict[selected_roi]
    image_height, image_width = np.shape(atlas)
    atlas = np.reshape(atlas, image_height * image_width)
    pixel_map = np.where(atlas == roi_label)[0]

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    colourmap = plt.get_cmap('plasma')


    for bin_index in range(n_bins):

        # Load Bin Data
        file_name = str(bin_start_list[bin_index]) + "_to_" + str(bin_stop_list[bin_index]) + ".npy"
        bin_data = np.load(os.path.join(output_directory, "RT_Bin_Means", file_name))

        # Get ROI Mean
        n_mice, image_height, image_width, n_timepoints = np.shape(bin_data)
        bin_data = np.reshape(bin_data, (n_mice, image_height * image_width, n_timepoints))
        roi_data = bin_data[:, pixel_map]
        roi_data = np.mean(roi_data, axis=1)

        # Cut Off Data at Lick
        roi_data = roi_data[:, 0:bin_stop_frame_list[bin_index]]

        # Get Mean and SD
        data_mean, upper_bound, lower_bound = get_mean_and_bounds(roi_data)

        # Plot Data
        colour = colourmap(float(bin_index) / n_bins)

        # Get X Values
        x_values = list(range(bin_stop_frame_list[bin_index]))
        x_values = np.add(x_values, start_window)
        x_values = np.multiply(x_values, 37)

        axis_1.plot(x_values, data_mean, c=colour, alpha=1)
        axis_1.scatter([x_values[-1]],[data_mean[-1]], c=colour)
        axis_1.fill_between(x_values, lower_bound, upper_bound, color=colour, alpha=0.1)

    axis_1.axvline(0, c='k', linestyle='dashed')
    axis_1.set_title(selected_roi)
    axis_1.spines['top'].set_visible(False)
    axis_1.spines['right'].set_visible(False)

    # Save Figures
    save_directory = os.path.join(output_directory, "Output_Plots")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    plt.savefig(os.path.join(save_directory, selected_roi + ".svg"))
    plt.close()

    """
    mean_response_list = np.reshape(mean_response_list, (n_bins, image_height * image_width, n_timepoints))
    print("mean_response_list", np.shape(mean_response_list))


    roi_responses = mean_response_list[:, pixel_map]
    print("roi responses", np.shape(roi_responses))

    roi_mean = np.mean(roi_responses, axis=1)
    print("roi mean", np.shape(roi_mean))


    count = 0
    for trace in roi_mean:
       

        # Scatter Final Value
        final_value = trace[bin_stop_frame_list[count]-1]
        plt.scatter(bin_stop_frame_list[count]-1,[final_value], c=colour)
        count += 1
    plt.show()

    """



"""
atlas_dict = {
    "Primary_Motor":1,
    "Somatosensory_Barrel":2,
    "Somatosensory_Limbs":3,
    "PPC":5,
    "Secondary_Visual_Medial":8,
    "Primary_Visual":9,
    "Retrosplenial":11,
    "Secondary_Visual_Lateral":12,
    "Olfactory_bulb":13,
    "Secondary_Motor_Medial":14,
    "Secondary_Motor_Anterolateral":15,
    "Secondary_Motor_Proximal":16,
}

# Load Atlas
atlas = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/M2_Three_Segments_Masked_All_Sessions.npy")
atlas = np.abs(atlas)
atlas_outlines = canny(atlas)
plt.imshow(atlas)
plt.show()

# Create RT Bins
bin_time_start = 500
bin_time_stop = 2500
n_bins = 16
bin_width = int((bin_time_stop - bin_time_start)/n_bins)
bin_start_list = list(range(bin_time_start, bin_time_stop - bin_width, bin_width))
bin_stop_list = np.add(bin_start_list, bin_width)
n_bins = len(bin_start_list)




print("bin width", bin_width)
print("bin_start_list", bin_start_list)
print("bin_stop_list", bin_stop_list)

start_window = -14
stop_window = 68

# Get RT Time List IN Frames


"""
"""
mean_directory = r"/media/matthew/29D46574463D2856/RT_Analysis"
n_files = 15

time_map_matrix = []
mean_response_list = []
stop_time_list = []
for file_index in range(n_files):
    stop_frames = np.abs(start_window) + ((int(bin_width / 36) * (file_index + 1)) + 14)
    stop_time_list.append(stop_frames)
    print("stop frames", stop_frames)
    mean_response = np.load(os.path.join(mean_directory, str(file_index).zfill(3) + ".npy"))
    mean_response_list.append(mean_response)
    print("mean response", np.shape(mean_response))
    mean_response = mean_response[:, :, 0:stop_frames]

    max_time_map = get_max_time_map(mean_response)
    time_map_matrix.append(max_time_map)

"""
#plot_roi_trace(mean_response_list, atlas, atlas_dict["Primary_Visual"], stop_time_list)