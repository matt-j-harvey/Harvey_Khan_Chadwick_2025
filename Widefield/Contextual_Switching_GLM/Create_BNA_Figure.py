import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize
from skimage.morphology import binary_dilation
from scipy import stats

from statsmodels.stats.multitest import fdrcorrection

from Widefield_Utils import widefield_utils



def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Allen_Atlas_Templates/churchland_outlines_aligned_single.npy")
    atlas_outline = np.roll(atlas_outline, -5, axis=1)
    atlas_outline = binary_dilation(atlas_outline)
    #atlas_outline[3:8, 115:186] = 0
    atlas_pixels = np.nonzero(atlas_outline)


    return atlas_pixels


def get_mean_response(coef_matrix_name, coef_directory, window_start, window_stop):

    # Load Matrix
    coef_matrix = np.load(os.path.join(coef_directory, coef_matrix_name + "_group_coefs.npy"))

    # Get Mean In Time Window
    mean_response = np.mean(coef_matrix[:, window_start:window_stop], axis=1)

    # Get Mean Across Mice
    mean_response = np.mean(mean_response, axis=0)

    return mean_response



def create_image(image_data, colourmap):

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = widefield_utils.get_background_pixels(indicies, image_height, image_width)

    # Get Atlas Outline Pixels
    atlas_outline_pixels = get_atlas_outline_pixels()

    # Reconstruct Into Pixel Space
    image_data = widefield_utils.create_image_from_data(image_data, indicies, image_height, image_width)

    # Convert To RGBA
    image_data = colourmap.to_rgba(image_data)

    # Add Atlas
    image_data[atlas_outline_pixels] = (1, 1, 1, 1)
    image_data[background_pixels] = (1, 1, 1, 1)

    # Remove Olfactory Bulb
    #image_data[0:62] = (1,1,1,1)
    image_data = image_data[63:]
    return image_data



def get_significance_map(coef_directory, condition_1, condition_2, window_start, window_stop):

    # Load Matrix
    group_1_coef_matrix = np.load(os.path.join(coef_directory, condition_1 + "_group_coefs.npy"))
    group_2_coef_matrix = np.load(os.path.join(coef_directory, condition_2 + "_group_coefs.npy"))

    # Get Mean In Window
    group_1_coef_matrix = np.mean(group_1_coef_matrix[:, window_start:window_stop], axis=1)
    group_2_coef_matrix = np.mean(group_2_coef_matrix[:, window_start:window_stop], axis=1)

    # Paired T Test
    t_stats, p_values = stats.ttest_rel(group_1_coef_matrix, group_2_coef_matrix, axis=0)

    # FDR Correction
    rejected, corrected_p = fdrcorrection(p_values, alpha=0.05, is_sorted=False)

    # Get Group Mean
    group_1_mean = np.mean(group_1_coef_matrix, axis=0)
    group_2_mean = np.mean(group_2_coef_matrix, axis=0)
    mean_diff = np.subtract(group_1_mean, group_2_mean)
    sig_mean_diff = np.where(rejected==1, mean_diff, 0)

    return sig_mean_diff






def create_mean_activity_figure(coef_directory, window_start, window_stop):

    # Get Mean Responeses
    vis_context_vis_1 = get_mean_response("vis_context_vis_1", coef_directory, window_start, window_stop)
    vis_context_vis_2 = get_mean_response("vis_context_vis_2", coef_directory, window_start, window_stop)
    odr_context_vis_1 = get_mean_response("odr_context_vis_1", coef_directory, window_start, window_stop)
    odr_context_vis_2 = get_mean_response("odr_context_vis_2", coef_directory, window_start, window_stop)
    vis_1_diff = np.subtract(vis_context_vis_1, odr_context_vis_1)
    vis_2_diff = np.subtract(vis_context_vis_2, odr_context_vis_2)

    # Get Sig Differences
    vis_1_sig_diff = get_significance_map(coef_directory, "vis_context_vis_1", "odr_context_vis_1", window_start, window_stop)
    vis_2_sig_diff = get_significance_map(coef_directory, "vis_context_vis_2", "odr_context_vis_2", window_start, window_stop)


    # Load CMAP
    cmap = widefield_utils.get_musall_cmap()
    mean_magnitude = 0.7
    mean_magnitude_odour = 0.3
    diff_magnitude = 0.2
    mean_colourmap = ScalarMappable(cmap=cmap, norm=(Normalize(vmin=-mean_magnitude, vmax=mean_magnitude)))
    diff_colourmap = ScalarMappable(cmap=cmap, norm=(Normalize(vmin=-diff_magnitude, vmax=diff_magnitude)))
    #odour_colourmap = ScalarMappable(cmap=cmap, norm=(Normalize(vmin=-mean_magnitude_odour, vmax=mean_magnitude_odour)))

    # Convert To Images
    vis_context_vis_1 = create_image(vis_context_vis_1, mean_colourmap)
    vis_context_vis_2 = create_image(vis_context_vis_2, mean_colourmap)
    odr_context_vis_1 = create_image(odr_context_vis_1, mean_colourmap)
    odr_context_vis_2 = create_image(odr_context_vis_2, mean_colourmap)
    vis_1_diff = create_image(vis_1_diff, diff_colourmap)
    vis_2_diff = create_image(vis_2_diff, diff_colourmap)
    vis_1_sig_diff = create_image(vis_1_sig_diff, diff_colourmap)
    vis_2_sig_diff = create_image(vis_2_sig_diff, diff_colourmap)


    # Create Figure
    figure_1 = plt.figure(figsize=(20,80))
    gridspec_1 = GridSpec(nrows=4, ncols=2)
    vis_context_vis_1_axis = figure_1.add_subplot(gridspec_1[0, 0])
    vis_context_vis_2_axis = figure_1.add_subplot(gridspec_1[0, 1])
    odr_context_vis_1_axis = figure_1.add_subplot(gridspec_1[1, 0])
    odr_context_vis_2_axis = figure_1.add_subplot(gridspec_1[1, 1])
    vis_1_diff_axis = figure_1.add_subplot(gridspec_1[2, 0])
    vis_2_diff_axis = figure_1.add_subplot(gridspec_1[2, 1])
    vis_1_sig_diff_axis = figure_1.add_subplot(gridspec_1[3, 0])
    vis_2_sig_diff_axis = figure_1.add_subplot(gridspec_1[3, 1])

    # Plot Images
    vis_context_vis_1_handle = vis_context_vis_1_axis.imshow(vis_context_vis_1, cmap=cmap, vmin=-mean_magnitude, vmax=mean_magnitude)
    vis_context_vis_2_handle = vis_context_vis_2_axis.imshow(vis_context_vis_2, cmap=cmap, vmin=-mean_magnitude, vmax=mean_magnitude)
    odr_context_vis_1_handle = odr_context_vis_1_axis.imshow(odr_context_vis_1, cmap=cmap, vmin=-mean_magnitude, vmax=mean_magnitude)
    odr_context_vis_2_handle = odr_context_vis_2_axis.imshow(odr_context_vis_2, cmap=cmap, vmin=-mean_magnitude, vmax=mean_magnitude)
    vis_1_diff_handle = vis_1_diff_axis.imshow(vis_1_diff, cmap=cmap, vmin=-diff_magnitude, vmax=diff_magnitude)
    vis_2_diff_handle = vis_2_diff_axis.imshow(vis_2_diff, cmap=cmap, vmin=-diff_magnitude, vmax=diff_magnitude)
    vis_1_sig_diff_handle = vis_1_sig_diff_axis.imshow(vis_1_sig_diff, cmap=cmap, vmin=-diff_magnitude, vmax=diff_magnitude)
    vis_2_sig_diff_handle = vis_2_sig_diff_axis.imshow(vis_2_sig_diff, cmap=cmap, vmin=-diff_magnitude, vmax=diff_magnitude)

    # Remove Axes
    vis_context_vis_1_axis.axis('off')
    vis_context_vis_2_axis.axis('off')
    odr_context_vis_1_axis.axis('off')
    odr_context_vis_2_axis.axis('off')
    vis_1_diff_axis.axis('off')
    vis_2_diff_axis.axis('off')
    vis_1_sig_diff_axis.axis('off')
    vis_2_sig_diff_axis.axis('off')

    # Add Colourbars
    shrink_fraction = 0.35
    aspect_ratio = 15
    figure_1.colorbar(vis_context_vis_1_handle, ax=vis_context_vis_1_axis,  orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(vis_context_vis_2_handle, ax=vis_context_vis_2_axis,  orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(odr_context_vis_1_handle, ax=odr_context_vis_1_axis,  orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(odr_context_vis_2_handle, ax=odr_context_vis_2_axis,  orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(vis_1_diff_handle, ax=vis_1_diff_axis,                orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(vis_2_diff_handle, ax=vis_2_diff_axis,                orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(vis_1_sig_diff_handle, ax=vis_1_sig_diff_axis,        orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)
    figure_1.colorbar(vis_2_sig_diff_handle, ax=vis_2_sig_diff_axis,        orientation='vertical', shrink=shrink_fraction, aspect=aspect_ratio)

    plt.show()

    #plt.savefig("/home/matthew/Pictures/BNA_Poster/Mean_Activity.svg")
    #plt.close()


coef_directory = r"/media/matthew/29D46574463D2856/Paper_Results/Contextual_Swtiching_GLM/Group_Regressors"
mean_window_start = 41
mean_window_stop = 83

create_mean_activity_figure(coef_directory, mean_window_start, mean_window_stop)