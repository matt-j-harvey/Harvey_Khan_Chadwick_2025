import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize
from scipy import stats
from statsmodels.stats.multitest import fdrcorrection


import Opto_GLM_Utils

def baseline_correct_regressors(regressor):

    baseline_corrected_regressor = []
    for mouse in regressor:
        mouse_baseline = mouse[0:14]
        mouse_baseline = np.mean(mouse_baseline, axis=0)
        mouse = np.subtract(mouse, mouse_baseline)
        baseline_corrected_regressor.append(mouse)
    baseline_corrected_regressor = np.array(baseline_corrected_regressor)
    return baseline_corrected_regressor




def create_image(data, colourmap, atlas_pixels, background_pixels):
    # Convert To Colour
    data = colourmap.to_rgba(data)

    # Set Background To Black
    data[background_pixels] = (1, 1, 1, 1)

    # Set Atlas Outlines To White
    data[atlas_pixels] = (1, 1, 1, 1)

    return data



def plot_mean_regressor(regressors, start_window, stop_window, save_directory):

    regressors = baseline_correct_regressors(regressors)
    mean_regressor = np.mean(regressors, axis=0)


    # Get Regressor Shape
    n_timepoints, image_height, image_width = np.shape(mean_regressor)

    # Get Colourmap
    cmap = Opto_GLM_Utils.get_musall_cmap()
    magnitude = np.percentile(np.abs(mean_regressor), q=95)
    norm = Normalize(vmin=-magnitude, vmax=magnitude)
    colourmap = ScalarMappable(cmap=cmap, norm=norm)

    # Get Atlas Outlines
    atlas_pixels = Opto_GLM_Utils.get_atlas_outline_pixels()

    # Load Mask
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = Opto_GLM_Utils.get_full_outlines()

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 37)

    for timepoint_index in range(n_timepoints):

        # Create Figure
        figure_1 = plt.figure(figsize=(7, 5))
        axis_1 = figure_1.add_subplot(1, 1, 1)

        # Get Timepoint Data
        timepoint_data = mean_regressor[timepoint_index]

        # Convert To Image
        timepoint_data = create_image(timepoint_data, colourmap, atlas_pixels, background_pixels)

        # Plot Data
        im = axis_1.imshow(timepoint_data, vmin=-magnitude, vmax=magnitude, cmap=cmap)

        # Remove Axis
        axis_1.axis('off')

        # Add Colourbar
        colourbar_ticks = np.around(np.linspace(start=-magnitude, stop=magnitude, num=5), 2)
        figure_1.colorbar(im, ax=axis_1, orientation='vertical', ticks=colourbar_ticks)

        # Set Title
        axis_1.set_title(str(x_values[timepoint_index]) + "ms")

        # Save
        plt.savefig(os.path.join(save_directory, str(timepoint_index).zfill(4) + ".png"))
        plt.close()


def fdr_p_values(p_values, indicies, image_height, image_width, alpha=0.05):
    p_values = np.nan_to_num(p_values, nan=1)
    p_values = np.reshape(p_values, image_height * image_width)
    brain_p_values = p_values[indicies]
    rejected, corrected_p = fdrcorrection(brain_p_values, alpha=alpha)
    p_values[indicies] = corrected_p
    p_values = np.reshape(p_values, (image_height, image_width))
    return p_values


def compare_regressors(regressor_1, regressor_2, save_directory, start_window, stop_window):

    # Baseline Correct Regressor
    regressor_1 = baseline_correct_regressors(regressor_1)
    regressor_2 = baseline_correct_regressors(regressor_2)

    # Get Mean Regressors
    mean_regressor_1 = np.mean(regressor_1, axis=0)
    mean_regressor_2 = np.mean(regressor_2, axis=0)

    # Get Signficance
    t_stats, p_values = stats.ttest_ind(regressor_1, regressor_2, axis=0)
    print("p_values", np.shape(p_values))

    # Should be size (n_timepoints, image_height, image_width)
    n_timepoints, image_height, image_width = np.shape(mean_regressor_1)
    print("mean regressor", np.shape(mean_regressor_1))

    # Get Colourmaps
    cmap = Opto_GLM_Utils.get_musall_cmap()

    magnitude = 0.2
    norm = Normalize(vmin=-magnitude, vmax=magnitude)
    colourmap = ScalarMappable(cmap=cmap, norm=norm)

    diff_magnitude = 0.1
    diff_norm = Normalize(vmin=-diff_magnitude, vmax=diff_magnitude)
    diff_colourmap = ScalarMappable(cmap=cmap, norm=diff_norm)

    # Get Atlas Outlines
    atlas_pixels = Opto_GLM_Utils.get_atlas_outline_pixels()

    # Load Mask
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()

    # Get Background Pixels
    # background_pixels = GLM_Utils.get_background_pixels(indicies, image_height, image_width)
    background_pixels = Opto_GLM_Utils.get_full_outlines()

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 37)

    for timepoint_index in range(n_timepoints):
        # Create Figure
        figure_1 = plt.figure(figsize=(20, 5))
        regressor_1_axis = figure_1.add_subplot(1, 4, 1)
        regressor_2_axis = figure_1.add_subplot(1, 4, 2)
        difference_axis = figure_1.add_subplot(1, 4, 3)
        sig_diff_axis = figure_1.add_subplot(1, 4, 4)

        # Get Timepoint Data
        regressor_1_timepoint_data = mean_regressor_1[timepoint_index]
        regressor_2_timepoint_data = mean_regressor_2[timepoint_index]
        difference = np.subtract(regressor_1_timepoint_data, regressor_2_timepoint_data)

        timepoint_p_values = p_values[timepoint_index]
        timepoint_p_values = fdr_p_values(timepoint_p_values, indicies, image_height, image_width)
        timepoint_sig = np.where(timepoint_p_values < 0.05, 1, 0)
        sig_difference = np.multiply(difference, timepoint_sig)

        # Convert To Image
        regressor_1_timepoint_data = create_image(regressor_1_timepoint_data, colourmap, atlas_pixels, background_pixels)
        regressor_2_timepoint_data = create_image(regressor_2_timepoint_data, colourmap, atlas_pixels, background_pixels)
        difference = create_image(difference, diff_colourmap, atlas_pixels, background_pixels)
        sig_difference = create_image(sig_difference, diff_colourmap, atlas_pixels, background_pixels)

        # Remove Olfactory Bulb
        regressor_1_timepoint_data = regressor_1_timepoint_data[58:]
        regressor_2_timepoint_data = regressor_2_timepoint_data[58:]
        difference = difference[58:]
        sig_difference = sig_difference[58:]

        # Plot Data
        regressor_1_handle = regressor_1_axis.imshow(regressor_1_timepoint_data, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        regressor_2_handle = regressor_2_axis.imshow(regressor_2_timepoint_data, vmin=-magnitude, vmax=magnitude, cmap=cmap)
        diff_handle = difference_axis.imshow(difference, vmin=-diff_magnitude, vmax=diff_magnitude, cmap=cmap)
        sig_diff_handle = sig_diff_axis.imshow(sig_difference, vmin=-diff_magnitude, vmax=diff_magnitude, cmap=cmap)

        # Remove Axis
        regressor_1_axis.axis('off')
        regressor_2_axis.axis('off')
        difference_axis.axis('off')
        sig_diff_axis.axis('off')

        # Add Colourbar
        colourbar_ticks = np.around(np.linspace(start=-magnitude, stop=magnitude, num=5), 2)
        diff_colourbar_ticks = np.around(np.linspace(start=-diff_magnitude, stop=diff_magnitude, num=5), 2)
        figure_1.colorbar(regressor_1_handle, ax=regressor_1_axis, orientation='vertical', ticks=colourbar_ticks, fraction=0.046, pad=0.04)
        figure_1.colorbar(regressor_2_handle, ax=regressor_2_axis, orientation='vertical', ticks=colourbar_ticks, fraction=0.046, pad=0.04)
        figure_1.colorbar(diff_handle, ax=difference_axis, orientation='vertical', ticks=diff_colourbar_ticks, fraction=0.046, pad=0.04)
        figure_1.colorbar(sig_diff_handle, ax=sig_diff_axis, orientation='vertical', ticks=diff_colourbar_ticks, fraction=0.046, pad=0.04)

        # Set Title
        figure_1.suptitle(str(x_values[timepoint_index]) + "ms")

        # Save
        plt.savefig(os.path.join(save_directory, str(timepoint_index).zfill(4) + ".png"))
        plt.close()



