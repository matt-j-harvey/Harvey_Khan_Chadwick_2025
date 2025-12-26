import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.cm import ScalarMappable
from matplotlib.pyplot import Normalize
from skimage.morphology import binary_dilation
from scipy import ndimage

import Session_List
import GLM_Utils



def reconstruct_regressor(regressor, indicies, image_height, image_width):
    reconstructed_regressor = []
    for timepoint in regressor:
        timepoint_reconstruction = GLM_Utils.create_image_from_data(timepoint, indicies, image_height, image_width)
        reconstructed_regressor.append(timepoint_reconstruction)

    reconstructed_regressor = np.array(reconstructed_regressor)
    return reconstructed_regressor


def visualise_regressor(regressor, start_window, stop_window):

    x_values = list(range(start_window, stop_window))
    plt.ion()
    count = 0

    for value in regressor:
        plt.imshow(value, cmap=GLM_Utils.get_musall_cmap(), vmin=-0.01, vmax=0.01)
        plt.title(x_values[count])
        plt.draw()
        plt.pause(0.1)
        plt.clf()
        count += 1




def extract_model_results(session_list, output_directory, start_window, stop_window):

    for session in tqdm(session_list):

        # Load Coefs
        regression_coefs = np.load(os.path.join(output_directory, session, "Model_Output", "Model_Coefs.npy"))
        regression_coefs = np.transpose(regression_coefs)
        print("regression_coefs", np.shape(regression_coefs))

        # Get Trial Structure
        n_timepoints = stop_window - start_window
        print("n_timepoints", n_timepoints)

        # Extract Stim Regressors
        vis_context_vis_1 = regression_coefs[0*n_timepoints:1*n_timepoints]
        vis_context_vis_2 = regression_coefs[1*n_timepoints:2*n_timepoints]
        odr_context_vis_1 = regression_coefs[2*n_timepoints:3*n_timepoints]
        odr_context_vis_2 = regression_coefs[3*n_timepoints:4*n_timepoints]

        # Save Results
        np.save(os.path.join(output_directory, session, "Model_Output", "vis_context_vis_1_coefs.npy"), vis_context_vis_1)
        np.save(os.path.join(output_directory, session, "Model_Output", "vis_context_vis_2_coefs.npy"), vis_context_vis_2)
        np.save(os.path.join(output_directory, session, "Model_Output", "odr_context_vis_1_coefs.npy"), odr_context_vis_1)
        np.save(os.path.join(output_directory, session, "Model_Output", "odr_context_vis_2_coefs.npy"), odr_context_vis_2)



def get_group_results(output_directory, nested_session_list):

    regressor_list = ["vis_context_vis_1", "vis_context_vis_2", "odr_context_vis_1", "odr_context_vis_2"]

    # Create Save Directory
    save_directory = os.path.join(output_directory, "Group_Regressors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    for regressor in regressor_list:
        group_regressor_list = []

        for mouse in tqdm(nested_session_list):
            mouse_regressor_list = []

            for session in mouse:

                # Load Regressor
                regressor_coefs = np.load(os.path.join(output_directory, session, "Model_Output", regressor + "_coefs.npy"))
                mouse_regressor_list.append(regressor_coefs)

            mouse_mean_coefs = np.mean(np.array(mouse_regressor_list), axis=0)
            group_regressor_list.append(mouse_mean_coefs)

        # Save Group
        np.save(os.path.join(save_directory, regressor + "_group_coefs.npy"), group_regressor_list)


def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load("/home/matthew/Documents/Allen_Atlas_Templates/churchland_outlines_aligned_single.npy")
    atlas_outline = np.roll(atlas_outline, -5, axis=1)
    atlas_outline = binary_dilation(atlas_outline)
    #atlas_outline[3:8, 115:186] = 0
    atlas_pixels = np.nonzero(atlas_outline)


    return atlas_pixels



def view_mean_results(output_directory, start_window, stop_window, frame_period):

    regressor_list = ["vis_context_vis_1", "vis_context_vis_2", "odr_context_vis_1", "odr_context_vis_2"]

    # Load Mask
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()

    # Get Background Pixels
    background_pixels = GLM_Utils.get_background_pixels(indicies, image_height, image_width)

    # Get Atlas Outline Pixels
    atlas_outline_pixels = get_atlas_outline_pixels()

    # Get Colourmap
    cmap = GLM_Utils.get_musall_cmap()

    for regressor in regressor_list:

        # Create Save Directory
        save_directory = os.path.join(output_directory, "Group_Regressors", regressor + "_images")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Load Regressor Data
        group_regressor_data = np.load(os.path.join(output_directory, "Group_Regressors", regressor + "_group_coefs.npy"))

        # get mean Regressor
        mean_regressor = np.mean(group_regressor_data, axis=0)

        # Deconvovle Spikes
        """
        kernel = np.load("/home/matthew/Documents/gcamp6s_kernel.npy", allow_pickle=True)[()]
        kernel = kernel["Kernel"]
        kernel = np.divide(kernel, np.max(np.abs(kernel)))
        mean_regressor = ndimage.convolve1d(mean_regressor, weights=kernel, mode="nearest")
        """

        # Get Regressor magnitude
        regressor_magnitude = np.percentile(np.abs(mean_regressor), q=95)
        colourmap = ScalarMappable(cmap=cmap, norm=(Normalize(vmin=-regressor_magnitude, vmax=regressor_magnitude)))

        # Recosntruct Regressor
        mean_regressor = reconstruct_regressor(mean_regressor, indicies, image_height, image_width)

        x_values = list(range(start_window, stop_window))
        x_values = np.multiply(x_values, frame_period)
        n_timepoints = len(mean_regressor)


        for timepoint_index in range(n_timepoints):

            figure_1 = plt.figure()

            coef_map = mean_regressor[timepoint_index]
            coef_map = colourmap.to_rgba(coef_map)
            coef_map[background_pixels] = (1,1,1,1)
            coef_map[atlas_outline_pixels] = (1,1,1,1)

            axis_1 = figure_1.add_subplot(1,1,1)

            axis_1.imshow(coef_map)

            axis_1.set_title(x_values[timepoint_index])

            axis_1.axis('off')

            figure_1.colorbar(colourmap, ax=axis_1)

            plt.savefig(os.path.join(save_directory, str(timepoint_index).zfill(3) + ".png"))
            plt.close()



"""
frame_period = 36
start_window_ms = -2800
stop_window_ms = 2000
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

# Load Session List
nested_session_list = Session_List.nested_session_list
flat_session_list = Session_List.flatten_nested_list(nested_session_list)

data_root_diretory = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"
output_directory = r"/media/matthew/29D46574463D2856/Paper_Results/Contextual_Swtiching_GLM"

extract_model_results(data_root_diretory, flat_session_list, output_directory, start_window, stop_window)
get_group_results(output_directory, nested_session_list)
view_mean_results(output_directory, start_window, stop_window, frame_period)
"""