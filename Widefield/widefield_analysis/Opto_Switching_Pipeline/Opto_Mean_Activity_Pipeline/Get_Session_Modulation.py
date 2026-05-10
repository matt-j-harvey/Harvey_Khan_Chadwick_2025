import numpy as np
import os
import matplotlib.pyplot as plt

import Opto_GLM_Utils
import Extract_Onsets
import Get_Activity_Tensors

def z_score_activity(data_root, session, activity):

    # Load Means and STD
    pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
    pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))

    # Reconstruct Into 2D
    indicies, image_height, image_width = Opto_GLM_Utils.load_tight_mask()
    pixel_means = Opto_GLM_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Opto_GLM_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    # Z Score Regressor
    activity = np.subtract(activity, pixel_means)
    activity = np.divide(activity, pixel_stds)
    activity = np.nan_to_num(activity)

    return activity


def get_session_modulation(data_root, session, start_window, stop_window, comparison_window_start, comparison_window_stop):

    # Extract Onsets
    [visual_context_control_onsets,
     visual_context_light_onsets,
     odour_context_control_onsets,
     odour_context_light_onsets] = Extract_Onsets.extract_opto_mapping_onsets(data_root, session)

    # Create Activity Tensors
    visual_context_control_tensor = Get_Activity_Tensors.get_activity_tensor_svd(data_root, session, visual_context_control_onsets, start_window, stop_window)
    visual_context_light_tensor = Get_Activity_Tensors.get_activity_tensor_svd(data_root, session, visual_context_light_onsets, start_window, stop_window)
    odour_context_control_tensor = Get_Activity_Tensors.get_activity_tensor_svd(data_root, session, odour_context_control_onsets, start_window, stop_window)
    odour_context_light_tensor = Get_Activity_Tensors.get_activity_tensor_svd(data_root, session, odour_context_light_onsets, start_window, stop_window)

    # Get Means
    visual_context_control_mean = np.mean(visual_context_control_tensor, axis=0)
    visual_context_light_mean = np.mean(visual_context_light_tensor, axis=0)
    odour_context_control_mean = np.mean(odour_context_control_tensor, axis=0)
    odour_context_light_mean = np.mean(odour_context_light_tensor, axis=0)

    # Get Means Across Time
    visual_context_control_mean = np.mean(visual_context_control_mean[comparison_window_start:comparison_window_stop], axis=0)
    visual_context_light_mean = np.mean(visual_context_light_mean[comparison_window_start:comparison_window_stop], axis=0)
    odour_context_control_mean = np.mean(odour_context_control_mean[comparison_window_start:comparison_window_stop], axis=0)
    odour_context_light_mean = np.mean(odour_context_light_mean[comparison_window_start:comparison_window_stop], axis=0)

    # Get Light Effects
    visual_context_light_effect = np.subtract(visual_context_light_mean, visual_context_control_mean)
    odour_context_light_effect = np.subtract(odour_context_light_mean, odour_context_control_mean)

    # Get Attention Light Interaction
    light_modulation = np.subtract(visual_context_light_effect, odour_context_light_effect)
    print("light_modulation", np.shape(light_modulation))

    # Load Spatial Components
    #spatial_components = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Registered_U.npy"))
    spatial_components = np.load(os.path.join(data_root, session, "Local_NMF", "Spatial_Components.npy"))
    print("spatial_components", np.shape(spatial_components))

    # Reconstruct Into Pixel Space
    light_modulation = np.dot(spatial_components, light_modulation)
    print("light_modulation", np.shape(light_modulation))

    # Z Score
    light_modulation = z_score_activity(data_root, session, light_modulation)

    return light_modulation