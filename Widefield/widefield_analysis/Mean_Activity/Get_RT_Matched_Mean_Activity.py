import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import Mean_Activity_Utils

def z_score_trial(regressor, pixel_means, pixel_stds):

    # Reconstruct Into 2D
    indicies, image_height, image_width = Mean_Activity_Utils.load_tight_mask()
    pixel_means = Mean_Activity_Utils.create_image_from_data(pixel_means, indicies, image_height, image_width)
    pixel_stds = Mean_Activity_Utils.create_image_from_data(pixel_stds, indicies, image_height, image_width)

    image_height, image_width, n_components = np.shape(pixel_stds)
    pixel_means = np.reshape(pixel_means, (image_height * image_width))
    pixel_stds = np.reshape(pixel_stds, (image_height * image_width))

    # Z Score Regressor
    regressor = np.subtract(regressor, pixel_means)
    regressor = np.divide(regressor, pixel_stds)
    print("regressor", np.shape(regressor))
    
    return regressor

def reconstruct_nmf_trial(spatial_components, temporal_components):

    # Reshape Spatial Components
    image_height, image_width, n_components = np.shape(spatial_components)
    spatial_components = np.reshape(spatial_components, (image_height*image_width, n_components))

    # Reconstruct Into Pixel Space
    reconstructed_trial = np.matmul(spatial_components, temporal_components)
    reconstructed_trial = np.transpose(reconstructed_trial)

    return reconstructed_trial


def get_mouse_average(mouse_df, data_root):

    mouse_activity = []

    # Cache NMF data so each session is only loaded once
    session_cache = {}

    for row in tqdm(mouse_df.itertuples(index=False)):
        session = row.session
        trial_start = int(row.trial_start_frame)
        trial_stop = int(row.trial_stop_frame)

        # Load NMF data for this session
        if session not in session_cache:
            temporal_components = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Corrected_SVT.npy"))
            spatial_components = np.load(os.path.join(data_root, session, "Preprocessed_Data", "Registered_U.npy"))
            pixel_means = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_means.npy"))
            pixel_stds = np.load(os.path.join(data_root, session, "Z_Scoring", "pixel_stds.npy"))
            session_cache[session] = {"temporal_components": temporal_components, "spatial_components": spatial_components, "pixel_means":pixel_means, "pixel_stds":pixel_stds}
        else:
            temporal_components = session_cache[session]["temporal_components"]
            spatial_components = session_cache[session]["spatial_components"]
            pixel_means = session_cache[session]["pixel_means"]
            pixel_stds = session_cache[session]["pixel_stds"]

        trial_temporal_components = temporal_components[:, trial_start:trial_stop]

        # Reconstruct into pixel space
        trial_activity = reconstruct_nmf_trial(spatial_components, trial_temporal_components)
        trial_activity = z_score_trial(trial_activity, pixel_means, pixel_stds)

        mouse_activity.append(trial_activity)

    if len(mouse_activity) == 0:
        return None
    elif len(mouse_activity) == 1:
        mouse_mean = mouse_activity[0]
    else:
        mouse_activity = np.array(mouse_activity)
        mouse_mean = np.mean(mouse_activity, axis=0)

    return mouse_mean

def get_group_average(group_df, data_root):

    group_average = []
    mice = group_df["mouse"].unique()
    for mouse in mice:

        # Get Mouse DF
        mouse_df = group_df[group_df["mouse"] == mouse]

        # Get Mouse Average
        mouse_average = get_mouse_average(mouse_df, data_root)
        if mouse_average is not None:
            group_average.append(mouse_average)

    group_average = np.array(group_average)
    group_mean = np.mean(group_average, axis=0)
    print("group_mean", np.shape(group_mean))
    return group_mean


def get_rt_matched_mean_activity(matched_df, wt_data_root, nx_data_root, rt_bin_starts, rt_bin_stops, output_root):

    wt_rt_bin_means = []
    nx_rt_bin_means = []

    for bin_index in range(len(rt_bin_starts)):
        bin_start = rt_bin_starts[bin_index]
        bin_stop = rt_bin_stops[bin_index]

        # Get the DF for this RT Bin
        bin_df = matched_df[(matched_df["reaction_time"] >= bin_start) & (matched_df["reaction_time"] < bin_stop)]

        # Split Dataframe By Genotype
        wt_df = bin_df[bin_df["group"] == 0]
        nx_df = bin_df[bin_df["group"] == 1]

        # Get Mean Activity For Each Genotype
        wt_mean = get_group_average(wt_df, wt_data_root)
        nx_mean = get_group_average(nx_df, nx_data_root)

        # Add To List
        wt_rt_bin_means.append(wt_mean)
        nx_rt_bin_means.append(nx_mean)

    # Save Data
    wt_rt_bin_means = np.array(wt_rt_bin_means)
    nx_rt_bin_means = np.array(nx_rt_bin_means)

    np.save(os.path.join(output_root, "wt_rt_bin_means"), wt_rt_bin_means)
    np.save(os.path.join(output_root, "nx_rt_bin_means"), nx_rt_bin_means)

    return wt_rt_bin_means, nx_rt_bin_means

