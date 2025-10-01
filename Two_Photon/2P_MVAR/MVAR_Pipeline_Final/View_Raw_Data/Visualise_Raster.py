import numpy as np
import os
import matplotlib.pyplot as plt

import Plotting_Functions

def visualise_raster(data_root, session, mvar_output_root):

    # Load dF/F
    df_matrix = np.load(os.path.join(mvar_output_root, session, "df_over_f_matrix.npy"))
    n_timepoints, n_neurons = np.shape(df_matrix)

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    n_seconds = n_timepoints / frame_rate
    n_mins = n_seconds / 60.0

    # Get Magnitudes
    vmin = np.percentile(df_matrix, q=10)
    vmax = np.percentile(df_matrix, q=99)

    # Plot PSTH Sorted by Lick Coding Dimension
    lick_cd = np.load(os.path.join(mvar_output_root, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))
    lick_cd_indicies = lick_cd.argsort()
    lick_cd_indicies = np.flip(lick_cd_indicies)

    # Sort By Lick CD
    df_matrix = df_matrix[:, lick_cd_indicies]

    figure_1 = plt.figure(figsize=(18,6))
    axis_1 = figure_1.add_subplot(1,1,1)
    image_handle = axis_1.imshow(np.transpose(df_matrix), vmin=vmin, vmax=vmax, extent=[0,n_mins,0,n_neurons])
    axis_1.set_xlabel("Time (Minutes)")
    axis_1.set_ylabel("Neurons")

    # Add Colourbar
    #plt.colorbar(image_handle, ax=axis_1)
    cbar = axis_1.figure.colorbar(image_handle, ax=axis_1, fraction=0.016, pad=0.02)
    cbar.ax.set_ylabel("dF/F", rotation=-90, va="bottom")

    Plotting_Functions.forceAspect(axis_1, aspect=3)

    # Save Figure
    save_directory = os.path.join(mvar_output_root, session, "Raw Data Visualisation")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    plt.savefig(os.path.join(save_directory, "Raster.png"))
    #plt.show()
    plt.close()