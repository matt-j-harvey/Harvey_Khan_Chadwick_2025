import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

import Session_List


def blue_black_red_perceptual(n_colors=256, gamma=1.8):
    """
    Perceptually improved blue → black → red colormap.

    Key properties:
    - Symmetric around zero
    - Smooth luminance change (controlled by gamma)
    - Reduced harsh contrast at black
    - Better for signed neural quantities

    Parameters
    ----------
    n_colors : int
        Number of colour levels
    gamma : float
        Controls how smoothly colours approach black
        (higher = flatter near zero, often nicer for data)

    Returns
    -------
    cmap : matplotlib.colors.Colormap
    """
    half = n_colors // 2

    # Parameter from centre (0) to extremes (1)
    t = np.linspace(0, 1, half)

    # Smooth luminance ramp
    t = t ** gamma

    # Negative side: blue → black
    neg = np.stack([
        np.zeros_like(t),     # R
        np.zeros_like(t),     # G
        t[::-1]               # B (bright → dark)
    ], axis=1)

    # Positive side: black → red
    pos = np.stack([
        t,                    # R (dark → bright)
        np.zeros_like(t),     # G
        np.zeros_like(t)      # B
    ], axis=1)

    colors = np.vstack([neg, pos])

    return ListedColormap(colors, name="blue_black_red_perceptual")

def calcium_blue_black_red_yellow_cmap(name="calcium_bbr_y", n_colors=256):
    """
    Recreate a widefield/calcium-style dF/F colormap:
    cyan → blue → black → red → yellow

    Intended for signed data where black is zero.
    """

    colors = [
        (0.00, "#00ffff"),  # cyan: strong negative
        (0.25, "#0000ff"),  # blue: negative
        (0.50, "#000000"),  # black: zero
        (0.75, "#ff0000"),  # red: positive
        (1.00, "#ffff00"),  # yellow: strong positive
    ]

    return LinearSegmentedColormap.from_list(name, colors, N=n_colors)


def view_matrix(matrix, start_window, stop_window, cmap=None, range=None):
    start_window_ms = start_window * 37
    stop_window_ms = stop_window * 37

    if cmap == None:
        cmap = "viridis"

    if range is None:
        vmin=0.5
        vmax = 1

    else:
        vmin = range[0]
        vmax = range[1]

    plt.imshow(matrix, vmin=vmin, vmax=vmax, extent=[start_window_ms, stop_window_ms, stop_window_ms, start_window_ms], cmap=cmap)
    plt.colorbar(label="Decoding accuracy")
    plt.xlabel("Timepoint 2")
    plt.ylabel("Timepoint 1")
    plt.show()

def get_group_results(session_list, output_root):

    group_results = []
    for mouse in session_list:
        for session in mouse:
            session_results = np.load(os.path.join(output_root, session, "Confusion_Matrix.npy"))
            #view_matrix(session_results, start_window, stop_window)
            group_results.append(session_results)
    group_results = np.array(group_results)
    return group_results


# Set Directories
#control_session_list = Session_List.control_post_learning_discrimination
#control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
#control_output_root = r"C:\Cr_Face_Motion\Controls"

#hom_session_list = Session_List.neurexin_post_learning_discrimination
#hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
#hom_output_root = r"C:\Cr_Face_Motion\Homs"

"""
# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)
"""


control_session_list = Session_List.control_intermediate_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Cr_Face_Motion\Int\Controls"

hom_session_list = Session_List.neurexin_intermediate_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Cr_Face_Motion\Int\Homs"


# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


# Run Pipeline
control_results = get_group_results(control_session_list, control_output_root)
control_mean = np.mean(control_results, axis=0)
view_matrix(control_mean, start_window, stop_window)

neurexin_results = get_group_results(hom_session_list, hom_output_root)
neurexin_mean = np.mean(neurexin_results, axis=0)
view_matrix(neurexin_mean, start_window, stop_window)

diff_matrix = np.subtract(neurexin_mean, control_mean)
magnitude = 0.25
cmap = calcium_blue_black_red_yellow_cmap()
view_matrix(diff_matrix, start_window, stop_window, cmap=cmap, range=[-magnitude, magnitude])


