import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from Widefield_Utils import widefield_utils

import Session_List
import Get_RT_Bin_Mean
import Plot_ROI_Means_By_RT





def reaction_time_analysis_pipeline(session_list, output_directory):

    # Create RT Bins
    bin_time_start = 500
    bin_time_stop = 2500
    n_bins = 8
    bin_width = int((bin_time_stop - bin_time_start)/n_bins)
    bin_start_list = list(range(bin_time_start, bin_time_stop, bin_width))
    bin_stop_list = np.add(bin_start_list, bin_width)
    n_bins = len(bin_start_list)
    print("bin width", bin_width)
    print("bin_start_list", bin_start_list)
    print("bin_stop_list", bin_stop_list)

    # Get Full Tensor Details
    start_window = -14
    stop_window = 68

    # Get Stop List In Frames
    bin_stop_list_frames = np.divide(bin_stop_list, 37)
    bin_stop_list_frames = np.ndarray.astype(bin_stop_list_frames, int)
    bin_stop_list_frames = np.add(bin_stop_list_frames, np.abs(start_window))
    print("bin_stop_list_frames", bin_stop_list_frames)

    # Get Activity List
    Get_RT_Bin_Mean.get_rt_bin_means(session_list, n_bins, bin_start_list, bin_stop_list, start_window, stop_window, output_directory)

    # Plot ROI Activity
    atlas_dict = {
        "Primary_Motor": 1,
        "Somatosensory_Barrel": 2,
        "Somatosensory_Limbs": 3,
        "PPC": 5,
        "Secondary_Visual_Medial": 8,
        "Primary_Visual": 9,
        "Retrosplenial": 11,
        "Secondary_Visual_Lateral": 12,
        "Olfactory_bulb": 13,
        "Secondary_Motor_Medial": 14,
        "Secondary_Motor_Anterolateral": 15,
        "Secondary_Motor_Proximal": 16,
    }

    # Load Atlas
    atlas = np.load(r"/home/matthew/Documents/Allen_Atlas_Templates/M2_Three_Segments_Masked_All_Sessions.npy")
    atlas = np.abs(atlas)

    roi_name_list = list(atlas_dict.keys())
    for roi in roi_name_list:
        Plot_ROI_Means_By_RT.plot_roi_trace(output_directory, n_bins, bin_start_list, bin_stop_list, atlas, atlas_dict, roi, bin_stop_list_frames, start_window)


session_list = Session_List.nested_session_list_with_root
output_directory = r"/media/matthew/29D46574463D2856/RT_Analysis"

reaction_time_analysis_pipeline(session_list, output_directory)