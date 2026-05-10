import os
import numpy as np

from tqdm import tqdm

import Session_List
import Get_RT_Bin_Mean_Running
import Plot_Running_Means_By_RT


def running_speed_rt_pipeline(data_root, session_list, output_directory):

    # Create RT Bins
    bin_time_start = 500
    bin_time_stop = 2250
    n_bins = 7
    bin_width = int((bin_time_stop - bin_time_start) / n_bins)
    bin_start_list = list(range(bin_time_start, bin_time_stop, bin_width))
    bin_stop_list = np.add(bin_start_list, bin_width)
    n_bins = len(bin_start_list)
    print("bin width", bin_width)
    print("bin_start_list", bin_start_list)
    print("bin_stop_list", bin_stop_list)

    # Get Full Tensor Details
    start_window = -41
    stop_window = 68

    # Get Bin Means
    Get_RT_Bin_Mean_Running.get_rt_bin_mean_running(data_root, session_list, n_bins, bin_start_list, bin_stop_list, start_window, stop_window, output_directory)

    # Plot Bins
    Plot_Running_Means_By_RT.plot_running_trace(output_directory, n_bins, bin_start_list, bin_stop_list, start_window, stop_window)






data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
session_list = Session_List.control_post_learning_discrimination
output_directory = r"C:\Neurexin_GLM\Post_Learning\Controls_Running"
running_speed_rt_pipeline(data_root, session_list, output_directory)


data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
session_list = Session_List.neurexin_post_learning_discrimination
output_directory = r"C:\Neurexin_GLM\Post_Learning\Homs_Running"
running_speed_rt_pipeline(data_root, session_list, output_directory)
