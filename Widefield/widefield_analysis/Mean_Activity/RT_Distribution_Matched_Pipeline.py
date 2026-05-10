import numpy as np
import os

import Session_List
import Create_Combined_Dataframe
import Get_Matched_RT_Distribution
import Get_RT_Matched_Mean_Activity
import Get_RT_Bin_Mean_Activity

# Get RT Distribution for each mouse
# Get RT Distribution for whole genotype
# Select Trials To Match RT Distributions



# Select Analysis Details
frame_period = 36
start_window_ms = -1500
stop_window_ms = 2500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

rt_bin_starts = list(range(500, 2250, 250))
rt_bin_stops = np.add(rt_bin_starts, 250)

wt_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
nx_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"

wt_session_list = Session_List.control_pre_learning
nx_session_list = Session_List.neurexin_pre_learning_list
output_root = r"C:\Learning_Mean_Activity\Pre_Learning_RT_Matched"



Get_RT_Bin_Mean_Activity.get_rt_bin_means(wt_data_root,
                                         nx_data_root,
                                         wt_session_list,
                                         nx_session_list,
                                         rt_bin_starts,
                                         rt_bin_stops,
                                         1,
                                         start_window,
                                         stop_window,
                                         output_root)

Get_RT_Bin_Mean_Activity.get_rt_bin_means(wt_data_root,
                                         nx_data_root,
                                         wt_session_list,
                                         nx_session_list,
                                         rt_bin_starts,
                                         rt_bin_stops,
                                         2,
                                         start_window,
                                         stop_window,
                                         output_root)

"""
# Create Combined Dataframe
combined_dataframe = Create_Combined_Dataframe.create_combined_dataframe(wt_session_list, nx_session_list, wt_data_root, nx_data_root, 1, start_window, stop_window)

# Get RT Matched Dataframe
matched_df, summary_df = Get_Matched_RT_Distribution.match_rt_distributions_across_groups(combined_dataframe, rt_bin_starts, rt_bin_stops)

# Get Mean Activity
Get_RT_Matched_Mean_Activity.get_rt_matched_mean_activity(matched_df, wt_data_root, nx_data_root, rt_bin_starts, rt_bin_stops, output_root)

# View Mean Activity
"""