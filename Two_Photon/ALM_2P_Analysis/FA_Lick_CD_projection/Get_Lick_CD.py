import numpy as np
import os

import Get_Data_Tensor
import Fa_Lick_CD_Utils

"0 trial_index,"  # 0
"1 trial_type,"  # 1
"2 lick,"  # 2
"3 correct,"  # 3
"4 rewarded,"  # 4
"5 preeceded_by_irrel,"  # 5
"6 irrel_type,"  # 6
"7 ignore_irrel,"  # 7
"8 block_number,"  # 8
"9 first_in_block,"  # 9
"10 in_block_of_stable_performance,"  # 10
"11 onset,"  # 11
"12 stimuli_offset,"  # 12
"13 irrel_onset,"  # 13
"14 irrel_offset,"  # 14
"15 trial_end,"  # 15
"16 photodiode_onset,"  # 16
"17 photodiode_offset,"  # 17
"18 onset_closest_frame,"  # 18
"19 offset_closest_frame,"  # 19
"20 irrel_onset_closest_frame,"  # 20
"21 irrel_offset_closest_frame,"  # 21
"22 lick_onset,"  # 22
"23 reaction_time,"  # 23
"24 reward_onset,"  # 24



def get_lick_cd(base_directory, df_matrix):

    # Load Behaviour Matrix
    behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

    # Get Hit Lick Onsets
    vis_hit_onsets = Fa_Lick_CD_Utils.get_vis_hit_lick_onsets(behaviour_matrix)
    odr_hit_onsets = Fa_Lick_CD_Utils.get_odour_hit_lick_onsets(behaviour_matrix)

    # Load Stack Onsets
    stack_onsets = np.load(os.path.join(base_directory, "Behaviour", "Stack_Onsets.npy"))

    # Convert To Frames
    vis_hit_onsets = Fa_Lick_CD_Utils.get_nearest_frames_to_onsets(vis_hit_onsets, stack_onsets)
    odr_hit_onsets = Fa_Lick_CD_Utils.get_nearest_frames_to_onsets(odr_hit_onsets, stack_onsets)

    # Get Activity Tensors
    start_window = -12
    stop_window = 0
    vis_hit_tensor = Get_Data_Tensor.get_activity_tensors(df_matrix, vis_hit_onsets, start_window, stop_window, True, 6)
    odr_hit_tensor = Get_Data_Tensor.get_activity_tensors(df_matrix, odr_hit_onsets, start_window, stop_window, True, 6)

    # Take Across Trials
    vis_hit_mean = np.mean(vis_hit_tensor, axis=0)
    odour_hit_mean = np.mean(odr_hit_tensor, axis=0)
    grand_mean = np.stack([vis_hit_mean, odour_hit_mean])
    grand_mean = np.mean(grand_mean, axis=0)

    # Get 1S Prior to Lick
    grand_mean = grand_mean[-6:]

    # Get Average Difference Across Time
    wt_time_average = np.mean(grand_mean, axis=0)

    # Normalise Vector - This ensures the coding dimension vector has a length of 1, making it comparable across different mice
    norm = np.linalg.norm(wt_time_average)
    coding_dimension = np.divide(wt_time_average, norm)

    return coding_dimension

