import os
import numpy as np
from tqdm import tqdm
import sys

import Downsample_AI_Matrix_Framewise_2P
import Extract_Onsets
import Create_Activity_Tensor
import Create_Behaviour_Matrix
import Create_Behaviour_Tensor

def preprocess_session(data_root_directory, session, mvar_output_directory, start_window, stop_window):

    """
    For Each Session Creates A Design Matrix Dict
    This is used to Fit the MVAR
    """

    # Downsample AI Matrix
    Downsample_AI_Matrix_Framewise_2P.downsample_ai_matrix(data_root_directory, session, mvar_output_directory)

    # Extract Onsets
    Extract_Onsets.extract_odour_onsets(os.path.join(data_root_directory, session))

    # Create Behaviour Matrix
    Create_Behaviour_Matrix.create_behaviour_matrix(data_root_directory, session, mvar_output_directory)

    # Create Activity Tensors
    Create_Activity_Tensor.create_activity_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
    Create_Activity_Tensor.create_activity_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
    Create_Activity_Tensor.create_activity_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
    Create_Activity_Tensor.create_activity_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)
    Create_Activity_Tensor.create_activity_tensor(data_root_directory, session, mvar_output_directory, "Odour_1_onset_frames.npy", start_window, stop_window)
    Create_Activity_Tensor.create_activity_tensor(data_root_directory, session, mvar_output_directory, "Odour_2_onset_frames.npy", start_window, stop_window)

    # Create Behaviour Tensors
    Create_Behaviour_Tensor.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
    Create_Behaviour_Tensor.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
    Create_Behaviour_Tensor.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
    Create_Behaviour_Tensor.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)
    Create_Behaviour_Tensor.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "Odour_1_onset_frames.npy", start_window, stop_window)
    Create_Behaviour_Tensor.create_behaviour_tensor(data_root_directory, session, mvar_output_directory, "Odour_2_onset_frames.npy", start_window, stop_window)


# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Final_No_Z"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

# Model Info
start_window = -17
stop_window = 12

# Control Switching
for session in tqdm(control_session_list, desc="Session"):
    preprocess_session(data_root, session, mvar_output_root, start_window, stop_window)

