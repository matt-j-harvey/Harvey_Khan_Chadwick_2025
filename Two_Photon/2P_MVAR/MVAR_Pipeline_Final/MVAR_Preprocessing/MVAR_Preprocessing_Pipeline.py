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
    For each session this pipeline will create:
    A behaviour matrix - matrix of shape (n_timepoints, n_behavioural_regressors)
    Tensors of neural activity for each trial type
    Tensors of behavioural data for each trial type
    These tensors are then used to fit the MVAR model
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



# Output directory where you want the data to be saved to
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results_Check"

# Directory which contains raw data
data_root  = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"

# List of Sessions to Process
control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

# Model Info
start_window = -17 # How many timepoints before the onset of each stimulus to include
stop_window = 12 # How many timepoints after the onset of each stimulus to include

# Control Switching
for session in tqdm(control_session_list, desc="Session"):
    preprocess_session(data_root, session, mvar_output_root, start_window, stop_window)

