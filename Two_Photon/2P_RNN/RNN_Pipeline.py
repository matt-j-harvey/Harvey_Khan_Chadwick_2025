import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

import Downsample_AI_Matrix_Framewise_2P
import Create_Behaviour_Matrix
import Create_Regression_Matricies
import MVAR_Utils_2P
import RNN_Cross_Fold_Validation


def rnn_pipeline(data_root_directory, session_list, output_directory, start_window, stop_window):

    for session in tqdm(session_list, position=0, desc="Session:"):

        # Downsample AI Matrix
        Downsample_AI_Matrix_Framewise_2P.downsample_ai_matrix(data_root_directory, session, output_directory)

        # Create Behaviour Matrix
        Create_Behaviour_Matrix.create_behaviour_matrix(data_root_directory, session, output_directory)

        # Create Activity Tensors
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, output_directory, "Odour_1_onset_frames.npy", start_window, stop_window)
        MVAR_Utils_2P.create_activity_tensor(data_root_directory, session, output_directory, "Odour_2_onset_frames.npy", start_window, stop_window)

        # Create Behaviour Tensors
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, output_directory, "visual_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, output_directory, "visual_context_stable_vis_2_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, output_directory, "odour_context_stable_vis_1_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, output_directory, "odour_context_stable_vis_2_onsets.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, output_directory, "Odour_1_onset_frames.npy", start_window, stop_window)
        MVAR_Utils_2P.create_behaviour_tensor(data_root_directory, session, output_directory, "Odour_2_onset_frames.npy", start_window, stop_window)

        # Create Regression Matricies
        Create_Regression_Matricies.create_regression_matricies_shared_recurrent_weights_odour(data_root_directory,
                                                                                         session,
                                                                                         output_directory,
                                                                                         start_window,
                                                                                         stop_window)

        # Load Data
        design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = MVAR_Utils_2P.load_regression_matrix(session, output_directory, "combined")
        delta_f_matrix = np.transpose(delta_f_matrix)

        # Create Weight Directory
        weight_directory = os.path.join(output_directory, session, "RNN_Weights")
        if not os.path.exists(weight_directory):
            os.makedirs(weight_directory)

        # Define Parameters
        #n_model_units = np.shape(delta_f_matrix)[1]
        n_model_units = 250
        device = torch.device('cpu')

        # Perform K Fold Cross Validation
        mean_r2 = RNN_Cross_Fold_Validation.perform_k_fold_validation(n_model_units, device, design_matrix, delta_f_matrix, weight_directory, n_folds=5)
        print("mean_r2", mean_r2)


# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_RNN_Results"

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
frame_rate = 6.37

# Control Switching
rnn_pipeline(data_root, control_session_list, output_root, start_window, stop_window)
