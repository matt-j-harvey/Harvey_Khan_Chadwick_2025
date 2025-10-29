import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

from matplotlib.pyplot import GridSpec

import Visualise_Results_Utils
import Extract_Model_Predictions
import MVAR_Output_Plotting_Functions
import Partition_Contributions







"""
def visualise_group_results():

    # Visualise Group Results?
    stimulus_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
        "odour_1",
        "odour_2"]

    # Plot Average Lick CDs

    # Create Mean Lick CD Save Directory
    group_lick_cd_save_directory = os.path.join(mvar_directory, "Group_Results", "Stimulus_Weights_Lick_CD_Projections")
    if not os.path.exists(group_lick_cd_save_directory):
        os.makedirs(group_lick_cd_save_directory)

    for stimulus in stimulus_list:
        stimulus_projection_list = []

        for session in tqdm(session_list, position=0, desc="Session:"):
            session_lick_cd_projection = np.load(os.path.join(mvar_directory, session, "MVAR_Results", "Lick_CD_Projections", stimulus + "_lick_cd_projection.npy"))
            stimulus_projection_list.append(session_lick_cd_projection)

        stimulus_projection_list = np.array(stimulus_projection_list)
        mean_projection = np.mean(stimulus_projection_list, axis=0)
        plt.plot(mean_projection)
        plt.savefig(os.path.join(group_lick_cd_save_directory, stimulus + "Lick_CD_Projection.png"))
        plt.close()
"""



def visualise_mvar_results_session(data_root_directory, mvar_directory, session_list, model_type):

    stimulus_list = [
        "visual_context_vis_1",
        "visual_context_vis_2",
        "odour_context_vis_1",
        "odour_context_vis_2",
        "odour_1",
        "odour_2"]

    # Plot Individual Sessions
    for session in session_list:

        # Load Design Matrix
        design_matrix, delta_f_matrix, Nvar, Nbehav, Nt, Nstim, Ntrials, timewindow = Visualise_Results_Utils.load_design_matrix(session, mvar_directory, model_type)

        # Load Frame Rate
        frame_rate = np.load(os.path.join(data_root_directory, session, "Frame_Rate.npy"))
        x_values = Visualise_Results_Utils.get_time_x_values(timewindow, frame_rate)

        # Load Model Dictionary
        model_dict = np.load(os.path.join(mvar_directory, session, "Full_Model", model_type + "_Model_Dict.npy"), allow_pickle=True)[()]
        print("model dict", model_dict.keys())

        # Create Prediction Save Directory
        save_directory = os.path.join(mvar_directory, session, "MVAR_Results", model_type)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Extract Model Predictions
        Extract_Model_Predictions.extract_model_predictions(mvar_directory, session, design_matrix, delta_f_matrix, Nvar, Nt, Nstim, Ntrials, model_dict, stimulus_list, save_directory)

        # Plot Full Raster
        MVAR_Output_Plotting_Functions.plot_full_raster(delta_f_matrix, save_directory)

        # Plot Stim Predictions
        MVAR_Output_Plotting_Functions.plot_stim_prediction(stimulus_list, save_directory, x_values)

        # Plot Lick CD Projections
        MVAR_Output_Plotting_Functions.plot_lick_cd_projections(save_directory, stimulus_list, x_values)

        # Plot Recurrent Weights
        MVAR_Output_Plotting_Functions.plot_recurrent_weights(model_dict, save_directory)

        # Plot Stim Weights
        MVAR_Output_Plotting_Functions.plot_stim_weights(model_dict, frame_rate, save_directory)

        # Partition Contributions
        Partition_Contributions.partition_model_contributions(mvar_directory, session, design_matrix, Nvar, Nt, Nstim, Ntrials, model_dict, save_directory)

        # Plot Partitioned Contribution
        MVAR_Output_Plotting_Functions.plot_partitioned_contributions(save_directory, stimulus_list, frame_rate)
        MVAR_Output_Plotting_Functions.plot_partitioned_lick_cds(save_directory, stimulus_list, frame_rate)

    # Plot Group Results
    MVAR_Output_Plotting_Functions.plot_group_lick_cd_projections(mvar_directory, session_list, model_type, stimulus_list[0:4], x_values)
    MVAR_Output_Plotting_Functions.plot_group_partitioned_lick_cd_projections(mvar_directory, session_list, model_type, stimulus_list[0:4], x_values)




# Output directory where you want the data to be saved to
#mvar_output_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR"

# Directory which contains raw data
#data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]




data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"
mvar_output_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR\Homs"

session_list = [
    r"64.1B\2024_09_09_Switching",
    r"70.1A\2024_09_09_Switching",
    r"70.1B\2024_09_12_Switching",
    r"72.1E\2024_08_23_Switching",
]



visualise_mvar_results_session(data_root, mvar_output_root, session_list, "Standard")



