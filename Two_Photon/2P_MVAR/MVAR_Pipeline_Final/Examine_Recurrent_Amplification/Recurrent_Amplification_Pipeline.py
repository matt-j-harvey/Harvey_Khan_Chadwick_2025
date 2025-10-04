import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

import Get_Stimuli_Weights
import Get_Recurrent_Weights
import Get_Stimuli_Recurrent_Interactions
import Plot_Stimuli_Weight_Interactions


def recurrent_amplification_pipeline(data_root_directory, mvar_directory, session_list):

    # Plot Individual Sessions
    for session in session_list:

        # Load Model Dictionary
        model_dict = np.load(os.path.join(mvar_directory, session, "Full_Model",  "Standard_Model_Dict.npy"), allow_pickle=True)[()]
        print("model dict", model_dict.keys())

        # Get Output Directory
        output_directory = os.path.join(mvar_directory, session, "Recurrent Amplification")

        # Get Stimuli Weights
        Get_Stimuli_Weights.get_stimuli_weights(model_dict, output_directory)

        # Get Recurrent Weights and Variants
        Get_Recurrent_Weights.get_recurrent_weights(model_dict, output_directory)

        # Get Interactions - Diagonal and Recurrent
        Get_Stimuli_Recurrent_Interactions.get_stimuli_recurrent_interactions(mvar_directory, session, output_directory, "recurrent_weights")
        Get_Stimuli_Recurrent_Interactions.get_stimuli_recurrent_interactions(mvar_directory, session, output_directory, "diagonal_weights")

        # Plot Recurrent interactions
        Plot_Stimuli_Weight_Interactions.plot_session_interactions(output_directory)

    # Plot Group Results
    Plot_Stimuli_Weight_Interactions.plot_group_interactions(mvar_directory, session_list)
    Plot_Stimuli_Weight_Interactions.plot_scatters(mvar_directory, session_list)

    Plot_Stimuli_Weight_Interactions.plot_total_interactions(mvar_directory, session_list)
    Plot_Stimuli_Weight_Interactions.compare_modulation_interaction(mvar_directory, session_list)


# Output directory where you want the data to be saved to
mvar_output_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR"

# Directory which contains raw data
data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


recurrent_amplification_pipeline(data_root, mvar_output_root, control_session_list)
