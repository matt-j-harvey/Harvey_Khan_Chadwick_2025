import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm


def plot_stim_weights(model_dict):
    pass





def visualise_mvar_results(mvar_directory, session_list):

    # General Preprocessing
    for session in tqdm(session_list, position=0, desc="Session:"):

        # Load Model Dict
        model_dict = np.load(os.path.join(mvar_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
        print("model_dict", model_dict.keys())

        # Unpack Dict
        Nt = model_dict["Nt"]
        model_params = model_dict["MVAR_Parameters"]
        n_neurons = np.shape(model_params)[0]

        # Load Regression Matrix
        regression_matrix = np.load(os.path.join(mvar_directory, session, "Design_Matricies", "Combined_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]
        N_trials = regression_matrix['N_trials']
        print("N_trials", N_trials)
        print("regression_matrix", regression_matrix.keys())




# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Final"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

visualise_mvar_results(mvar_output_root, control_session_list)