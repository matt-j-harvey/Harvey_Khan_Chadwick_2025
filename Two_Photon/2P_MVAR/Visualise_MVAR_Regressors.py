import numpy as np
import matplotlib.pyplot as plt
import os

import MVAR_Utils_2P


def sort_raster(raster, sorting_window_start, sorting_window_stop):

    # Get Mean Response in Sorting Window
    response = raster[sorting_window_start:sorting_window_stop]
    response = np.mean(response, axis=0)

    # Get Sorted Indicies
    sorted_indicies = response.argsort()
    sorted_indicies = np.flip(sorted_indicies)

    # Sort Rasters
    sorted_raster = raster[:, sorted_indicies]

    return sorted_raster



def visualise_regressors(session, output_directory):

    # Load model Dict
    model_dict = np.load(os.path.join(output_directory, session, "Full_Model", "Combined_Model_Dict.npy"), allow_pickle=True)[()]
    Nvar = model_dict["Nvar"]
    N_stim = model_dict["N_stim"]
    Nt = model_dict["Nt"]
    print("model dict", model_dict.keys())
    print("Nvar", Nvar)
    print("N_stim", N_stim)
    print("Nt", Nt)


    # Extract MVAR Parameters
    mvar_parameters = model_dict['MVAR_Parameters']
    print("mvar_parameters", np.shape(mvar_parameters))

    stimuli_weights_list = []
    stim_name_list = ["Visual Context Vis 1", "Visual Context Vis 2", "Odour Context Vis 1", "Odour Context Vis 2", "Odour 1", "Odour 2"]
    for stimulus_index in range(N_stim):
        stim_start = Nvar + (stimulus_index * Nt)
        stim_stop = stim_start + Nt
        stim_weights = mvar_parameters[:, stim_start:stim_stop]
        stimuli_weights_list.append(stim_weights)
        weights_magnitude = np.percentile(np.abs(stim_weights), q=99)
        print("stim_weights", np.shape(stim_weights))
        stim_weights = np.transpose(stim_weights)
        baseline = np.mean(stim_weights[0:3], axis=0)
        stim_weights = np.subtract(stim_weights, baseline)

        stim_weights = sort_raster(stim_weights, 10, sorting_window_stop=15)

        plt.imshow(np.transpose(stim_weights), cmap="bwr", vmin=-weights_magnitude, vmax=weights_magnitude)
        plt.title(stim_name_list[stimulus_index])
        MVAR_Utils_2P.forceAspect(plt.gca())
        plt.show()

    # View Recurrent Weights
    recurrent_weights = mvar_parameters[:, 0:Nvar]
    weight_magnitude = np.percentile(np.abs(recurrent_weights), q=99)
    sorted_matrix = MVAR_Utils_2P.sort_matrix(recurrent_weights)
    plt.imshow(sorted_matrix, cmap="bwr", vmin=-weight_magnitude, vmax=weight_magnitude)
    plt.show()




data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_Odours_Not_Delta"

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
for session in control_session_list:
    visualise_regressors(session, mvar_output_root)