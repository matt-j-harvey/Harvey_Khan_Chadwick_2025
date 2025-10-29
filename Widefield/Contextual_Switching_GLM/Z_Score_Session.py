import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm
from Widefield_Utils import widefield_utils


def remove_early_frames(frame_list, cutoff=3000):
    kept_frames = []
    for frame in frame_list:
        if frame > cutoff:
            kept_frames.append(frame)
    return kept_frames


def z_score_session(session, opto):

    # Load Corrected SVT
    corrected_svt = np.load(os.path.join(session, "Preprocessed_Data", "Corrected_SVT.npy"))
    registered_u = np.load(os.path.join(session, "Preprocessed_Data", "Registered_U.npy"))
    print("corrected_svt", np.shape(corrected_svt))
    print("registered_u", np.shape(registered_u))

    # Load Baseline Frames
    if opto == True:
        baseline_frames = np.load(os.path.join(session, "Stimuli_Onsets", "Frames_Outside_Opto_Window.npy"))
        baseline_frames = remove_early_frames(baseline_frames)
        corrected_svt = corrected_svt[:, baseline_frames]

    else:
        corrected_svt = corrected_svt[:, 3000:]

    # Flatten Reg U
    image_height, image_width, n_components = np.shape(registered_u)
    registered_u = np.reshape(registered_u, (image_height * image_width, n_components))
    indicies, image_height, image_width = widefield_utils.load_tight_mask()
    registered_u = registered_u[indicies]

    # Reconstruct Data
    reconstructed_data = np.dot(registered_u, corrected_svt)
    print("reconstructed_data", np.shape(reconstructed_data))

    # Get Mean and STD
    data_mean = np.mean(reconstructed_data, axis=1)
    data_std = np.std(reconstructed_data, axis=1)

    # Save Data
    save_directory = os.path.join(session, "Z_Scoring")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "pixel_means.npy"), data_mean)
    np.save(os.path.join(save_directory, "pixel_stds.npy"), data_std)


    # Visualise These
    colourmap = widefield_utils.get_musall_cmap()

    data_mean = widefield_utils.create_image_from_data(data_mean, indicies, image_height, image_width)
    plt.imshow(data_mean, cmap=colourmap, vmin=-0.05, vmax=0.05)
    plt.savefig(os.path.join(save_directory, "pixel_means.png"))
    plt.close()

    data_std = widefield_utils.create_image_from_data(data_std, indicies, image_height, image_width)
    plt.imshow(data_std, cmap=colourmap, vmin=-0.05, vmax=0.05)
    plt.savefig(os.path.join(save_directory, "pixel_STD.png"))
    plt.close()


opto_session_list = [

     "KPGC11.1C/2024_08_22_Switching_V1_Pre_03",
     "KPGC11.1C/2024_08_23_Switching_PPC_Pre_03",
     "KPGC11.1C/2024_08_26_Switching_ProxM_Pre_03",
     "KPGC11.1C/2024_08_28_Switching_PM_Pre_03",
     "KPGC11.1C/2024_08_30_Switching_MM_Pre_03",
     "KPGC11.1C/2024_09_03_Switching_RSC_Pre_03",
     "KPGC11.1C/2024_09_11_Switching_ALM_Pre_03",
     "KPGC11.1C/2024_09_17_Switching_SS_Pre_03",
     "KPGC11.1C/2024_09_20_Switching_V1_Pre_03",

     "KPGC12.2A/2024_09_11_Switching_MM_Pre_03",
     "KPGC12.2A/2024_09_16_Switching_V1_Pre_03",
     "KPGC12.2A/2024_09_17_Switching_PPC_Pre_03",
     "KPGC12.2A/2024_09_18_Switching_MM_Pre_03",
     "KPGC12.2A/2024_09_19_Switching_SS_Pre_03",
     "KPGC12.2A/2024_09_20_Switching_PM_Pre_03",
     "KPGC12.2A/2024_09_25_Switching_RSC_Pre_03",
     "KPGC12.2A/2024_09_28_Switching_ALM_Pre_03",
     "KPGC12.2A/2024_09_29_Switching_ProxM_Pre_03",

     "KPGC12.2B/2024_09_10_Switching_V1_Pre_03",
     "KPGC12.2B/2024_09_11_Switching_MM_Pre_03",
     "KPGC12.2B/2024_09_17_Switching_RSC_Pre_03",
     "KPGC12.2B/2024_09_18_Switching_MM_Pre_03",
     "KPGC12.2B/2024_09_19_Switching_SS_Pre_03",
     "KPGC12.2B/2024_09_20_Switching_PPC_Pre_03",
     "KPGC12.2B/2024_09_24_Switching_PM_Pre_03",
     "KPGC12.2B/2024_09_26_Switching_ALM_Pre_03",
     "KPGC12.2B/2024_09_28_Switching_ProxM_Pre_03",

     #"KPGC3.3E/2023_07_03_Switch_V1_1F_03_Pre",
     #"KPGC3.3E/2023_07_06_Switch_MM_1F_03_Pre",
     #"KPGC3.3E/2023_07_14_Switch_ALM_1F_03_Pre",
     #"KPGC3.3E/2023_07_19_Switch_RSC_1F_03_Pre",
     "KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_28_Switch_V1_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
     "KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
     "KPGC3.3E/2023_08_07_Switch_SS_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre",

     #"KPGC6.2E/2023_07_27_Switch_MM_1F_04_1S_Pre",
     "KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_18_Switch_ALM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_21_Switch_BC_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre",

]





control_session_list = [

    "KPGC12.3B/2024_09_03_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_05_Switching_PPC_Pre_03",
     "KPGC12.3B/2024_09_09_Switching_MM_Pre_03",
     "KPGC12.3B/2024_09_10_Switching_RSC_Pre_03",
     "KPGC12.3B/2024_09_12_Switching_SS_Pre_03",
     "KPGC12.3B/2024_09_16_Switching_ALM_Pre_03",
     "KPGC12.3B/2024_09_17_Switching_PM_Pre_03",
     "KPGC12.3B/2024_09_18_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_19_Switching_Pre_03",
     "KPGC12.3B/2024_09_20_Switching_MM_Pre_03",
     "KPGC12.3B/2024_09_23_Switching_SS_Pre_03",
     "KPGC12.3B/2024_09_28_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_29_Switching_ProxM_Pre_03",

    #"KPGC1.3A/2023_07_07_Switch_V1_1F_03_Pre",
     #"KPGC1.3A/2023_07_13_Switch_MM_1F_03_Pre",
     #"KPGC1.3A/2023_07_18_Switch_ALM_1F_03_Pre",
     #"KPGC1.3A/2023_07_20_Switch_RSC_1F_03_Pre",
     "KPGC1.3A/2023_07_27_Switch_MM_1f_04_1S_Pre",
     "KPGC1.3A/2023_08_01_Switch_V1_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_03_Switch_RSC_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_08_Switch_PM_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_10_Switch_ALM_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_22_Switch_V1_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_24_Switch_PPC_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_28_Switch_BC_1F_04_1S_Pre",

    "KPGC3.4A/2023_08_24_Switch_V1_1F_04_1S_Pre",
     "KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
     "KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",

    "KPGC7.4A/2023_08_25_Switch_V1_1F_04_1S_Pre",
     "KPGC7.4A/2023_08_28_Switch_ALM_1F_04_1S_Pre",
     "KPGC7.4A/2023_08_30_Switch_PM_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_01_Switch_V1_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_04_Switch_MM_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_06_Switch_BC_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_08_Switch_RSC_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
]


data_root_directory = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"




"""
for session in tqdm(opto_session_list):
    base_directory = os.path.join(data_root_directory, session)
    z_score_session(base_directory, opto=True)
"""

for session in tqdm(control_session_list):
    base_directory = os.path.join(data_root_directory, session)
    z_score_session(base_directory, opto=True)