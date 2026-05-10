import os

"""
number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import canny
from skimage.morphology import binary_dilation
from matplotlib.gridspec import GridSpec
from skimage.transform import downscale_local_mean, resize
from scipy import ndimage
import shutil
from tqdm import tqdm

import run_local_nmf
from Files import Session_List
from Widefield_Utils import widefield_utils


def load_mask():

    mask_dict = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Tight_Mask_No_Olfactory_Dict.npy", allow_pickle=True)[()]
    print("mask_dict", mask_dict)
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]
    template = np.zeros(image_height * image_width)
    template[indicies] = 1
    template = np.reshape(template, (image_height, image_width))
    template = np.ndarray.astype(template, bool)

    return template






def view_aligned_u(u, atlas):

    atlas_edges = canny(atlas)
    edge_indicies = np.nonzero(atlas_edges)

    image_height, image_width, number_of_components = np.shape(u)
    for component_index in range(number_of_components):

        component = u[:, :, component_index]
        component[edge_indicies] = np.max(component)
        plt.imshow(component)
        plt.show()


def view_components(components, nmf_directory):

    num_components = np.shape(components)[2]

    """
    # View Combined Mapping
    x, y = widefield_utils.get_best_grid(num_components)
    figure_1 = plt.figure()
    for component_index in range(num_components):
        axis_1 = figure_1.add_subplot(x, y, component_index + 1)
        axis_1.imshow(components[:, :, component_index])
        axis_1.axis('off')
    plt.show()
    """

    # Plot Individual Comps
    save_directory = os.path.join(nmf_directory, "Spatial_Components")
    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)
    os.mkdir(save_directory)

    for component_index in range(num_components):
        plt.imshow(components[:, :, component_index])
        plt.savefig(os.path.join(save_directory, str(component_index).zfill(3) + ".png"))
        plt.close()



def reconstruct_data(spatial_components, temporal_components, sample_size=10000):

    print("Spatial components", np.shape(spatial_components))
    print("Temporal components", np.shape(temporal_components))
    data = np.dot(spatial_components, temporal_components[:, 0:sample_size])

    print("Data Shape", np.shape(data))

    colourmap = widefield_utils.get_musall_cmap()

    plt.ion()
    for frame_index in range(sample_size):
        plt.imshow(data[:, :, frame_index], cmap=colourmap, vmin=-0.05, vmax=0.05)
        plt.draw()
        plt.pause(0.1)
        plt.clf()




def run_local_nmf_pipeline(base_directory, early_cutoff=3000):

    # Load Mask
    mask = load_mask()
    print("Mask Shape", np.shape(mask))
    mask.astype(dtype=bool)
    print("mask dtype", mask.dtype)

    # Load Atlas
    atlas = np.load("/home/matthew/Documents/Allen_Atlas_Templates/Local_NMF_Atlas_No_Olfactory.npy")
    print("Atlas Shape", np.shape(atlas))


    # Load Data
    u = np.load(os.path.join(base_directory, "Preprocessed_Data", "Registered_U.npy"))
    v = np.load(os.path.join(base_directory, "Preprocessed_Data", "Corrected_SVT.npy"))
    v = v[:, early_cutoff:]

    # Perform NMF
    spatial_components, temporal_components = run_local_nmf.run_local_nmf(u, v, atlas, mask)

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Local_NMF")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

        # View Components
    view_components(spatial_components, save_directory)
    print("Temporal Components Shape", np.shape(temporal_components))

    # Add Earlycutoff Back
    n_temporal_components = np.shape(temporal_components)[0]
    zero_padding = np.zeros((n_temporal_components, early_cutoff))
    print("Temporal Components Pre Padding Shape", np.shape(temporal_components))
    padded_temporal_components = np.hstack([zero_padding, temporal_components])
    print("Temporal Components Post Padding Shape", np.shape(padded_temporal_components))

    np.save(os.path.join(save_directory, "Spatial_Components.npy"), spatial_components)
    np.save(os.path.join(save_directory, "Temporal_Components.npy"), padded_temporal_components)

    #reconstruct_data(spatial_components, temporal_components)



session_list = [

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


    "KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre",
    "KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
    "KPGC3.3E/2023_07_28_Switch_V1_1F_04_1S_Pre",
    "KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
    "KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
    "KPGC3.3E/2023_08_07_Switch_SS_1F_04_1S_Pre",
    "KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
    "KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
    "KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre",


    "KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
    "KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre",
    "KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre",
    "KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
    "KPGC6.2E/2023_08_18_Switch_ALM_1F_06_1S_Pre",
    "KPGC6.2E/2023_08_21_Switch_BC_1F_06_1S_Pre",
    "KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre",
    ]

data_root_directory = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"
for session in tqdm(session_list):
    run_local_nmf_pipeline(os.path.join(data_root_directory, session))
