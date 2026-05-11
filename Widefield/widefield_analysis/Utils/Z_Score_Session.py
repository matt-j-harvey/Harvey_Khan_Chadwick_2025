import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm

import GLM_Utils
import Session_List


def remove_early_frames(frame_list, cutoff=3000):
    kept_frames = []
    for frame in frame_list:
        if frame > cutoff:
            kept_frames.append(frame)
    return kept_frames


def z_score_session(session):

    # Load Corrected SVT
    corrected_svt = np.load(os.path.join(session, "Preprocessed_Data", "Corrected_SVT.npy"))
    registered_u = np.load(os.path.join(session, "Preprocessed_Data", "Registered_U.npy"))
    print("corrected_svt", np.shape(corrected_svt))
    print("registered_u", np.shape(registered_u))

    # Remove Early Frames
    corrected_svt = corrected_svt[:, 3000:]

    # Flatten Reg U
    image_height, image_width, n_components = np.shape(registered_u)
    registered_u = np.reshape(registered_u, (image_height * image_width, n_components))
    indicies, image_height, image_width = GLM_Utils.load_tight_mask()
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
    colourmap = GLM_Utils.get_musall_cmap()

    data_mean = GLM_Utils.create_image_from_data(data_mean, indicies, image_height, image_width)
    plt.imshow(data_mean, cmap=colourmap, vmin=-0.05, vmax=0.05)
    plt.savefig(os.path.join(save_directory, "pixel_means.png"))
    plt.close()

    data_std = GLM_Utils.create_image_from_data(data_std, indicies, image_height, image_width)
    plt.imshow(data_std, cmap=colourmap, vmin=-0.05, vmax=0.05)
    plt.savefig(os.path.join(save_directory, "pixel_STD.png"))
    plt.close()





# Set Directories
control_session_list = Session_List.control_all_post_learning
control_session_list = Session_List.flatten_nested_list(control_session_list)
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"

for session in tqdm(control_session_list):
    base_directory = os.path.join(control_data_root, session)
    z_score_session(base_directory)