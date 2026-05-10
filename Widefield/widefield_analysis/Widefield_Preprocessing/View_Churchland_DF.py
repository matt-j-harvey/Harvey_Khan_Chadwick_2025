import numpy as np
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from matplotlib import cm
import h5py
import os
import pickle
from scipy import ndimage
from tqdm import tqdm
from matplotlib.colors import LinearSegmentedColormap
import cv2

def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    return cmap



def view_churchland_df(save_directory, corrected_svt, u, movie_name, sample_start=10000, sample_size=5000):

    # Reconstruct Sample
    print("Corrected SVT", np.shape(corrected_svt))
    svt_sample = corrected_svt[:, sample_start:sample_start + sample_size]

    sample_data = np.dot(u, svt_sample)
    print("Sample Data Shape", np.shape(sample_data))

    # Get Data Shape
    print(np.shape(sample_data))
    image_height, image_width, number_of_frames = np.shape(sample_data)

    colourmap = get_musall_cmap()
    cm = plt.cm.ScalarMappable(norm=None, cmap=colourmap)
    colour_magnitude = 0.05
    cm.set_clim(vmin=-1 * colour_magnitude, vmax=colour_magnitude)

    # Create Video File
    video_name = os.path.join(save_directory, movie_name + ".avi")
    video_codec = cv2.VideoWriter_fourcc(*'DIVX')
    video = cv2.VideoWriter(video_name, video_codec, frameSize=(image_width, image_height), fps=30)  # 0, 12

    window_size=3
    for frame_index in range(0, number_of_frames-window_size):
        image = sample_data[:, :, frame_index:frame_index + window_size]
        image = np.mean(image, axis=2)

        # Set Image Colours
        colored_image = cm.to_rgba(image)

        colored_image = colored_image * 255
        image = np.ndarray.astype(colored_image, np.uint8)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        video.write(image)

    cv2.destroyAllWindows()
    video.release()

