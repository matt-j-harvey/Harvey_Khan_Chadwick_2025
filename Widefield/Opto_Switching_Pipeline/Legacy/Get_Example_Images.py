import h5py
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from Preprocessing import Preprocessing_Utils


def get_example_images(base_directory, output_directory, default_position=10000):
    print("Getting Example Image For Session", base_directory)

    # Load Motion Corrected Data
    motion_corrected_filename = "Motion_Corrected_Downsampled_Data.hdf5"
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Blue_Data"]
    violet_matrix = motion_corrected_data_container["Violet_Data"]

    # Get Blue and Violet Example Images
    blue_image = blue_matrix[:, default_position]
    violet_image = violet_matrix[:, default_position]

    # Load Mask
    indicies, image_height, image_width = Preprocessing_Utils.load_downsampled_mask(base_directory)

    # Reconstruct Images
    blue_image = Preprocessing_Utils.create_image_from_data(blue_image, indicies, image_height, image_width)
    violet_image = Preprocessing_Utils.create_image_from_data(violet_image, indicies, image_height, image_width)
    print("Blue Image Shae")

    # Save Images
    np.save(os.path.join(output_directory, "Blue_Example_Image.npy"), blue_image)
    np.save(os.path.join(output_directory, "Violet_Example_Image.npy"), violet_image)

    # Close File
    motion_corrected_data_container.close()



def get_example_images_full_size(base_directory, output_directory, default_position=10000):
    print("Getting Example Image For Session", base_directory)

    # Load Motion Corrected Data
    motion_corrected_filename = "Motion_Corrected_Mask_Data.hdf5"
    motion_corrected_file = os.path.join(base_directory, motion_corrected_filename)
    motion_corrected_data_container = h5py.File(motion_corrected_file, 'r')
    blue_matrix = motion_corrected_data_container["Blue_Data"]
    violet_matrix = motion_corrected_data_container["Violet_Data"]

    # Get Blue and Violet Example Images
    blue_image = blue_matrix[:, default_position]
    violet_image = violet_matrix[:, default_position]

    # Load Mask
    indicies, image_height, image_width = Preprocessing_Utils.load_generous_mask(base_directory)

    # Reconstruct Images
    blue_image = Preprocessing_Utils.create_image_from_data(blue_image, indicies, image_height, image_width)
    violet_image = Preprocessing_Utils.create_image_from_data(violet_image, indicies, image_height, image_width)
    print("Blue Image Shae")

    # Save Images
    np.save(os.path.join(output_directory, "Blue_Example_Image_Full_Size.npy"), blue_image)
    np.save(os.path.join(output_directory, "Violet_Example_Image_Full_Size.npy"), violet_image)

    # Close File
    motion_corrected_data_container.close()


base_directory = r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_08_14_Retinotopy_Left"
base_directory = r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_08_29_Retinotopy_Right_2"
session_list = [r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC13.3a/2024_08_20_Retinotopy_Left",
                r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC13.3a/2024_08_22_Retinotopy_Right"]


get_example_images_full_size(session_list[1], session_list[1])