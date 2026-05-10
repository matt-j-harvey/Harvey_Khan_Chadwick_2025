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
from scipy import ndimage, stats
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime


import Churchland_Heamocorrect
import Churchland_SVD
import Get_Frames_Excluding_Opto_Stims
import Reshape_Data_For_Churchland
import Reshape_Full_Size_Data_For_Churchland
import View_Churchland_DF

from Preprocessing import Create_Downsampled_Mask_Dict, Run_Motion_Correction_Pipeline




def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def reconstruct_data(base_directory, data):

    # Load Mask
    mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]

    reconstructed_data = []

    for frame in tqdm(data):
        template = np.zeros(image_width * image_height)
        template[indicies] = frame
        template = np.reshape(template, (image_height, image_width))
        template = ndimage.gaussian_filter(template, sigma=1)
        reconstructed_data.append(template)

    reconstructed_data = np.array(reconstructed_data)
    return reconstructed_data


def load_their_data():
    # Load Data
    dataset = r"/home/matthew/Documents/Churchland_Lab_Example_Data/demo_dataset1_2_540_640_uint16.bin"
    data = Load_Binary_File.load_dat(dataset)
    print("Data", np.shape(data))
    return data


def load_my_data(base_directory):

    data_file = os.path.join(base_directory, r"Motion_Corrected_Downsampled_Data.hdf5")

    data_container = h5py.File(data_file, mode="r")
    blue_data = data_container["Blue_Data"]
    violet_data = data_container["Violet_Data"]

    #blue_sample = blue_data[:, sample_start:sample_start + sample_size]
    #violet_sample = violet_data[:, sample_start:sample_start + sample_size]

    blue_sample = blue_data
    blue_sample = np.transpose(blue_sample)
    blue_sample = reconstruct_data(base_directory, blue_sample)
    blue_sample = np.add(blue_sample, 2000)

    violet_sample = violet_data
    violet_sample = np.transpose(violet_sample)
    violet_sample = reconstruct_data(base_directory, violet_sample)
    violet_sample = np.add(violet_sample, 2000)

    combined_data = np.array([blue_sample, violet_sample])
    combined_data = np.swapaxes(combined_data, 0, 1)
    return combined_data


def get_incremental_mean(dataframe, frame_index_list, channel):

    mean_frame = np.zeros(np.shape(dataframe[0, 0]))
    channel_data = dataframe[:, channel]
    n_frames = len(frame_index_list)
    dataframe_size = np.shape(dataframe)[0]

    print("Datafrane Size", dataframe_size)
    for frame_index in tqdm(frame_index_list):

        if frame_index < dataframe_size:
            frame_data = channel_data[frame_index]
            frame_data = np.divide(frame_data, n_frames)
            mean_frame = np.add(mean_frame, frame_data)

        else:
            print("Possile dropped frame: ", frame_index)

    return mean_frame



def get_mean_data(data, exclude_opto_frames, early_cutoff, base_directory):


    if exclude_opto_frames == True:

        # Load Exluded Frame Indexes
        frame_indexes = np.load(os.path.join(base_directory, "Frames_Outside_Opto_Window.npy"))
        channel_1_data = get_incremental_mean(data, frame_indexes, 0)
        channel_2_data = get_incremental_mean(data, frame_indexes, 1)

    else:
        channel_1_data = data[:, 0]
        channel_1_data = np.mean(channel_1_data, axis=0)

        channel_2_data = data[:, 1]
        channel_2_data = np.mean(channel_2_data, axis=0)

    frame_average = np.array([channel_1_data, channel_2_data])
    print("Frame Average", np.shape(frame_average))
    return frame_average





def run_churchland_pipeline_our_data(base_directory, early_cutoff=3000, full_size='downsampled', exlcude_opto_frames=False):

    # Create Save Directory
    save_directory = os.path.join(base_directory, "Churchland_Preprocessing")
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    # Reformat Data
    if full_size == 'downsampled':

        # Check Mask
        if not os.path.exists(os.path.join(base_directory, "Downsampled_mask_dict.npy")):
            print("Creating Downsampled Mask Dict")
            Create_Downsampled_Mask_Dict.create_downsampled_mask_dict(base_directory)

        print("reshaping downsampled data")
        Reshape_Data_For_Churchland.create_churchland_format_dataset(base_directory, save_directory, early_cutoff)

    elif full_size == 'full_size':
        Reshape_Full_Size_Data_For_Churchland.create_churchland_format_dataset(base_directory, save_directory, early_cutoff)

    # Open Data
    datafile = os.path.join(save_directory, "Churchland_Formatted_Data.hdf5")
    datacontainer = h5py.File(datafile, mode="r")
    data = datacontainer["Combined_Data"]
    print("Data Shape", np.shape(data))

    # Get Frame Average
    print("Getting Average", datetime.now())
    frames_average = get_mean_data(data, exlcude_opto_frames, early_cutoff, base_directory)
    np.save(os.path.join(save_directory, "Frame_Mean.npy"), frames_average)

    # Perform SVD
    frames_average = np.load(os.path.join(save_directory, "Frame_Mean.npy"))
    print("Frame Average Shape", np.shape(frames_average))

    print("Performing SVD", datetime.now())
    u, svt = Churchland_SVD.approximate_svd(data, frames_average, k=500)
    np.save(os.path.join(save_directory, "U.npy"), u)
    np.save(os.path.join(save_directory, "SVT.npy"), svt)

    # Perform Heamocorrection
    print("Performing Heamocorrection", datetime.now())
    u = np.load(os.path.join(save_directory, "U.npy"))
    svt = np.load(os.path.join(save_directory, "SVT.npy"))
    blue_svt = svt[:, 0::2]
    violet_svt = svt[:, 1::2]

    corrected_svt, r_coefs, T_matrix = Churchland_Heamocorrect.hemodynamic_correction(u, blue_svt, violet_svt, freq_lowpass=None, freq_highpass=0.0033)

    # Add Padding Back
    if early_cutoff > 0:
        number_of_components, number_of_timepoints = np.shape(corrected_svt)
        zero_padding = np.zeros((number_of_components, early_cutoff))
        corrected_svt = np.hstack([zero_padding, corrected_svt])

    print("Corrected SVT Shape", np.shape(corrected_svt))
    np.save(os.path.join(save_directory, "Corrected_SVT"), corrected_svt)
    np.save(os.path.join(save_directory, "r_coeffs"), r_coefs)
    np.save(os.path.join(save_directory, "T_matrix"), T_matrix)


    # View DF
    movie_name = "Reconstructed_SVT"
    corrected_svt = np.load(os.path.join(save_directory, "Corrected_SVT.npy"))
    u = np.load(os.path.join(save_directory, "U.npy"))
    View_Churchland_DF.view_churchland_df(save_directory, corrected_svt, u, movie_name, sample_start=5000, sample_size=5000)


    # Delete Intermediate File
    os.remove(os.path.join(save_directory, "Churchland_Formatted_Data.hdf5"))

"""

#base_directory = r"//media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging"



session_list = ["/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_14_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_15_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_16_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_17_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_19_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_21_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_24_Discrimination_Imaging",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_28_Switching_Imaging",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_05_Switching_Imaging",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_12_09_Switching_Imaging",

                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_14_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_15_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_16_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_17_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_19_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_21_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_23_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_25_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_29_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_05_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_12_07_Switching_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_04_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_06_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_08_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_10_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_12_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_14_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_02_22_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_04_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_03_06_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_02_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_08_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_04_10_Transition_Imaging"
                
                
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_01_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_03_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_05_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_07_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_09_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_11_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_13_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_15_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_17_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_19_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_22_Discrimination_Imaging",
                 "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_24_Discrimination_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_26_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_02_28_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_02_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_23_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_03_31_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_04_02_Transition_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_29_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_01_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_03_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_05_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_07_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_09_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_21_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_05_23_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_11_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_13_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_15_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_06_17_Transition_Imaging",

                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_25_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_29_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_01_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_03_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_05_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_07_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_08_Discrimination_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_14_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_20_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_22_Switching_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_10_29_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_03_Transition_Imaging",
                "/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_11_05_Transition_Imaging",

                ]





session_list = [

     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging", - Too big to do in 1
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_15_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_16_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_17_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_19_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_21_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_23_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_25_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_27_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_29_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_01_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_03_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_05_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_07_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_12_09_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_08_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_10_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_12_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_16_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_18_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_23_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_25_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_02_27_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_01_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_03_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_03_05_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_08_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_10_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_12_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_05_14_Discrimination_Imaging",
]

session_list = [
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging", # motion corected file not downloaded
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_02_Discrimination_Imaging", # motion corected file not downloaded
     #"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging", # motion corected file not downloaded
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_08_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_10_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_12_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_14_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_16_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_18_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_20_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_22_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_24_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_26_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_06_15_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_28_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_09_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_11_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_13_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_15_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_17_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_10_19_Discrimination_Imaging",

     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_20_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_22_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_24_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_26_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_28_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_30_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_02_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_04_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_06_Discrimination_Imaging",
     "/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_10_08_Discrimination_Imaging",
]


session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2023_02_27_Switching_v1_inhibition"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_15_Retinotopy_Right"]

session_list = [
                r"/media/matthew/Expansion/Control_Data/NRXN78.1A/2020_11_02_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK4.1B/2021_01_25_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK7.1B/2021_01_25_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK14.1A/2021_04_21_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NRXN78.1D/2020_11_02_Spontaneous",
                ]

session_list = [
                #r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK24.1C/2021_09_10_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK20.1B/2021_09_23_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_21_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK10.1A/2021_04_20_Spontaenous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK4.1A/2021_01_29_Spontaneous",
                r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_04_Spontaneous",
                r"/media/matthew/Expansion/Control_Data/NXAK22.1A/2021_09_15_Spontaneous",
                ]



session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_04_17_Retinotopy_Left"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_04_18_Retinotopy_Right",

                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_04_18_Retinotopy_Right"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_04_18_Retinotopy_Right"]

session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_20_Retinotopy_Left"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_21_Retinotopy_Right"]

# Have Already Reformatted The Data For This One
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1F/2023_04_25_Switching_V1"]


session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_28_Grid_Anterior",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_04_28_Grid_Anterior"]

session_list = [r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_05_01_Grid_Anterior"]
#session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_26_Grid_Posterior"]

session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_05_01_Grid_Anterior",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_05_01_Grid_Anterior"]

session_list = [
    "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_05_02_Grid_Posterior",
    "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_05_02_Grid_Posterior",
    "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_05_02_Grid_Posterior",
]


session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_05_22_Switching_V1"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_02_Switching_V1_1_Filter"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_08_Switching_ALM_1_Filter"]


session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_12_Switching_M2_1_Filter"]
session_list = [r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_15_Switching_V1_1_Filter_UC"]
session_list = [r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2023_06_15_Switching_V1_1F_UC"]
session_list = [r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_20_Switching_MM_1_Filter"]
"""

session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_22_Switching_MM_2_Filter",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_06_22_Switch_V1_1F_03",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2023_06_21_Switching_m2_2F",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_06_21_Switch_MM_2F"]

session_list = [
                #r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_06_28_Switch_MM_1F_03",
                r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_28_Switching_V1_1_Filter_03"
]

session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC6.2D/2023_06_28_Retinotopy_Left",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC6.2D/2023_06_29_Retinotopy_Right"]

session_list = [
                #"/media/matthew/Expansion1/Cortex_Wide_Opto/KPGC3.3E/2023_06_30_Switch_ALM_1F_03",
                "/media/matthew/Expansion1/Cortex_Wide_Opto/KPGC3.3A/2023_06_30_Switching_MM_1_Filter_03",
                ]


session_list = [
    r"/media/matthew/Expansion1/Cortex_Wide_Opto/KPGC6.2E/2023_06_28_Retinotopy_Left",
    r"/media/matthew/Expansion1/Cortex_Wide_Opto/KPGC6.2E/2023_06_29_Retinotopy_Right"
]



session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_03_Switching_ALM_1_Filter_03",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_03_Switch_V1_1F_03_Pre"
                ]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_07_04_Switching_ALM_1F_03"]
session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_07_04_Switch_ALM_1F_03"]

session_list = [
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_07_06_Switching_V1_1F_03_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_06_Switch_MM_1F_03_Pre"
                ]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_07_Switching_V1_1F_03_Pre"]
session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_10_Switching_MM_1F_04"]
session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_12_Switching_MM_1F_03_Pre"]





session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_14_Switching_ALM_1F_03_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_17_Switching_RSC_1F_03_Pre"]

session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_14_Switch_ALM_1F_03_Pre",

                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_07_07_Switch_V1_1F_03_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_07_13_Switch_MM_1F_03_Pre",]

session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_07_18_Switch_ALM_1F_03_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_07_18_Switching_MM_1F_03_Pre"]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/KPGC3.3a_Switch_MM_1F_04_1s_Pre/1"]


session_list = [
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_07_20_Switching_RSC_1F_03_Pre",

               r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_19_Switch_RSC_1F_03_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_19_Switching_PPC_1F_03_Pre",
               #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_07_20_Switch_RSC_1F_03_Pre",
            ]


session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre"]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_07_27_Switch_MM_04_1F_1S_Pre"]

session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_28_Switch_RSC_1F_04_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_07_27_Switch_MM_1F_04_1S_Pre"
]

session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_07_28_Switching_MM_1F_04_1s_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_28_Switch_V1_1F_04_1S_Pre",
                ]


session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_07_27_Switch_MM_1f_04_1S_Pre"
]

session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_07_31_Switch_V1_1F_04_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
]

session_list = [
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_07_24_Retinotopy_Left",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_07_25_Retinotopy_Right",
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_07_24_Retinotopy_Left",
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_07_25_Retinotopy_Right"

]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_01_Switch_V1_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre",]


session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_07_Switch_SS_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_08_07_Switch_PM_1F_04_1s_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_08_09_Switch_PPC_1F_04_1s_Pre",
]


session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_08_03_Switching_RSC_1F_04_1s_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_08_03_Switching_RSC_1F_04_1s_Pre",

                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_07_28_Switch_RSC_04_1F_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_08_01_Switch_V1_04_1F_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_08_08_Switch_PM_1F_04_1S_Pre",
                ]


session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_08_10_Switching_V1_1F_04_1s_Pre", #6.2D V1
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_03_Switch_RSC_1F_04_1S_Pre", # r"", #1.3A RsC
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre", #6.2E RSC

]

session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_08_Switch_PM_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
]


session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_08_03_Switch_ALM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3A/2023_08_02_Switch_ALM_1F_04_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_10_Switch_ALM_1F_04_1S_Pre",
]



session_list = [

    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_23_Switch_V1_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_18_Switching_ALM_1F_06_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_21_Switching_BC_1F_06_1S_Pre",
]

session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_25_Switch_ALM_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_08_25_Switch_PPC_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_08_23_Switch_BC_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_24_Switch_PPC_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_25_Switch_V1_1F_06_Stim",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre"
]


session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_28_Switch_BC_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_08_28_Switch_MM_1F_04_Stim",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_08_24_Switch_V1_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_08_24_Switching_PM_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_08_25_Switch_V1_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_08_28_Switch_ALM_1F_04_1S_Pre",
]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_08_30_Switch_V1_1F_04_Stim",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_08_29_Switch_PPC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_08_29_Switching_ALM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_08_29_Switch_ALM_1F_06_Stim",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_08_30_Switch_PM_1F_04_1S_Pre"
                ]

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC1.3A/2023_09_05_Switch_MM_1F_04_Stim",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.1D/2023_09_08_Switch_V1_1F_04_Stim",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.3E/2023_09_07_Switch_V1_and_MM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_09_Switch_20_Degrees"]

"""
session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_12_Switch_V1_1F_04_Stim"
                ]
"""

session_list = [r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_09_Switch_20_Degrees"]

session_list = [
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
                #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_12_Switch_V1_1F_04_Stim",
                ]



session_list = [
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_01_Switch_V1_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_04_Switch_MM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_06_Switch_BC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_08_Switch_RSC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_09_12_Switch_V1_1F_04_Stim",

                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
                r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
                ]


session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_09_07_Switch_MM_1F_06_Stim",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2E/2023_09_13_Switch_ALM_1F_06_Stim"
]

session_list = [
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC6.2D/2023_09_01_Switching_BC_1F_04_1S_Pre",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_09_09_Switch_20_Degrees"
]


session_list = [
    r"/media/matthew/External_Harddrive_1/Neurexin_Data/NRXN71.2A/2020_11_13_Discrimination_Imaging",
]

# Will Need Re-Downloading
#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_04_30_Discrimination_Imaging",
#r"/media/matthew/External_Harddrive_1/Neurexin_Data/NXAK16.1B/2021_05_04_Discrimination_Imaging"

session_list = [
    r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_01_17_Retinotopy_Left",
    r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_01_18_Retinotopy_Right"
]

session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_01_31_Opsin_Screen"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_02_02"]

session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_02_21_Opsin_Screen"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_02_21_Opsin_Screen_1F"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_02_22_Opsin_Screen"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_03_26_Opto_Screen_6"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_25_Switching_Imaging"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_02_14_Discrimination_Imaging"]


session_list = [#r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_02_16_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_02_21_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_02_23_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_02_26_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_02_28_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_01_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_04_Discrimination_Imaging",
                #r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_07_Discrimination_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_11_Discrimination_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_13_Discrimination_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_15_Discrimination_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_18_Discrimination_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_20_Switching_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_22_Switching_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/2024_03_27_Switching_Imaging",

                ]

session_list = [
    r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/KGCA17.1A/2024_04_01_Screen",
    r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/KGCA17.1D/2024_04_01_Screen",
]

session_list = [
    r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_03_Screen"
]

session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_08_Screen"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_08_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_09_Retinotopy_Right"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_09_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/20024_04_10_Retinotopy_Right"]
#session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_11_Mapping_Square_Pulse"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK57.1E/2024_04_12_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_12_Mapping_Square_Pulse"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK57.1E/2024_04_14_Retinotopy_Right"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_15_Mapping_Square_Pulse",
                r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_15_Mapping_Square_Pulse"]

session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK63.1A/2024_04_15_Discrimination_Imaging",
                r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK59.3A/2024_04_16_Discrimination_Imaging"]

session_list = ["/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_15_Mapping_Square_Wave"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK63.1A/2024_04_18_Discrimination_Imaging"]

session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_19_Mapping_Square_Wave",
                r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_19_Mapping_Square_Wave",
                ]

# Get Frames Outside Of Opto Stim
"""
for base_directory in tqdm(session_list, desc="Getting Opto Baseline", position=0):
    Get_Frames_Excluding_Opto_Stims.get_frames_outside_of_opto(base_directory)
"""

# Run Motion Correction
Run_Motion_Correction_Pipeline.run_motion_correction_pipeline(session_list)

# Extract Signal
for base_directory in tqdm(session_list, desc="Session", position=0):
    #run_churchland_pipeline_our_data(base_directory, early_cutoff=0, full_size='full_size')
    #run_churchland_pipeline_our_data(base_directory, early_cutoff=0, exlcude_opto_frames=True)
    run_churchland_pipeline_our_data(base_directory, early_cutoff=3000, exlcude_opto_frames=False)
    #run_churchland_pipeline_our_data(base_directory, early_cutoff=0, exlcude_opto_frames=False)
