import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
import pickle
from skimage.morphology import binary_dilation
from skimage.segmentation import flood_fill


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def create_stimuli_dictionary():

    channel_index_dictionary = {
        "Photodiode"        :0,
        "Reward"            :1,
        "Lick"              :2,
        "Visual 1"          :3,
        "Visual 2"          :4,
        "Odour 1"           :5,
        "Odour 2"           :6,
        "Irrelevance"       :7,
        "Running"           :8,
        "Trial End"         :9,
        "Camera Trigger"    :10,
        "Camera Frames"     :11,
        "LED 1"             :12,
        "LED 2"             :13,
        "Mousecam"          :14,
        "Optogenetics"      :15,
        "Optogenetics"      :15,
        }

    return channel_index_dictionary






def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image









def load_tight_mask():
    tight_mask_file = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\Tight_Mask_Dict.npy"
    tight_mask_dict = np.load(tight_mask_file, allow_pickle=True)[()]
    indicies = tight_mask_dict["indicies"]
    image_height = tight_mask_dict["image_height"]
    image_width = tight_mask_dict["image_width"]
    return indicies, image_height, image_width

def load_atlas():
    atlas = np.load(r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\M2_Three_Segments_Masked_All_Sessions.npy")
    return atlas



def get_roi_pixels(atlas, roi_list):

    # Mask Atlas
    indicies, image_height, image_width = load_tight_mask()
    atlas = np.reshape(atlas, image_height * image_width)
    atlas = atlas[indicies]

    selected_indicies = []
    for roi in roi_list:
        roi_mask = np.where(atlas==roi, 1, 0)
        roi_indicies = np.argwhere(roi_mask)

        for index in roi_indicies:
            selected_indicies.append(index[0])

    selected_indicies = np.array(selected_indicies)
    return selected_indicies




def get_atlas_outline_pixels():

    # Load Atlas
    atlas_outline = np.load(r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\churchland_outlines_aligned_single.npy")
    atlas_outline = np.roll(atlas_outline, -5, axis=1)
    atlas_outline = binary_dilation(atlas_outline)
    #atlas_outline[3:8, 115:186] = 0
    atlas_pixels = np.nonzero(atlas_outline)


    return atlas_pixels


def get_full_outlines():

    # Load Atlas
    atlas_outline = np.load(r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Code_Resource_Files\churchland_outlines_aligned_single.npy")
    atlas_outline = np.roll(atlas_outline, -5, axis=1)
    atlas_outline = binary_dilation(atlas_outline)
    flood_filled_atlas_outline = flood_fill(atlas_outline, seed_point=(0,0), new_value=2)
    flood_filled_atlas_outline = np.ndarray.astype(flood_filled_atlas_outline, int)
    new_atlas_outline = np.subtract(flood_filled_atlas_outline, atlas_outline)
    outline_pixels = np.nonzero(new_atlas_outline)
    return outline_pixels


def get_background_pixels(indicies, image_height, image_width):
    template = np.ones(image_height * image_width)
    template[indicies] = 0
    template = np.reshape(template, (image_height, image_width))
    background_pixels = np.nonzero(template)
    return background_pixels



def create_image_from_data(data, indicies, image_height, image_width):
    template = np.zeros((image_height, image_width))
    data = np.nan_to_num(data)
    np.put(template, indicies, data)
    image = np.ndarray.reshape(template, (image_height, image_width))
    return image


def get_musall_cmap():
    cmap = LinearSegmentedColormap.from_list('mycmap', [

        (0, 0.87, 0.9, 1),
        (0, 0, 1, 1),
        (0, 0, 0, 1),
        (1, 0, 0, 1),
        (1, 1, 0, 1),

    ])

    return cmap



def get_mean_sd(data):
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound
