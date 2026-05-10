import numpy as np
import os
from scipy import ndimage
import h5py
from tqdm import tqdm


def reconstruct_data(data, indicies, image_height, image_width):

    reconstructed_data = []
    for frame in data:
        template = np.zeros(image_width * image_height)
        template[indicies] = frame
        template = np.reshape(template, (image_height, image_width))
        template = ndimage.gaussian_filter(template, sigma=1)
        reconstructed_data.append(template)

    reconstructed_data = np.array(reconstructed_data)
    return reconstructed_data


def get_chunk_structure(chunk_size, array_size):
    number_of_chunks = int(np.ceil(array_size / chunk_size))
    remainder = array_size % chunk_size

    # Get Chunk Sizes
    chunk_sizes = []
    if remainder == 0:
        for x in range(number_of_chunks):
            chunk_sizes.append(chunk_size)

    else:
        for x in range(number_of_chunks - 1):
            chunk_sizes.append(chunk_size)
        chunk_sizes.append(remainder)

    # Get Chunk Starts
    chunk_starts = []
    chunk_start = 0
    for chunk_index in range(number_of_chunks):
        chunk_starts.append(chunk_size * chunk_index)

    # Get Chunk Stops
    chunk_stops = []
    chunk_stop = 0
    for chunk_index in range(number_of_chunks):
        chunk_stop += chunk_sizes[chunk_index]
        chunk_stops.append(chunk_stop)

    return number_of_chunks, chunk_sizes, chunk_starts, chunk_stops



def create_churchland_format_dataset(base_directory, save_directory, early_cutoff, datafile=r"Motion_Corrected_Downsampled_Data.hdf5"):

    # Their Container Shape = [N Frames x N Channels X Height X Width]
    data_file = os.path.join(base_directory, datafile)

    data_container = h5py.File(data_file, mode="r")
    blue_data = data_container["Blue_Data"]
    violet_data = data_container["Violet_Data"]

    blue_data = np.array(blue_data)
    violet_data = np.array(violet_data)
    print("blue data", np.shape(blue_data))
    n_pixel, n_frames = np.shape(blue_data)


    # Define Chunking Settings
    preferred_chunk_size = 1000
    number_of_chunks, chunk_sizes, chunk_starts, chunk_stops = get_chunk_structure(preferred_chunk_size, n_frames - early_cutoff)

    # Load Mask
    mask_dict = np.load(os.path.join(base_directory, "Downsampled_mask_dict.npy"), allow_pickle=True)[()]
    indicies = mask_dict["indicies"]
    image_height = mask_dict["image_height"]
    image_width = mask_dict["image_width"]
    print("Indiices", np.shape(indicies))

    # Create New Dataframe
    churchland_formatted_data = os.path.join(save_directory, "Churchland_Formatted_Data.hdf5")
    with h5py.File(churchland_formatted_data, "w") as f:
        combined_dataset = f.create_dataset("Combined_Data", (n_frames - early_cutoff, 2, image_height, image_width), dtype=np.float32, chunks=True, compression=False)

        for chunk_index in tqdm(range(number_of_chunks), desc="Reshaping Data"):

            # Get Selected Indicies
            chunk_start = int(chunk_starts[chunk_index])
            chunk_stop = int(chunk_stops[chunk_index])

            # Extract Chunk
            blue_chunk = blue_data[:, early_cutoff + chunk_start: early_cutoff + chunk_stop]
            violet_chunk = violet_data[:, early_cutoff + chunk_start: early_cutoff + chunk_stop]

            # Transpose
            blue_chunk = np.transpose(blue_chunk)
            violet_chunk = np.transpose(violet_chunk)

            # Reconstruct
            blue_chunk = reconstruct_data(blue_chunk, indicies, image_height, image_width)
            violet_chunk = reconstruct_data(violet_chunk, indicies, image_height, image_width)

            # Add Offset To Reduce The Impact Of Very Dark Pixels Which Will Have High D/F Variance
            blue_chunk = np.add(blue_chunk, 2000)
            violet_chunk = np.add(violet_chunk, 2000)

            # Insert Back
            combined_dataset[chunk_start:chunk_stop, 0] = blue_chunk
            combined_dataset[chunk_start:chunk_stop, 1] = violet_chunk

