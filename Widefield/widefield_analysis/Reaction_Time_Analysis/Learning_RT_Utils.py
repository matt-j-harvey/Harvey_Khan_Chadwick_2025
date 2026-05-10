import numpy as np
import tables
import os



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
        }

    return channel_index_dictionary


def get_ai_filename(base_directory):

    #Get List of all files
    file_list = os.listdir(base_directory)
    ai_filename = None

    #Get .h5 files
    h5_file_list = []
    for file in file_list:
        if file[-3:] == ".h5":
            h5_file_list.append(file)

    #File the H5 file which is two dates seperated by a dash
    for h5_file in h5_file_list:
        original_filename = h5_file

        #Remove Ending
        h5_file = h5_file[0:-3]

        #Split By Dashes
        h5_file = h5_file.split("-")

        if len(h5_file) == 2 and h5_file[0].isnumeric() and h5_file[1].isnumeric():
            return original_filename


def load_ai_recorder_file(base_directory):

    ai_filename = get_ai_filename(base_directory)
    ai_recorder_file_location = os.path.join(base_directory, ai_filename)

    table = tables.open_file(ai_recorder_file_location, mode='r')
    data = table.root.Data

    number_of_seconds = np.shape(data)[0]
    number_of_channels = np.shape(data)[1]
    sampling_rate = np.shape(data)[2]

    data_matrix = np.zeros((number_of_channels, number_of_seconds * sampling_rate))

    for second in range(number_of_seconds):
        data_window = data[second]
        start_point = second * sampling_rate

        for channel in range(number_of_channels):
            data_matrix[channel, start_point:start_point + sampling_rate] = data_window[channel]

    data_matrix = np.clip(data_matrix, a_min=0, a_max=None)
    table.close()
    return data_matrix