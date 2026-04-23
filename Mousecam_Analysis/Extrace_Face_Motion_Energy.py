import os

number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.decomposition import TruncatedSVD
from matplotlib.gridspec import GridSpec

import Session_List
import Mousecam_Utils







def take_closest(myList, myNumber):

    """
    Assumes myList is sorted. Returns closest value to myNumber.
    If two numbers are equally close, return the smallest number.
    """

    pos = bisect_left(myList, myNumber)
    if pos == 0:
        return myList[0]
    if pos == len(myList):
        return myList[-1]
    before = myList[pos - 1]
    after = myList[pos]
    if after - myNumber < myNumber - before:
        return after
    else:
        return before


def match_mousecam_to_widefield_frames(base_directory):

    # Load Frame Times
    widefield_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Frame_Times.npy"), allow_pickle=True)[()]
    mousecam_frame_times = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Mousecam_Frame_Times.npy"), allow_pickle=True)[()]

    widefield_frame_times = Regression_Utils.invert_dictionary(widefield_frame_times)
    widefield_frame_time_keys = list(widefield_frame_times.keys())
    mousecam_frame_times_keys = list(mousecam_frame_times.keys())
    mousecam_frame_times_keys.sort()

    # Get Number of Frames
    number_of_widefield_frames = len(widefield_frame_time_keys)

    # Dictionary - Keys are Widefield Frame Indexes, Values are Closest Mousecam Frame Indexes
    widfield_to_mousecam_frame_dict = {}

    for widefield_frame in range(number_of_widefield_frames):
        frame_time = widefield_frame_times[widefield_frame]
        closest_mousecam_time = take_closest(mousecam_frame_times_keys, frame_time)
        closest_mousecam_frame = mousecam_frame_times[closest_mousecam_time]
        widfield_to_mousecam_frame_dict[widefield_frame] = closest_mousecam_frame

    # Save Directory
    save_directoy = os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy")
    np.save(save_directoy, widfield_to_mousecam_frame_dict)




def get_face_data(video_file, face_pixels):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Extract Selected Frames
    face_data = []
    for frame_index in range(frameCount):
        ret, frame = cap.read()
        frame = frame[:, :, 0]

        face_frame = []
        for pixel in face_pixels:
            face_frame.append(frame[pixel[0], pixel[1]])

        face_data.append(face_frame)

    cap.release()
    face_data = np.array(face_data)
    return face_data, frameHeight, frameWidth



def match_whisker_motion_to_widefield_motion(base_directory, transformed_whisker_data):

    print("Matching")

    # Load Widefield To Mousecam Frame Dict
    widefield_to_mousecam_frame_dict = np.load(os.path.join(base_directory, "Stimuli_Onsets", "widfield_to_mousecam_frame_dict.npy"), allow_pickle=True)[()]
    widefield_frame_list = list(widefield_to_mousecam_frame_dict.keys())
    number_of_mousecam_frames = np.shape(transformed_whisker_data)[0]

    print("Widefield Frames", len(widefield_frame_list))
    print("Mousecam Frames", number_of_mousecam_frames)
    print("Minimum Matched Mousecam Frame", np.min(list(widefield_to_mousecam_frame_dict.values())))
    print("Maximum Matched Mousecam Frame", np.max(list(widefield_to_mousecam_frame_dict.values())))
    print("Transformed Whisker Data Shape", np.shape(transformed_whisker_data))

    # Match Whisker Activity To Widefield Frames
    matched_whisker_data = []
    for widefield_frame in widefield_frame_list:
        corresponding_mousecam_frame = widefield_to_mousecam_frame_dict[widefield_frame]
        if corresponding_mousecam_frame < number_of_mousecam_frames:
            matched_whisker_data.append(transformed_whisker_data[corresponding_mousecam_frame])
        else:
            print("unmatched, mousecam frame: ", "Widefield Frame:", widefield_frame, "Corresponding Mousecam Frame", corresponding_mousecam_frame)
    matched_whisker_data = np.array(matched_whisker_data)


    return matched_whisker_data



def extract_face_motion(data_directory_root, base_directory, save_directory_root):

    # Get Save Directory
    save_directory = os.path.join(save_directory_root, base_directory, "Mousecam_Analysis")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    print("base_directory", base_directory)
    print("Save Directory", save_directory)

    # Load Whisker Pixels
    face_pixels = np.load(os.path.join(data_directory_root, base_directory, "Mousecam_Analysis", "Full_Face_Pixels.npy"))
    face_pixels = np.transpose(face_pixels)

    # Get Bodycam Filename
    bodycam_filename = Mousecam_Utils.get_bodycam_filename(os.path.join(data_directory_root, base_directory))
    bodycam_file = os.path.join(data_directory_root, base_directory, bodycam_filename)

    # Get Face Data
    face_data, frame_height, frame_width = get_face_data(bodycam_file, face_pixels)
    face_data = np.ndarray.astype(face_data, float)
    print("Face Data Shape", np.shape(face_data))

    # Get Face Motion Energy
    face_motion_energy = np.diff(face_data, axis=0)
    face_motion_energy = np.abs(face_motion_energy)
    print("Motion Energy Shape", np.shape(face_motion_energy))

    # Match This TO Widefield Frames
    face_motion_energy = match_whisker_motion_to_widefield_motion(os.path.join(data_directory_root, base_directory), face_motion_energy)

    # Take Mean Average Face Motion Energy
    #mean_face_motion_energy = np.mean(face_motion_energy, axis=1)
    #np.save(os.path.join(save_directory, "Mean_Face_Motion_Energy.npy"), mean_face_motion_energy)
    #print("face_motion_energy", np.shape(face_motion_energy))


    # Perform SVD
    print("Matched Face Data Shape", np.shape(face_motion_energy))
    model = TruncatedSVD(n_components=500)
    model.fit(face_motion_energy)
    transformed_data = model.transform(face_motion_energy)
    model_components = model.components_

    # Save This
    np.save(os.path.join(save_directory, "Full_Face_Motion_SVD.npy"), transformed_data)
    np.save(os.path.join(save_directory, "Full_Face_Motion_Components.npy"), model_components)



"""
# Load Session List
session_list = Session_List.nested_session_list
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/Expansion1/Cortex_Wide_Opto/Experimental_Mice"
save_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"
"""

"""
session_list = Session_List.control_learning_session_list
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/Expansion/Control_Data"


session_list = Session_List.neurexin_learning_session_list
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"




session_list = Session_List.neurexin_pre_learning_list
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

session_list = Session_List.neurexin_intermediate_learning_sessions
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

session_list = Session_List.control_pre_learning
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/Expansion/Control_Data"

"""

session_list = Session_List.control_all_post_learning
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/Expansion/Control_Data"
save_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Neurexin_Widefield/Controls"


session_list = Session_List.neurexin_all_post_learning
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
save_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Neurexin_Widefield/Homs"

session_list = Session_List.neurexin_intermediate_learning
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"
save_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Neurexin_Widefield/Homs"

session_list = Session_List.control_intermediate_learning
session_list = Session_List.flatten_nested_list(session_list)
data_root = r"/media/matthew/Expansion1/Control_Data"
save_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Neurexin_Widefield/Controls"

for session in tqdm(session_list):
    print(session)
    extract_face_motion(data_root, session, save_root)