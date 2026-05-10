import os
number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1


import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

import Session_List
import Extract_CR_FA_Onsets
import Get_Mousecam_Tensors
import Perform_CV_Decoding




def get_combined_data(condition_1_data, condition_2_data):
    combined_data = np.vstack([condition_1_data, condition_2_data])
    condition_1_labels = np.ones(np.shape(condition_1_data)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_data)[0])
    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])
    return combined_data, combined_labels



def decode_timepoint_pair(cr_tensor, timepoint_1_index, timepoint_2_index):

    if timepoint_1_index == timepoint_2_index:
        return timepoint_1_index, timepoint_2_index, 0.5

    timepoint_1_data = cr_tensor[:, timepoint_1_index]
    timepoint_2_data = cr_tensor[:, timepoint_2_index]

    combined_data, combined_labels = get_combined_data(
        timepoint_1_data,
        timepoint_2_data
    )

    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs"
    )

    with threadpool_limits(limits=1):
        average_score, average_coefs = Perform_CV_Decoding.perform_cv(
            model,
            x_all=combined_data,
            y_all=combined_labels,
            n_balance_iterations=1,
            n_folds=5
        )

    return timepoint_1_index, timepoint_2_index, average_score



def get_session_decoding(data_root, session, start_window, stop_window, n_jobs=8):

    # Extract Onsets
    cr_onsets, fa_onsets = Extract_CR_FA_Onsets.extract_cr_fa_onsets(data_root, session)

    # Create Activity Tensor
    cr_tensor = Get_Mousecam_Tensors.get_facecam_tensor(data_root, session, cr_onsets, start_window, stop_window)
    print("cr_tensor", np.shape(cr_tensor))

    # Get Confusion Matrix
    n_timepoints = stop_window - start_window
    confusion_matrix = np.zeros((n_timepoints, n_timepoints))

    # Only upper triangle
    timepoint_pairs = [
        (i, j)
        for i in range(n_timepoints)
        for j in range(i, n_timepoints)
    ]

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(decode_timepoint_pair)(cr_tensor, i, j)
        for i, j in tqdm(timepoint_pairs)
    )

    # Fill symmetric matrix
    for i, j, score in results:
        confusion_matrix[i, j] = score
        confusion_matrix[j, i] = score

    """
    plt.imshow(confusion_matrix, vmin=0.5, vmax=1.0)
    plt.colorbar(label="Decoding accuracy")
    plt.xlabel("Timepoint 2")
    plt.ylabel("Timepoint 1")
    plt.show()
    """
    return confusion_matrix



def cr_decoding_pipeline(data_root, session_list, output_root, start_window, stop_window):

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    # Perform Decoding for Each Session
    for mouse in session_list:
        for session in tqdm(mouse, desc="Mouse"):

            # Get Session Confusion Matrix
            session_confusion_matrix = get_session_decoding(data_root, session, start_window, stop_window)

            # Save Session Confusion Matrix
            save_directory = os.path.join(output_root, session)
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)

            np.save(os.path.join(save_directory, "Confusion_Matrix.npy"), session_confusion_matrix)














# Set Directories
#control_session_list = Session_List.control_post_learning_discrimination
control_session_list = Session_List.control_intermediate_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Cr_Face_Motion\Int\Controls"

#hom_session_list = Session_List.neurexin_post_learning_discrimination
hom_session_list = Session_List.neurexin_intermediate_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Cr_Face_Motion\Int\Homs"


# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


# Run Pipeline
cr_decoding_pipeline(control_data_root, control_session_list, control_output_root, start_window, stop_window)
cr_decoding_pipeline(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window)