import os


number_of_threads = 2
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

import Session_List
import Extract_CR_FA_Onsets
import Get_Mousecam_Tensors
import Perform_CV_Decoding



def get_combined_data(condition_1_data, condition_2_data):
    combined_data = np.vstack([condition_1_data, condition_2_data])
    condition_1_labels = np.ones(np.shape(condition_1_data)[0])
    condition_2_labels = np.zeros(np.shape(condition_2_data)[0])
    combined_labels = np.concatenate([condition_1_labels, condition_2_labels])

    print("combined Data", np.shape(combined_data))
    print("combined lables", np.shape(combined_labels))

    return combined_data, combined_labels



def face_decoding_pipeline(data_root, session_list, output_root, start_window, stop_window):

    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    # Perform Decoding for Each Session
    for mouse in session_list:
        for session in tqdm(mouse, desc="Mouse"):
            print(session)

            # Extract Onsets
            cr_onsets, fa_onsets = Extract_CR_FA_Onsets.extract_cr_fa_onsets(data_root, session)

            if len(cr_onsets) > 5 and len(fa_onsets) > 5:

                # Create Activity Tensors
                cr_tensor = Get_Mousecam_Tensors.get_facecam_tensor(data_root, session, cr_onsets, start_window, stop_window)
                fa_tensor = Get_Mousecam_Tensors.get_facecam_tensor(data_root, session, fa_onsets, start_window, stop_window)
                print("cr_tensor", np.shape(cr_tensor))
                print("fa_tensor", np.shape(fa_tensor))

                # Combine Data
                combined_data, combined_labels = get_combined_data(cr_tensor, fa_tensor)

                # Perform Decoding
                n_timepoints = np.shape(combined_data)[1]
                score_list = []
                for timepoint_index in range(n_timepoints):

                    # Get Timepoint Data
                    timepoint_data = combined_data[:, timepoint_index]

                    model = LogisticRegression(max_iter=2000)

                    # Perform Decoding
                    average_score, average_coefs = Perform_CV_Decoding.perform_cv(model, x_all=timepoint_data, y_all=combined_labels, n_balance_iterations=20, n_folds=5)
                    score_list.append(average_score)

                save_directory = os.path.join(output_root, session)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                np.save(os.path.join(save_directory, "Decoding_Scores.npy"), score_list)

                #plt.plot(x_values, score_list)
                #plt.show()

        """
        visual_mean = np.mean(visual_context_tensor, axis=0)
        odour_mean = np.mean(odour_context_tensor, axis=0)
        diff = np.subtract(visual_mean, odour_mean)

        plt.imshow(visual_mean, cmap="bwr", vmin=-2, vmax=2)
        plt.show()

        plt.imshow(odour_mean, cmap="bwr", vmin=-2, vmax=2)
        plt.show()

        plt.imshow(diff, cmap="bwr", vmin=-1, vmax=1)
        plt.show()
        """



# Set Directories
control_session_list = Session_List.control_all_post_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Face_Decoding\Controls"

hom_session_list = Session_List.neurexin_all_post_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Face_Decoding\Homs"


# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


# Run Pipeline
face_decoding_pipeline(control_data_root, control_session_list, control_output_root, start_window, stop_window)
face_decoding_pipeline(hom_data_root, hom_session_list, hom_output_root, start_window, stop_window)