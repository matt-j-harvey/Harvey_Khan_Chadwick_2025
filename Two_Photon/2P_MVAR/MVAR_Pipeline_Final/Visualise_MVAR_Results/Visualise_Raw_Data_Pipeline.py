import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


"""
View Raw data which we will compare the MVAR results to do:

This predominantly invovles:
    Trial average histograms
    Trial average lick CD Projections

"""

def load_activity_tensor(file_path):

    with open(file_path, 'rb') as f:
        x = pickle.load(f)
        tensor = x['tensor']

    return tensor



def visualise_raw_data_pipeline(data_root, session_list, mvar_root):

    for session in session_list:

        # Load Activity Tensor
        tensor_filepath = os.path.join(mvar_root, session, "Activity_Tensors", "visual_context_stable_vis_1.pickle")
        activity_tensor = load_activity_tensor(tensor_filepath)
        print("Activity tensor", np.shape(activity_tensor))

        # Get Mean Activity
        trial_mean = np.mean(activity_tensor, axis=0)


# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Final"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

visualise_raw_data_pipeline(data_root, control_session_list, mvar_output_root)