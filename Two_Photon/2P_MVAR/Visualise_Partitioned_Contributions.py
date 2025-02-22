import os
import numpy as np
import matplotlib.pyplot as plt



def visualise_component_contribution(data_root, session_list, mvar_output_root, start_window, stop_window, frame_rate, model_component):

    visual_group_list = []
    odour_group_list = []

    for session in session_list:

        # Load Lick CD
        lick_cd = np.load(os.path.join(data_root, session,"Coding_Dimensions", "Lick_CD.npy"))

        # Load Contributions
        visual_contribution = np.load(os.path.join(mvar_output_root, session, "Partitioned_Contribution", "visual", "vis_1_" + model_component + "_contribution.npy"))
        odour_contribution = np.load(os.path.join(mvar_output_root, session, "Partitioned_Contribution", "visual", "vis_2_" + model_component + "_contribution.npy"))
        print("visual_contribution", np.shape(visual_contribution))

    # Visualise
