import os
import numpy as np
from pathlib import PurePath
import pandas as pd
import pickle

from Shared_Utils.Recurrent_Matrix_Analysis_Utils import load_recurrent_weights, get_stimuli_weights
import Shared_Utils.Session_Lists as Session_Lists

def create_session_dict(mvar_directory, session, genotype):

    # Load Recurrent Weight Matrix
    recurrent_matrix = load_recurrent_weights(mvar_directory, session)

    # Load Stimulus Weights
    vis_1_weights, vis_2_weights, vis_1_time, vis_2_time = get_stimuli_weights(mvar_directory, session)

    # Load Lick CD
    lick_cd = np.load(os.path.join(mvar_directory, session, "Lick_Tuning", "Lick_Coding_Dimension.npy"))

    session_path = PurePath(session).parts
    mouse = session_path[0]

    session_dict = {
        "Lick_CD": lick_cd.tolist(),
        "Recurrent_Matrix": recurrent_matrix,
        "Vis_1_Input": vis_1_weights,
        "Vis_2_Input": vis_2_weights,
        "Genotype": genotype,
        "Mouse": mouse,
    }

    return session_dict


def get_group_dicts(mvar_root, session_list, genotype):

    dictionary_list = []
    for mouse in session_list:
        for session in mouse:
            session_dict = create_session_dict(mvar_root, session, genotype)
            dictionary_list.append(session_dict)

    return dictionary_list


def save_dataset_pickle(dataset, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "wb") as handle:
        pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved dataset to: {output_path}")

# Model Info
start_window = -10 # How many timepoints before the onset of each stimulus to include
stop_window = 10 # How many timepoints after the onset of each stimulus to include

# Data root Directory
wt_data_root =  r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
hom_data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"

# MVAR Root Directory
wt_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR_Multi_Session\WT"
hom_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR_Multi_Session\Homs"

# Output Directories
output_root = r"C:\Recurrent_Matrix_Analysis\Data_Dicts"


wt_session_list = Session_Lists.wt_post_learning_sessions
hom_session_list = Session_Lists.hom_post_learning_sessions

wt_dicts = get_group_dicts(wt_mvar_root, wt_session_list, "Wildtype")
hom_dicts = get_group_dicts(hom_mvar_root, hom_session_list, "Hom")
dataset = wt_dicts + hom_dicts

save_dataset_pickle(dataset, os.path.join(output_root, "Dataset.pkl"))




