import os

number_of_threads = 1
os.environ["OMP_NUM_THREADS"] = str(number_of_threads) # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = str(number_of_threads) # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = str(number_of_threads) # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = str(number_of_threads) # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = str(number_of_threads) # export NUMEXPR_NUM_THREADS=1

import numpy as np
from tqdm import tqdm
from multiprocessing import Process
import sys

import MVAR_Ridge_Penalty_CV
import Fit_Full_Model_N_Folds


def mvar_pipeline(session_list, mvar_output_directory):

    """
    //// Running The MVAR Pipeline ////
    1.) Perform Ridge Penalty CV to Find Best Ridge Penalty
    2.) Fit Full Model With These Penalties
    """

    # General Preprocessing
    for session in tqdm(session_list, position=0, desc="Session:"):

        # Perform CV For Each Context
        MVAR_Ridge_Penalty_CV.get_cv_ridge_penalties(session, mvar_output_directory, "Combined")

        # Fit Models
        Fit_Full_Model_N_Folds.fit_full_model(mvar_output_directory, session, "Combined")


# File Directory Info
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Final"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

# Control Switching
mvar_pipeline(control_session_list, mvar_output_root)

