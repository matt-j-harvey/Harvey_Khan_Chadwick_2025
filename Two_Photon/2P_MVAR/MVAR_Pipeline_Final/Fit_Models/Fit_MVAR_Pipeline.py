import os
from tqdm import tqdm

import Create_Design_Matrix
import MVAR_Ridge_Penalty_CV
import Fit_Full_Model_N_Folds


def fit_mvar_models_pipeline(data_root, session, mvar_output_root,  start_window, stop_window, model_type):

    """
    //// Running The MVAR Pipeline ////
    1.) Create Design Matrix
    2.) Load Model
    3.) Perform Ridge Penalty CV to Find Best Ridge Penalties
    4.) Fit Full Model With These Penalties
    """

    # Create Regression Matricies
    Create_Design_Matrix.create_regression_matrix(data_root, session, mvar_output_root, start_window, stop_window, model_type)

    # Perform CV
    MVAR_Ridge_Penalty_CV.get_cv_ridge_penalties(session, mvar_output_root, model_type)

    # Fit Models
    Fit_Full_Model_N_Folds.fit_full_model(mvar_output_root, session, model_type)




# Output directory where you want the data to be saved to
mvar_output_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR"

# Directory which contains raw data
data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

# Model Info
start_window = -17
stop_window = 12

for session in control_session_list:

    fit_mvar_models_pipeline(data_root, session, mvar_output_root, start_window, stop_window, model_type="No_Recurrent")
    fit_mvar_models_pipeline(data_root, session, mvar_output_root, start_window, stop_window, model_type="Standard")
    fit_mvar_models_pipeline(data_root, session, mvar_output_root, start_window, stop_window, model_type="Seperate_Contexts")

