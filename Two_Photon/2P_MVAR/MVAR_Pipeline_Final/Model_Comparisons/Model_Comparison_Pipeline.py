import os
import numpy as np
import matplotlib.pyplot as plt

import Get_Previous_Step_R2

def model_comparison_pipeline(mvar_output_root, session_list):

    for session in session_list:

        # Get Previous Step R2
        Get_Previous_Step_R2.get_previous_step_r2(mvar_output_root, session)





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
model_comparison_pipeline(mvar_output_root, control_session_list)

