import numpy as np
import os
from tqdm import tqdm

import View_Lick_Tuning
import Visualise_Raster
import Visualise_PSTHs
import View_Lick_CD_Projections


def visualise_raw_data(data_root, session_list, mvar_output_root):

    for session in tqdm(session_list):

        # View Lick Tuning
        View_Lick_Tuning.view_lick_tuning(data_root, session, mvar_output_root)

        # View Full Raster
        Visualise_Raster.visualise_raster(data_root, session, mvar_output_root)

        # View Stimuli Aligned PSTHs
        Visualise_PSTHs.view_psths(data_root, session, mvar_output_root,"visual_context_stable_vis_1", -15, 12)
        Visualise_PSTHs.view_psths(data_root, session, mvar_output_root, "visual_context_stable_vis_2", -15, 12)
        Visualise_PSTHs.view_psths(data_root, session, mvar_output_root,"odour_context_stable_vis_1", -15, 12)
        Visualise_PSTHs.view_psths(data_root, session, mvar_output_root, "odour_context_stable_vis_2", -15, 12)

        # View Stimuli Aligned Lick CD Projections
        View_Lick_CD_Projections.get_lick_cd_projection(data_root, session, mvar_output_root, "visual_context_stable_vis_1",-15, 12)
        View_Lick_CD_Projections.get_lick_cd_projection(data_root, session, mvar_output_root, "visual_context_stable_vis_2", -15, 12)
        View_Lick_CD_Projections.get_lick_cd_projection(data_root, session, mvar_output_root, "odour_context_stable_vis_1", -15, 12)
        View_Lick_CD_Projections.get_lick_cd_projection(data_root, session, mvar_output_root, "odour_context_stable_vis_2", -15, 12)
        View_Lick_CD_Projections.plot_all_lick_cds(data_root, session, mvar_output_root, -15, 12)

    # View Group Results
    View_Lick_CD_Projections.view_group_lick_cd(data_root, session_list, mvar_output_root, -15, 12)



# Output directory where you want the data to be saved to
#mvar_output_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR"

# Directory which contains raw data
#data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"
mvar_output_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR\Homs"

session_list = [
    r"64.1B\2024_09_09_Switching",
    r"70.1A\2024_09_19_Switching",
    r"70.1B\2024_09_12_Switching",
    r"72.1E\2024_08_23_Switching",
]




visualise_raw_data(data_root, session_list, mvar_output_root)