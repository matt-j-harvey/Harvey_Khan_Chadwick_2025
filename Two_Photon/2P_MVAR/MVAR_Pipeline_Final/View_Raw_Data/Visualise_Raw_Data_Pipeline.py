import numpy as np
import os
from tqdm import tqdm

import Get_Delta_F
import View_Lick_Tuning
import Visualise_Raster
import Visualise_PSTHs
import View_Lick_CD_Projections


def visualise_raw_data(data_root, session_list, mvar_output_root):

    for session in tqdm(session_list):

        # Get Delta F Matrix
        Get_Delta_F.get_delta_f(data_root, session, mvar_output_root)

        # View Lick Tuning and get Lick CD
        View_Lick_Tuning.view_lick_tuning(data_root, session, mvar_output_root)

        """
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
    """

# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Full_Pipeline_Results_Check"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


visualise_raw_data(data_root, control_session_list, mvar_output_root)