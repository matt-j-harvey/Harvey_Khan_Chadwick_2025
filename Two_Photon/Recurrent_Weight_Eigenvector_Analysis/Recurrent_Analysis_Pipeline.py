import os
from tqdm import tqdm

import Shared_Utils.Session_Lists as Session_Lists
import Plot_Group_Comparison
from Analyse_Recurrent_Matrix import analyse_recurrent_weight_matrix
import Compare_Group_Projections


def analyse_group(data_root, mvar_root, session_list, output_root, group_name, pre_learning):
    for mouse in tqdm(session_list):
        for session in mouse:
            analyse_recurrent_weight_matrix(data_root, mvar_root, session, output_root, pre_learning)







# Model Info
start_window = -10 # How many timepoints before the onset of each stimulus to include
stop_window = 10 # How many timepoints after the onset of each stimulus to include

# Data root Directory
wt_data_root =  r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
hom_data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"

wt_data_root_pre = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\ALM 2P\Data\Controls"
hom_data_root_pre = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\ALM 2P\Data\Homs"


# MVAR Root Directory
wt_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR_Multi_Session\WT"
hom_mvar_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\MVAR_Multi_Session\Homs"

wt_mvar_root_pre = r"C:\Learning_MVAR\WT"
hom_mvar_root_pre = r"C:\Learning_MVAR\Hom"

# Output Directories
wt_output_root = r"C:\Recurrent_Matrix_Analysis\Switching\Wt"
hom_output_root = r"C:\Recurrent_Matrix_Analysis\Switching\Hom"

# Session Lists
wt_pre_learning_sessions = Session_Lists.wt_pre_learning_sessions
hom_pre_learning_sessions = Session_Lists.hom_pre_learning_sessions

wt_post_learning_sessions = Session_Lists.wt_post_learning_sessions
hom_post_learning_sessions = Session_Lists.hom_post_learning_sessions

wt_sessions = wt_pre_learning_sessions + wt_post_learning_sessions
hom_sessions = hom_pre_learning_sessions + hom_post_learning_sessions

# Analyse Groups

analyse_group(wt_data_root, wt_mvar_root, wt_post_learning_sessions, wt_output_root, "wildtype", pre_learning=False)
analyse_group(hom_data_root, hom_mvar_root, hom_post_learning_sessions, hom_output_root, "neurexin", pre_learning=False)

analyse_group(wt_data_root_pre, wt_mvar_root_pre, wt_pre_learning_sessions, wt_output_root, "wildtype", pre_learning=True)
analyse_group(hom_data_root_pre, hom_mvar_root_pre, hom_pre_learning_sessions, hom_output_root, "neurexin", pre_learning=True)


# Plot Results
Plot_Group_Comparison.plot_group_comparison(wt_post_learning_sessions, wt_output_root, hom_post_learning_sessions, hom_output_root)
Plot_Group_Comparison.plot_group_comparison(wt_pre_learning_sessions, wt_output_root, hom_pre_learning_sessions, hom_output_root)


#Compare_Group_Projections.compare_projections_group(wt_post_learning_sessions, wt_data_root, wt_mvar_root, hom_post_learning_sessions, hom_data_root, hom_mvar_root)