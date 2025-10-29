import numpy as np

import Get_Delta_F
import Downsample_AI_Matrix_Framewise_2P
import Get_Lick_CD
import Plot_Lick_Tuning
import Quantify_Lick_Tuning
import Plot_Lick_CD_Modulation
import Split_Hits_By_RT
import Plot_Lick_CD_By_RT



def alm_analysis_pipeline(data_root, session_list, output_root):

    for session in session_list:

        # Get DF/F
        Get_Delta_F.get_delta_f(data_root, session, output_root, plot_rastermap=False)

        # Downsample AI Matrix
        Downsample_AI_Matrix_Framewise_2P.downsample_ai_matrix(data_root, session, output_root)

        # Get Lick CD
        Get_Lick_CD.get_lick_cd(data_root, session, output_root)

        # Plot Lick Tuning
        Plot_Lick_Tuning.get_lick_tuning(data_root, session, output_root)

        # Quantify Neuron Proportions
        Quantify_Lick_Tuning.quantify_lick_tuning_session(data_root, session, output_root)

        # Plot Lick CD Modulation By Context
        Plot_Lick_CD_Modulation.plot_lick_cd_modulation(data_root, session, output_root)

        # Split Hits By RT
        Split_Hits_By_RT.split_hits_by_rt(data_root, session, output_root, lick_threshold=600)


    # Plot Group Lick Tuning
    Plot_Lick_Tuning.plot_group_lick_tuning(data_root, session_list, output_root)

    # Plot Group Quantification
    Plot_Lick_Tuning.plot_group_pichart(session_list, output_root)
    Plot_Lick_Tuning.plot_group_scatter(session_list, output_root)

    # Plot Group Modulation By Context
    Plot_Lick_CD_Modulation.plot_group_modulation(data_root, session_list, output_root)

    # Plot Lick CD By RT
    Plot_Lick_CD_By_RT.plot_lick_cd_by_rt(data_root, session_list, output_root)



# Output directory where you want the data to be saved to
control_data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
control_output_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\2P_Analysis\Controls"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


# Output directory where you want the data to be saved to
hom_data_root = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"
hom_output_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\2P_Analysis\Homs"

hom_session_list = [
    r"64.1B\2024_09_09_Switching", # Looks Fine
    #r"70.1A\2024_09_09_Switching", # Looks very strange --- # washing....??
    r"70.1A\2024_09_19_Switching",
    r"70.1B\2024_09_12_Switching", # Activity starts after lick?
    r"72.1E\2024_08_23_Switching", # Looks Fine
]

alm_analysis_pipeline(control_data_root, control_session_list, control_output_directory)
#alm_analysis_pipeline(hom_data_root, hom_session_list, hom_output_directory)