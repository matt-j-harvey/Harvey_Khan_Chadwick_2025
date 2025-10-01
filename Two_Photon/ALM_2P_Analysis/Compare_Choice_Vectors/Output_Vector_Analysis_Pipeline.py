import os



import Get_Lick_Onsets
import Get_Data_Tensor
import Plot_PSTHs
import Get_Coding_Dimensions
import Test_Cosine_Simmilarity
import Output_Vector_Plotting_Functions

def output_vector_analysis_pipeline(session_list, data_root, output_root):

    """
    Possible Output Vectors
    1.) Combined Lick Dimension
    2.) Visual Lick Dimension
    3.) Odour lick Dimension
    4.) Visual Choice Dimension
    5.) Odour Choice Dimension
    """

    # Window Settings
    lick_start_window = -1.5
    lick_stop_window = 0
    choice_start_window = -1
    choice_stop_window = 1.5

    # Iterate Through Each Session
    for session in session_list:

        # Set Data Directory
        data_directory = os.path.join(data_root, session)

        # Create Output Directory
        output_directory = os.path.join(output_root, session)
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        """
        # Get Lick Onsets
        Get_Lick_Onsets.get_lick_onsets(data_directory, output_directory, min_rt=0.5, max_rt=2.5, lick_threshold=600)

        # Get Data Tensors
        Get_Data_Tensor.get_lick_tensors(data_directory, output_directory, start_window=-1.5, stop_window=0)
        Get_Data_Tensor.get_choice_tensors(data_directory, output_directory, start_window=-1, stop_window=1.5)

        # Plot PSTHs
        Plot_PSTHs.view_psths(data_directory, output_directory, lick_start_window, lick_stop_window, choice_start_window, choice_stop_window)
        """
        # Save Vectors
        Get_Coding_Dimensions.get_coding_dimensions(output_directory)

        # Test Cosine Simmilarity
        #Test_Cosine_Simmilarity.perform_similarity_checks(data_directory, output_directory)

        # Consider Scatterplot of visual lick coding v odour lick coding

        # Consider Pi Chart Of Tuning

    # Create Group Save Directory
    save_directory = os.path.join(output_root, "Group_Results")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    Output_Vector_Plotting_Functions.plot_group_simmilarities(output_root, session_list, save_directory)


# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Output_Vector_Analysis"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

output_vector_analysis_pipeline(control_session_list, data_root, output_root)



