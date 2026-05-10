import Check_Photodiode_Trace
import Continous_Retinotopy_Fourier_Analysis_SVT
import Create_Sweep_Aligned_Movie
import Extract_Trial_Aligned_Activity_Continous_Retinotopy_From_SVT


#Retinotopy_Utils

session_list = [
                # "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_14_Retinotopy_Left",
                # "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1D/2022_12_15_Retinotopy_Right",
                # "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1F/2022_12_15_Continuous_Retinotopic_Mapping_Left",
                # "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.1F/2022_12_14_Continuous_Retinotopic_Mapping_Right"


                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3a/2023_04_17_Retinotopy_Left",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_04_18_Retinotopy_Right",

                # "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_04_17_Retinotopy_Left"


                #"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_04_18_Retinotopy_Right",
                #"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_20_Retinotopy_Left",
                "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_21_Retinotopy_Right"
                ]


session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_04_20_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC6.2D/2023_06_28_Retinotopy_Left"]
session_list = ["/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC6.2D/2023_06_29_Retinotopy_Right"]
session_list = [
    #r"/media/matthew/Expansion1/Cortex_Wide_Opto/KPGC6.2E/2023_06_28_Retinotopy_Left",
    r"/media/matthew/Expansion1/Cortex_Wide_Opto/KPGC6.2E/2023_06_29_Retinotopy_Right"
]



session_list = [
    #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_07_24_Retinotopy_Left",
    #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC3.4A/2023_07_25_Retinotopy_Right",
    #r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_07_24_Retinotopy_Left",
    r"/media/matthew/Expansion/Cortex_Wide_Opto/KPGC7.4A/2023_07_25_Retinotopy_Right"
]
session_list = [
    #r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_01_17_Retinotopy_Left",
    r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KPG6.3B/2024_01_18_Retinotopy_Right"
]

session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_08_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1D/2024_04_09_Retinotopy_Right"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/2024_04_09_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Closed_Loop_Opto_Data/KGCA17.1A/20024_04_10_Retinotopy_Right"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK57.1E/2024_04_12_Retinotopy_Left"]
session_list = [r"/media/matthew/External_Harddrive_2/Neurexin_ATX_Data/NXAK57.1E/2024_04_14_Retinotopy_Right"]

session_list = [

    #r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1E/2024_06_07_Retinotopy_Left",
    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.1D/2024_06_07_Retinotopy_Left",
    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC14.1A/2024_06_07_Retinotopy_Left",
]


session_list = [

    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1B/2024_06_10_Retinotopy_Left",
    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1B/2024_06_12_Retinotopy_Right",

    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_06_10_Retinotopy_Left",
    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_06_12_Retinotopy_Right",

    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1E/2024_06_10_Retinotopy_Right",

    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.1D/2024_06_11_Retinotopy_Right",

    r"/media/matthew/External HDD/Closed_Loop_Opto/KPGC14.1A/2024_06_11_Retinotopy_Right",

]


for base_directory in session_list:

    # Get Stimuli Onsets
    Check_Photodiode_Trace.check_photodiode_times(base_directory)

    # Extract Trial Aligned Activity
    #Extract_Trial_Aligned_Activity_Continous_Retinotopy.extract_trial_aligned_activity(base_directory)
    Extract_Trial_Aligned_Activity_Continous_Retinotopy_From_SVT.extract_trial_aligned_activity(base_directory)

    # Create Trial Averaged Movie
    Create_Sweep_Aligned_Movie.create_activity_video(base_directory, "Horizontal_Sweep", video_downsampled=False)
    Create_Sweep_Aligned_Movie.create_activity_video(base_directory, "Vertical_Sweep", video_downsampled=False)

    # Perform Fourrier Analysis
    #Continous_Retinotopy_Fourier_Analysis.perform_fourier_analysis(base_directory)
    Continous_Retinotopy_Fourier_Analysis_SVT.perform_fourier_analysis(base_directory)