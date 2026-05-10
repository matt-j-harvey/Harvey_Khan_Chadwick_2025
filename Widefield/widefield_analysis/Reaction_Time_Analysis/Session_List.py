
def flatten_nested_list(nested_list):
    flat_list = []
    for mouse in nested_list:
        for session in mouse:
            flat_list.append(session)
    return flat_list


nested_session_list_with_root = [

    ["/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_08_22_Switching_V1_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_08_23_Switching_PPC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_08_26_Switching_ProxM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_08_28_Switching_PM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_08_30_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_09_03_Switching_RSC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_09_11_Switching_ALM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_09_17_Switching_SS_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC11.1C/2024_09_20_Switching_V1_Pre_03"],

    ["/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_11_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_16_Switching_V1_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_17_Switching_PPC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_18_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_19_Switching_SS_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_20_Switching_PM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_25_Switching_RSC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_28_Switching_ALM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2A/2024_09_29_Switching_ProxM_Pre_03"],

    ["/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_10_Switching_V1_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_11_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_17_Switching_RSC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_18_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_19_Switching_SS_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_20_Switching_PPC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_24_Switching_PM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_26_Switching_ALM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.2B/2024_09_28_Switching_ProxM_Pre_03"],

    ["/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_03_Switching_V1_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_05_Switching_PPC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_09_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_10_Switching_RSC_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_12_Switching_SS_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_16_Switching_ALM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_17_Switching_PM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_18_Switching_V1_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_19_Switching_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_20_Switching_MM_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_23_Switching_SS_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_28_Switching_V1_Pre_03",
     "/media/matthew/External HDD/Closed_Loop_Opto/KPGC12.3B/2024_09_29_Switching_ProxM_Pre_03"],

    [
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_07_07_Switch_V1_1F_03_Pre",
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_07_13_Switch_MM_1F_03_Pre",
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_07_18_Switch_ALM_1F_03_Pre",
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_07_20_Switch_RSC_1F_03_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_07_27_Switch_MM_1f_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_01_Switch_V1_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_03_Switch_RSC_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_08_Switch_PM_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_10_Switch_ALM_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_22_Switch_V1_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_24_Switch_PPC_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC1.3A/2023_08_28_Switch_BC_1F_04_1S_Pre"],

    ["/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_08_24_Switch_V1_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre"],

    ["/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_08_25_Switch_V1_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_08_28_Switch_ALM_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_08_30_Switch_PM_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_09_01_Switch_V1_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_09_04_Switch_MM_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_09_06_Switch_BC_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_09_08_Switch_RSC_1F_04_1S_Pre",
     "/media/matthew/Expansion1/Cortex_Wide_Opto/Control_Mice/KPGC7.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre"],

    [  # "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_03_Switch_V1_1F_03_Pre",
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_06_Switch_MM_1F_03_Pre",
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_14_Switch_ALM_1F_03_Pre",
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_19_Switch_RSC_1F_03_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_28_Switch_V1_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_08_07_Switch_SS_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre"],

    [
        # "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_07_27_Switch_MM_1F_04_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_08_18_Switch_ALM_1F_06_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_08_21_Switch_BC_1F_06_1S_Pre",
        "/media/matthew/Expansion1/Cortex_Wide_Opto/Opto_Mice/KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre"],
]





nested_session_list = [

    ["KPGC11.1C/2024_08_22_Switching_V1_Pre_03",
     "KPGC11.1C/2024_08_23_Switching_PPC_Pre_03",
     "KPGC11.1C/2024_08_26_Switching_ProxM_Pre_03",
     "KPGC11.1C/2024_08_28_Switching_PM_Pre_03",
     "KPGC11.1C/2024_08_30_Switching_MM_Pre_03",
     "KPGC11.1C/2024_09_03_Switching_RSC_Pre_03",
     "KPGC11.1C/2024_09_11_Switching_ALM_Pre_03",
     "KPGC11.1C/2024_09_17_Switching_SS_Pre_03",
     "KPGC11.1C/2024_09_20_Switching_V1_Pre_03"],

    ["KPGC12.2A/2024_09_11_Switching_MM_Pre_03",
     "KPGC12.2A/2024_09_16_Switching_V1_Pre_03",
     "KPGC12.2A/2024_09_17_Switching_PPC_Pre_03",
     "KPGC12.2A/2024_09_18_Switching_MM_Pre_03",
     "KPGC12.2A/2024_09_19_Switching_SS_Pre_03",
     "KPGC12.2A/2024_09_20_Switching_PM_Pre_03",
     "KPGC12.2A/2024_09_25_Switching_RSC_Pre_03",
     "KPGC12.2A/2024_09_28_Switching_ALM_Pre_03",
     "KPGC12.2A/2024_09_29_Switching_ProxM_Pre_03"],

    ["KPGC12.2B/2024_09_10_Switching_V1_Pre_03",
     "KPGC12.2B/2024_09_11_Switching_MM_Pre_03",
     "KPGC12.2B/2024_09_17_Switching_RSC_Pre_03",
     "KPGC12.2B/2024_09_18_Switching_MM_Pre_03",
     "KPGC12.2B/2024_09_19_Switching_SS_Pre_03",
     "KPGC12.2B/2024_09_20_Switching_PPC_Pre_03",
     "KPGC12.2B/2024_09_24_Switching_PM_Pre_03",
     "KPGC12.2B/2024_09_26_Switching_ALM_Pre_03",
     "KPGC12.2B/2024_09_28_Switching_ProxM_Pre_03"],

    ["KPGC12.3B/2024_09_03_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_05_Switching_PPC_Pre_03",
     "KPGC12.3B/2024_09_09_Switching_MM_Pre_03",
     "KPGC12.3B/2024_09_10_Switching_RSC_Pre_03",
     "KPGC12.3B/2024_09_12_Switching_SS_Pre_03",
     "KPGC12.3B/2024_09_16_Switching_ALM_Pre_03",
     "KPGC12.3B/2024_09_17_Switching_PM_Pre_03",
     "KPGC12.3B/2024_09_18_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_19_Switching_Pre_03",
     "KPGC12.3B/2024_09_20_Switching_MM_Pre_03",
     "KPGC12.3B/2024_09_23_Switching_SS_Pre_03",
     "KPGC12.3B/2024_09_28_Switching_V1_Pre_03",
     "KPGC12.3B/2024_09_29_Switching_ProxM_Pre_03"],

    [#"KPGC1.3A/2023_07_07_Switch_V1_1F_03_Pre",
     #"KPGC1.3A/2023_07_13_Switch_MM_1F_03_Pre",
     #"KPGC1.3A/2023_07_18_Switch_ALM_1F_03_Pre",
     #"KPGC1.3A/2023_07_20_Switch_RSC_1F_03_Pre",
     "KPGC1.3A/2023_07_27_Switch_MM_1f_04_1S_Pre",
     "KPGC1.3A/2023_08_01_Switch_V1_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_03_Switch_RSC_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_08_Switch_PM_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_10_Switch_ALM_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_22_Switch_V1_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_24_Switch_PPC_1F_04_1S_Pre",
     "KPGC1.3A/2023_08_28_Switch_BC_1F_04_1S_Pre"],

    ["KPGC3.4A/2023_08_24_Switch_V1_1F_04_1S_Pre",
     "KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
     "KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
     "KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre"],

    ["KPGC7.4A/2023_08_25_Switch_V1_1F_04_1S_Pre",
     "KPGC7.4A/2023_08_28_Switch_ALM_1F_04_1S_Pre",
     "KPGC7.4A/2023_08_30_Switch_PM_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_01_Switch_V1_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_04_Switch_MM_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_06_Switch_BC_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_08_Switch_RSC_1F_04_1S_Pre",
     "KPGC7.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre"],

    [#"KPGC3.3E/2023_07_03_Switch_V1_1F_03_Pre",
     #"KPGC3.3E/2023_07_06_Switch_MM_1F_03_Pre",
     #"KPGC3.3E/2023_07_14_Switch_ALM_1F_03_Pre",
     #"KPGC3.3E/2023_07_19_Switch_RSC_1F_03_Pre",
     "KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_28_Switch_V1_1F_04_1S_Pre",
     "KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
     "KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
     "KPGC3.3E/2023_08_07_Switch_SS_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
     "KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre"],

    [#"KPGC6.2E/2023_07_27_Switch_MM_1F_04_1S_Pre",
     "KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_18_Switch_ALM_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_21_Switch_BC_1F_06_1S_Pre",
     "KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre"],
]


control_post_learning_discrimination = [
    [r"NRXN78.1A/2020_11_24_Discrimination_Imaging"],
    [r"NRXN78.1D/2020_11_25_Discrimination_Imaging"],
    [r"NXAK4.1B/2021_02_22_Discrimination_Imaging"],
    [r"NXAK7.1B/2021_02_24_Discrimination_Imaging"],
    [r"NXAK14.1A/2021_05_09_Discrimination_Imaging"],
    [r"NXAK22.1A/2021_10_08_Discrimination_Imaging"]
]


neurexin_post_learning_discrimination = [
    [r"NRXN71.2A/2020_12_09_Discrimination_Imaging"],
    [r"NXAK4.1A/2021_03_05_Discrimination_Imaging"],
    [r"NXAK10.1A/2021_05_14_Discrimination_Imaging"],
    [r"NXAK16.1B/2021_06_15_Discrimination_Imaging"],
    [r"NXAK20.1B/2021_10_19_Discrimination_Imaging"],
    [r"NXAK24.1C/2021_10_06_Discrimination_Imaging"],
]



control_pre_learning = [

    ["NRXN78.1D/2020_11_15_Discrimination_Imaging"],

    ["NRXN78.1A/2020_11_15_Discrimination_Imaging"],

    ["NXAK4.1B/2021_02_06_Discrimination_Imaging",
     "NXAK4.1B/2021_02_08_Discrimination_Imaging",
     "NXAK4.1B/2021_02_10_Discrimination_Imaging"],

    ["NXAK7.1B/2021_02_03_Discrimination_Imaging",
     "NXAK7.1B/2021_02_05_Discrimination_Imaging",
     "NXAK7.1B/2021_02_07_Discrimination_Imaging",
     "NXAK7.1B/2021_02_09_Discrimination_Imaging"],

    ["NXAK14.1A/2021_05_01_Discrimination_Imaging"],
    ["NXAK14.1A/2021_05_03_Discrimination_Imaging"],

    ["NXAK22.1A/2021_09_25_Discrimination_Imaging"]
]



control_intermediate_learning = [

    ["NRXN78.1D/2020_11_16_Discrimination_Imaging",
     "NRXN78.1D/2020_11_17_Discrimination_Imaging",
     "NRXN78.1D/2020_11_19_Discrimination_Imaging"],

    ["NRXN78.1A/2020_11_16_Discrimination_Imaging"],

    ["NXAK4.1B/2021_02_12_Discrimination_Imaging"],

    ["NXAK7.1B/2021_02_11_Discrimination_Imaging",
     "NXAK7.1B/2021_02_15_Discrimination_Imaging",
     "NXAK7.1B/2021_02_17_Discrimination_Imaging",
     "NXAK7.1B/2021_02_19_Discrimination_Imaging",
     "NXAK7.1B/2021_02_22_Discrimination_Imaging"],

    ["NXAK14.1A/2021_05_03_Discrimination_Imaging"],

    ["NXAK22.1A/2021_09_29_Discrimination_Imaging",
     "NXAK22.1A/2021_10_01_Discrimination_Imaging",
     "NXAK22.1A/2021_10_03_Discrimination_Imaging",
     "NXAK22.1A/2021_10_05_Discrimination_Imaging"],

]



neurexin_pre_learning = [

    [
    "NRXN71.2A/2020_11_14_Discrimination_Imaging",
    "NRXN71.2A/2020_11_16_Discrimination_Imaging",
    "NRXN71.2A/2020_11_17_Discrimination_Imaging",
    "NRXN71.2A/2020_11_19_Discrimination_Imaging",
    "NRXN71.2A/2020_11_21_Discrimination_Imaging",
    "NRXN71.2A/2020_11_23_Discrimination_Imaging",
    "NRXN71.2A/2020_11_25_Discrimination_Imaging",
    "NRXN71.2A/2020_11_27_Discrimination_Imaging",
    "NRXN71.2A/2020_11_29_Discrimination_Imaging",
    "NRXN71.2A/2020_12_01_Discrimination_Imaging",
    "NRXN71.2A/2020_12_03_Discrimination_Imaging",
    ],

    [
    "NXAK4.1A/2021_02_04_Discrimination_Imaging",
    "NXAK4.1A/2021_02_06_Discrimination_Imaging",
    ],

    [
    "NXAK10.1A/2021_05_04_Discrimination_Imaging",
    "NXAK10.1A/2021_05_06_Discrimination_Imaging",
    "NXAK10.1A/2021_05_08_Discrimination_Imaging",
    ],

    [
    "NXAK16.1B/2021_05_02_Discrimination_Imaging",
    "NXAK16.1B/2021_05_04_Discrimination_Imaging",
    "NXAK16.1B/2021_05_06_Discrimination_Imaging",
    "NXAK16.1B/2021_05_08_Discrimination_Imaging",
    "NXAK16.1B/2021_05_10_Discrimination_Imaging",
    "NXAK16.1B/2021_05_12_Discrimination_Imaging",
    "NXAK16.1B/2021_05_14_Discrimination_Imaging",
    "NXAK16.1B/2021_05_16_Discrimination_Imaging",
    "NXAK16.1B/2021_05_18_Discrimination_Imaging",
    ],

    [
    "NXAK20.1B/2021_09_30_Discrimination_Imaging",
    "NXAK20.1B/2021_10_02_Discrimination_Imaging",
    ],

    [
    "NXAK24.1C/2021_09_22_Discrimination_Imaging",
    "NXAK24.1C/2021_09_24_Discrimination_Imaging",
    "NXAK24.1C/2021_09_26_Discrimination_Imaging",
    ],

]




neurexin_intermediate_learning = [

    ["NRXN71.2A/2020_12_03_Discrimination_Imaging"],

    ["NXAK4.1A/2021_02_08_Discrimination_Imaging",
     "NXAK4.1A/2021_02_10_Discrimination_Imaging",
     "NXAK4.1A/2021_02_12_Discrimination_Imaging"],

    ["NXAK10.1A/2021_05_02_Discrimination_Imaging",
     "NXAK10.1A/2021_05_10_Discrimination_Imaging"],

    ["NXAK16.1B/2021_05_20_Discrimination_Imaging",
     "NXAK16.1B/2021_05_22_Discrimination_Imaging",
     "NXAK16.1B/2021_05_24_Discrimination_Imaging",
     "NXAK16.1B/2021_05_26_Discrimination_Imaging",
     "NXAK16.1B/2021_06_04_Discrimination_Imaging"],

    ["NXAK20.1B/2021_10_04_Discrimination_Imaging",
     "NXAK20.1B/2021_10_06_Discrimination_Imaging",
     "NXAK20.1B/2021_10_09_Discrimination_Imaging"],

    ["NXAK24.1C/2021_09_28_Discrimination_Imaging"]

]



flat_session_list = flatten_nested_list(nested_session_list)
flat_session_list_with_root = flatten_nested_list(nested_session_list_with_root)