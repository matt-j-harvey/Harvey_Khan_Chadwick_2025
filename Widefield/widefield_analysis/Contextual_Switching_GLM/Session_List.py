
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

    [
    #"KPGC11.1C/2024_08_22_Switching_V1_Pre_03",
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
     #"KPGC12.3B/2024_09_18_Switching_V1_Pre_03",
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
     #"KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre", # - Facecam too short
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





wildtype_switching_sessions = [

    [r"NRXN78.1A/2020_11_28_Switching_Imaging",
     r"NRXN78.1A/2020_12_05_Switching_Imaging",
     r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_29_Switching_Imaging",
     r"NRXN78.1D/2020_12_05_Switching_Imaging",
     r"NRXN78.1D/2020_12_07_Switching_Imaging"],

    [r"NXAK4.1B/2021_03_02_Switching_Imaging",
     r"NXAK4.1B/2021_03_04_Switching_Imaging",
     r"NXAK4.1B/2021_03_06_Switching_Imaging",
     r"NXAK4.1B/2021_04_02_Transition_Imaging",
     r"NXAK4.1B/2021_04_08_Transition_Imaging",
     r"NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"NXAK7.1B/2021_02_26_Switching_Imaging",
     r"NXAK7.1B/2021_02_28_Switching_Imaging",
     r"NXAK7.1B/2021_03_23_Transition_Imaging",
     r"NXAK7.1B/2021_03_31_Transition_Imaging",
     r"NXAK7.1B/2021_04_02_Transition_Imaging"],

    [r"NXAK14.1A/2021_05_21_Switching_Imaging",
     r"NXAK14.1A/2021_05_23_Switching_Imaging",
     r"NXAK14.1A/2021_06_11_Switching_Imaging",
     r"NXAK14.1A/2021_06_13_Transition_Imaging",
     r"NXAK14.1A/2021_06_15_Transition_Imaging",
     r"NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"NXAK22.1A/2021_10_14_Switching_Imaging",
     r"NXAK22.1A/2021_10_20_Switching_Imaging",
     r"NXAK22.1A/2021_10_22_Switching_Imaging",
     r"NXAK22.1A/2021_10_29_Transition_Imaging",
     r"NXAK22.1A/2021_11_03_Transition_Imaging",
     r"NXAK22.1A/2021_11_05_Transition_Imaging"]

]




neurexin_switching_sessions = [

    [r"NRXN71.2A/2020_12_13_Switching_Imaging",
     r"NRXN71.2A/2020_12_15_Switching_Imaging",
     r"NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"NXAK4.1A/2021_03_31_Switching_Imaging",
     r"NXAK4.1A/2021_04_02_Switching_Imaging",
     r"NXAK4.1A/2021_04_04_Switching_Imaging",
     r"NXAK4.1A/2021_04_08_Transition_Imaging",
     r"NXAK4.1A/2021_04_10_Transition_Imaging",
     r"NXAK4.1A/2021_04_12_Transition_Imaging"],

    [r"NXAK10.1A/2021_05_20_Switching_Imaging",
     r"NXAK10.1A/2021_05_22_Switching_Imaging",
     r"NXAK10.1A/2021_05_24_Switching_Imaging",
     r"NXAK10.1A/2021_06_14_Transition_Imaging",
     r"NXAK10.1A/2021_06_16_Transition_Imaging",
     r"NXAK10.1A/2021_06_18_Transition_Imaging"],

    [r"NXAK16.1B/2021_06_17_Switching_Imaging",
     r"NXAK16.1B/2021_06_19_Switching_Imaging",
     r"NXAK16.1B/2021_06_23_Switching_Imaging",
     r"NXAK16.1B/2021_06_30_Transition_Imaging",
     r"NXAK16.1B/2021_07_06_Transition_Imaging",
     r"NXAK16.1B/2021_07_08_Transition_Imaging"],

    [r"NXAK20.1B/2021_11_15_Switching_Imaging",
     r"NXAK20.1B/2021_11_17_Switching_Imaging",
     r"NXAK20.1B/2021_11_19_Switching_Imaging",
     r"NXAK20.1B/2021_11_22_Transition_Imaging",
     r"NXAK20.1B/2021_11_24_Transition_Imaging",
     r"NXAK20.1B/2021_11_26_Transition_Imaging"],

    [r"NXAK24.1C/2021_10_14_Switching_Imaging",
     r"NXAK24.1C/2021_10_20_Switching_Imaging",
     r"NXAK24.1C/2021_10_26_Switching_Imaging",
     r"NXAK24.1C/2021_11_05_Transition_Imaging",
     r"NXAK24.1C/2021_11_08_Transition_Imaging",
     r"NXAK24.1C/2021_11_10_Transition_Imaging"],


]