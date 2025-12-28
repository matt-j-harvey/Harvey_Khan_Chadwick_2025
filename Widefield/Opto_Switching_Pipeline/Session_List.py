
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
     #"KPGC12.2B/2024_09_11_Switching_MM_Pre_03",
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


v1_control_session_list = [ "KPGC12.3B/2024_09_28_Switching_V1_Pre_03",
                            "KPGC1.3A/2023_08_22_Switch_V1_1F_04_1S_Pre",
                            "KPGC3.4A/2023_08_24_Switch_V1_1F_04_1S_Pre",
                            "KPGC7.4A/2023_09_01_Switch_V1_1F_04_1S_Pre"]

v1_opto_session_list = [
                            "KPGC11.1C/2024_09_20_Switching_V1_Pre_03",
                            "KPGC12.2A/2024_09_16_Switching_V1_Pre_03",
                            "KPGC12.2B/2024_09_10_Switching_V1_Pre_03",
                            "KPGC3.3E/2023_08_21_Switch_V1_1F_04_1S_Pre",
                            "KPGC6.2E/2023_08_07_Switch_V1_1F_06_1S_Pre"]



ppc_opto_session_list = [
                            "KPGC12.2A/2024_09_17_Switching_PPC_Pre_03",
                            "KPGC12.2B/2024_09_20_Switching_PPC_Pre_03",
                            "KPGC11.1C/2024_08_23_Switching_PPC_Pre_03",
                            "KPGC3.3E/2023_08_09_Switch_PPC_1F_04_1S_Pre",
                            "KPGC6.2E/2023_08_23_Switch_PPC_1F_06_1S_Pre"]

ppc_control_session_list = [
                            "KPGC1.3A/2023_08_24_Switch_PPC_1F_04_1S_Pre",
                            "KPGC3.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
                            "KPGC7.4A/2023_09_11_Switch_PPC_1F_04_1S_Pre",
                            "KPGC12.3B/2024_09_05_Switching_PPC_Pre_03"]

ss_opto_session_list = [
                            "KPGC12.2A/2024_09_19_Switching_SS_Pre_03",
                            "KPGC12.2B/2024_09_19_Switching_SS_Pre_03",
                            "KPGC11.1C/2024_09_17_Switching_SS_Pre_03",
                            "KPGC6.2E/2023_08_21_Switch_BC_1F_06_1S_Pre",
                            "KPGC3.3E/2023_08_18_Switch_BC_1F_04_1S_Pre",
]


ss_control_session_list = [
                    "KPGC1.3A/2023_08_28_Switch_BC_1F_04_1S_Pre",
                    "KPGC3.4A/2023_09_08_Switch_BC_1F_04_1S_Pre",
                    "KPGC7.4A/2023_09_06_Switch_BC_1F_04_1S_Pre",
                    "KPGC12.3B/2024_09_23_Switching_SS_Pre_03"
]



mm_opto_session_list = [
    "KPGC12.2A/2024_09_18_Switching_MM_Pre_03",
    "KPGC12.2B/2024_09_18_Switching_MM_Pre_03",
    "KPGC11.1C/2024_08_30_Switching_MM_Pre_03",
    "KPGC6.2E/2023_07_31_Switch_MM_1F_06_1S_Pre",
    "KPGC3.3E/2023_07_21_Switch_MM_1F_04_1S_Pre",
]

mm_control_session_list = [
    "KPGC1.3A/2023_07_27_Switch_MM_1f_04_1S_Pre",
    "KPGC3.4A/2023_09_06_Switch_MM_1F_04_1s_Pre",
    "KPGC7.4A/2023_09_04_Switch_MM_1F_04_1S_Pre",
    "KPGC12.3B/2024_09_20_Switching_MM_Pre_03",
]

alm_opto_session_list = [
    "KPGC11.1C/2024_09_11_Switching_ALM_Pre_03",
    "KPGC12.2A/2024_09_28_Switching_ALM_Pre_03",
    "KPGC12.2B/2024_09_26_Switching_ALM_Pre_03",
    "KPGC6.2E/2023_08_18_Switch_ALM_1F_06_1S_Pre",
    "KPGC3.3E/2023_07_31_Switch_ALM_1F_04_Pre",
]


alm_control_session_list = [
    "KPGC1.3A/2023_08_10_Switch_ALM_1F_04_1S_Pre",
    "KPGC3.4A/2023_08_29_Switch_ALM_1F_04_1S_Pre",
    "KPGC7.4A/2023_08_28_Switch_ALM_1F_04_1S_Pre",
    "KPGC12.3B/2024_09_16_Switching_ALM_Pre_03",
]

rsc_opto_session_list = [
    "KPGC11.1C/2024_09_03_Switching_RSC_Pre_03",
    "KPGC12.2A/2024_09_25_Switching_RSC_Pre_03",
    "KPGC12.2B/2024_09_17_Switching_RSC_Pre_03",
    "KPGC6.2E/2023_08_02_Switch_RSC_1F_06_1S_Pre",
    "KPGC3.3E/2023_07_26_Switch_RSC_1F_04_1S_Pre",
]

rsc_control_session_list = [
    "KPGC1.3A/2023_08_03_Switch_RSC_1F_04_1S_Pre",
    "KPGC3.4A/2023_09_04_Switch_RSC_1F_04_1S_Pre",
    "KPGC7.4A/2023_09_08_Switch_RSC_1F_04_1S_Pre",
    "KPGC12.3B/2024_09_10_Switching_RSC_Pre_03",
]

pm_opto_session_list = [
    "KPGC11.1C/2024_08_28_Switching_PM_Pre_03",
    "KPGC12.2A/2024_09_20_Switching_PM_Pre_03",
    "KPGC12.2B/2024_09_24_Switching_PM_Pre_03",
    "KPGC6.2E/2023_08_09_Switch_PM_1F_06_1S_Pre",
    "KPGC3.3E/2023_08_02_Switch_PM_1f_04_Pre",
]

pm_control_session_list = [
    "KPGC1.3A/2023_08_08_Switch_PM_1F_04_1S_Pre",
    "KPGC3.4A/2023_09_01_Switch_PM_1F_04_1S_Pre",
    "KPGC7.4A/2023_08_30_Switch_PM_1F_04_1S_Pre",
    "KPGC12.3B/2024_09_17_Switching_PM_Pre_03"
]
