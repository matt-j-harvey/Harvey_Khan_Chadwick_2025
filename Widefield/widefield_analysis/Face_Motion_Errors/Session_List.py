def flatten_nested_list(nested_list):
 flat_list = []
 for mouse in nested_list:
  for session in mouse:
   flat_list.append(session)
 return flat_list

control_all_post_learning = [

    [r"NRXN78.1A/2020_11_24_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_28_Switching_Imaging",
     r"NRXN78.1A/2020_12_05_Switching_Imaging",
     r"NRXN78.1A/2020_12_09_Switching_Imaging"],

    [r"NRXN78.1D/2020_11_25_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_29_Switching_Imaging",
     r"NRXN78.1D/2020_12_05_Switching_Imaging",
     r"NRXN78.1D/2020_12_07_Switching_Imaging"],

    [r"NXAK4.1B/2021_02_22_Discrimination_Imaging",
     r"NXAK4.1B/2021_03_02_Switching_Imaging",
     r"NXAK4.1B/2021_03_04_Switching_Imaging",
     r"NXAK4.1B/2021_03_06_Switching_Imaging",
     r"NXAK4.1B/2021_04_02_Transition_Imaging",
     r"NXAK4.1B/2021_04_08_Transition_Imaging",
     r"NXAK4.1B/2021_04_10_Transition_Imaging"],

    [r"NXAK7.1B/2021_02_24_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_26_Switching_Imaging",
     r"NXAK7.1B/2021_02_28_Switching_Imaging",
     r"NXAK7.1B/2021_03_23_Transition_Imaging",
     r"NXAK7.1B/2021_03_31_Transition_Imaging",
     r"NXAK7.1B/2021_04_02_Transition_Imaging"],

    [r"NXAK14.1A/2021_05_09_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_21_Switching_Imaging",
     r"NXAK14.1A/2021_05_23_Switching_Imaging",
     r"NXAK14.1A/2021_06_11_Switching_Imaging",
     r"NXAK14.1A/2021_06_13_Transition_Imaging",
     r"NXAK14.1A/2021_06_15_Transition_Imaging",
     r"NXAK14.1A/2021_06_17_Transition_Imaging"],

    [r"NXAK22.1A/2021_10_08_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_14_Switching_Imaging",
     r"NXAK22.1A/2021_10_20_Switching_Imaging",
     r"NXAK22.1A/2021_10_22_Switching_Imaging",
     r"NXAK22.1A/2021_10_29_Transition_Imaging",
     r"NXAK22.1A/2021_11_03_Transition_Imaging",
     r"NXAK22.1A/2021_11_05_Transition_Imaging"]

]




neurexin_all_post_learning = [

    [r"NRXN71.2A/2020_12_09_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_13_Switching_Imaging",
     r"NRXN71.2A/2020_12_15_Switching_Imaging",
     r"NRXN71.2A/2020_12_17_Switching_Imaging"],

    [r"NXAK4.1A/2021_03_05_Discrimination_Imaging",
     r"NXAK4.1A/2021_03_31_Switching_Imaging",
     r"NXAK4.1A/2021_04_02_Switching_Imaging",
     r"NXAK4.1A/2021_04_04_Switching_Imaging",
     r"NXAK4.1A/2021_04_08_Transition_Imaging",
     r"NXAK4.1A/2021_04_10_Transition_Imaging",
     r"NXAK4.1A/2021_04_12_Transition_Imaging"],

    [r"NXAK10.1A/2021_05_14_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_20_Switching_Imaging",
     r"NXAK10.1A/2021_05_22_Switching_Imaging",
     r"NXAK10.1A/2021_05_24_Switching_Imaging",
     r"NXAK10.1A/2021_06_14_Transition_Imaging",
     r"NXAK10.1A/2021_06_16_Transition_Imaging",
     r"NXAK10.1A/2021_06_18_Transition_Imaging"],

    [r"NXAK16.1B/2021_06_15_Discrimination_Imaging",
     r"NXAK16.1B/2021_06_17_Switching_Imaging",
     r"NXAK16.1B/2021_06_19_Switching_Imaging",
     r"NXAK16.1B/2021_06_23_Switching_Imaging",
     r"NXAK16.1B/2021_06_30_Transition_Imaging",
     r"NXAK16.1B/2021_07_06_Transition_Imaging",
     r"NXAK16.1B/2021_07_08_Transition_Imaging"],

    [r"NXAK20.1B/2021_10_19_Discrimination_Imaging",
     r"NXAK20.1B/2021_11_15_Switching_Imaging",
     r"NXAK20.1B/2021_11_17_Switching_Imaging",
     r"NXAK20.1B/2021_11_19_Switching_Imaging",
     r"NXAK20.1B/2021_11_22_Transition_Imaging",
     r"NXAK20.1B/2021_11_24_Transition_Imaging",
     r"NXAK20.1B/2021_11_26_Transition_Imaging"],

    [r"NXAK24.1C/2021_10_06_Discrimination_Imaging",
     #r"NXAK24.1C/2021_10_08_Switching_Imaging",
     r"NXAK24.1C/2021_10_14_Switching_Imaging",
     r"NXAK24.1C/2021_10_20_Switching_Imaging",
     r"NXAK24.1C/2021_10_26_Switching_Imaging",
     r"NXAK24.1C/2021_11_05_Transition_Imaging",
     r"NXAK24.1C/2021_11_08_Transition_Imaging",
     r"NXAK24.1C/2021_11_10_Transition_Imaging"],


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
