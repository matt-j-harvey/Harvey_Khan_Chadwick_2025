
def flatten_nested_list(nested_list):
    flat_list = []
    for mouse in nested_list:
        for session in mouse:
            flat_list.append(session)
    return flat_list



neurexin_learning_session_list = [

    [r"NRXN71.2A/2020_11_13_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_14_Discrimination_Imaging",
     #r"NRXN71.2A/2020_11_15_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_16_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_17_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_19_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_21_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_23_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_25_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_27_Discrimination_Imaging",
     r"NRXN71.2A/2020_11_29_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_01_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_03_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_05_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_07_Discrimination_Imaging",
     r"NRXN71.2A/2020_12_09_Discrimination_Imaging"],

    [r"NXAK4.1A/2021_02_02_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_04_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_06_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_08_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_10_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_12_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_14_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_16_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_18_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_23_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_25_Discrimination_Imaging",
     r"NXAK4.1A/2021_02_27_Discrimination_Imaging",
     r"NXAK4.1A/2021_03_01_Discrimination_Imaging",
     r"NXAK4.1A/2021_03_03_Discrimination_Imaging",
     r"NXAK4.1A/2021_03_05_Discrimination_Imaging"],

    [r"NXAK10.1A/2021_04_30_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_02_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_04_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_06_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_08_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_10_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_12_Discrimination_Imaging",
     r"NXAK10.1A/2021_05_14_Discrimination_Imaging"],


    [r"NXAK16.1B/2021_04_30_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_02_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_04_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_06_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_08_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_10_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_12_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_14_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_16_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_18_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_20_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_22_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_24_Discrimination_Imaging",
     r"NXAK16.1B/2021_05_26_Discrimination_Imaging",
     r"NXAK16.1B/2021_06_04_Discrimination_Imaging",
     r"NXAK16.1B/2021_06_15_Discrimination_Imaging"],

    [r"NXAK20.1B/2021_09_28_Discrimination_Imaging",
     r"NXAK20.1B/2021_09_30_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_02_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_04_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_06_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_09_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_11_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_13_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_15_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_17_Discrimination_Imaging",
     r"NXAK20.1B/2021_10_19_Discrimination_Imaging"],

    [r"NXAK24.1C/2021_09_20_Discrimination_Imaging",
     r"NXAK24.1C/2021_09_22_Discrimination_Imaging",
     r"NXAK24.1C/2021_09_24_Discrimination_Imaging",
     r"NXAK24.1C/2021_09_26_Discrimination_Imaging",
     r"NXAK24.1C/2021_09_28_Discrimination_Imaging",
     r"NXAK24.1C/2021_09_30_Discrimination_Imaging",
     r"NXAK24.1C/2021_10_02_Discrimination_Imaging",
     r"NXAK24.1C/2021_10_04_Discrimination_Imaging",
     r"NXAK24.1C/2021_10_06_Discrimination_Imaging"],

]



control_learning_session_list = [

    [r"NRXN78.1D/2020_11_14_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_15_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_16_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_17_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_19_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_21_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_23_Discrimination_Imaging",
     r"NRXN78.1D/2020_11_25_Discrimination_Imaging"],

    [r"NRXN78.1A/2020_11_14_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_15_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_16_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_17_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_19_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_21_Discrimination_Imaging",
     r"NRXN78.1A/2020_11_24_Discrimination_Imaging"],

    [r"NXAK4.1B/2021_02_04_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_06_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_08_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_10_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_12_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_14_Discrimination_Imaging",
     r"NXAK4.1B/2021_02_22_Discrimination_Imaging"],

    [r"NXAK7.1B/2021_02_01_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_03_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_05_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_07_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_09_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_11_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_13_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_15_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_17_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_19_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_22_Discrimination_Imaging",
     r"NXAK7.1B/2021_02_24_Discrimination_Imaging"],

    [r"NXAK14.1A/2021_04_29_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_01_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_03_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_05_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_07_Discrimination_Imaging",
     r"NXAK14.1A/2021_05_09_Discrimination_Imaging"],

    [r"NXAK22.1A/2021_09_25_Discrimination_Imaging",
     r"NXAK22.1A/2021_09_29_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_01_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_03_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_05_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_07_Discrimination_Imaging",
     r"NXAK22.1A/2021_10_08_Discrimination_Imaging"],

]





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
     r"NXAK24.1C/2021_10_14_Switching_Imaging",
     r"NXAK24.1C/2021_10_20_Switching_Imaging",
     r"NXAK24.1C/2021_10_26_Switching_Imaging",
     r"NXAK24.1C/2021_11_05_Transition_Imaging",
     r"NXAK24.1C/2021_11_08_Transition_Imaging",
     r"NXAK24.1C/2021_11_10_Transition_Imaging"],


]


neurexin_pre_learning_list = [

    [
    "NRXN71.2A/2020_11_14_Discrimination_Imaging",
    #"NRXN71.2A/2020_11_15_Discrimination_Imaging",
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


control_post_learning_discrimination = [
    [r"NRXN78.1D/2020_11_25_Discrimination_Imaging"],
    [r"NRXN78.1A/2020_11_24_Discrimination_Imaging"],
    [r"NXAK4.1B/2021_02_22_Discrimination_Imaging"],
    [r"NXAK7.1B/2021_02_24_Discrimination_Imaging"],
    [r"NXAK14.1A/2021_05_09_Discrimination_Imaging"],
    [r"NXAK22.1A/2021_10_08_Discrimination_Imaging"]
]




# Days 1,2,3
neurexin_early_list = [
    [r"NRXN71.2A\2020_11_14_Discrimination_Imaging"], # Day 2
    [r"NXAK4.1A\2021_02_04_Discrimination_Imaging"], # Day 3
    [r"NXAK10.1A\2021_05_02_Discrimination_Imaging"], # Day 3
    [r"NXAK16.1B\2021_05_02_Discrimination_Imaging"], # Day 3
    [r"NXAK20.1B\2021_09_30_Discrimination_Imaging"], # Day 3
    [r"NXAK24.1C\2021_09_22_Discrimination_Imaging"], # Day 3
]






# Days 4,5,6
neurexin_mid_list  = [

    [r"NRXN71.2A\2020_11_16_Discrimination_Imaging",  # Day 4,
     r"NRXN71.2A\2020_11_16_Discrimination_Imaging"], # Day 5

    [r"NXAK4.1A\2021_02_06_Discrimination_Imaging"],  # Day 5

    [r"NXAK10.1A\2021_05_04_Discrimination_Imaging"], # Day 5

    [r"NXAK16.1B\2021_05_04_Discrimination_Imaging"], # Day 5

    [r"NXAK20.1B\2021_10_02_Discrimination_Imaging"], # Day 5

    [r"NXAK24.1C\2021_09_24_Discrimination_Imaging"], # Day 5
]

# Days 7,8,9
neurexin_late_list = [

    [r"NRXN71.2A\2020_11_19_Discrimination_Imaging", # NRXN71.2A Day 7
     r"NRXN71.2A\2020_11_21_Discrimination_Imaging"], # NRXN71.2A Day 9

    [r"NXAK4.1A\2021_02_08_Discrimination_Imaging", # NXAK4.1A Day 7
     r"NXAK4.1A\2021_02_10_Discrimination_Imaging"], # NXAK4.1A Day 9

    [r"NXAK10.1A\2021_05_06_Discrimination_Imaging", # NXAK10.1A Day 7
     r"NXAK10.1A\2021_05_08_Discrimination_Imaging"], # NXAK10.1A Day 9

    [r"NXAK16.1B\2021_05_06_Discrimination_Imaging", # NXAK16.1B Day 7
     r"NXAK16.1B\2021_05_08_Discrimination_Imaging"], # NXAK16.1B Day 9

    [r"NXAK20.1B\2021_10_04_Discrimination_Imaging", # NXAK20.1B Day 7
     r"NXAK20.1B\2021_10_06_Discrimination_Imaging"], # NXAK20.1B Day 9

    [r"NXAK24.1C\2021_09_26_Discrimination_Imaging", # NXAK24.1C Day 7
     r"NXAK24.1C\2021_09_28_Discrimination_Imaging"] # NXAK24.1C Day 9
]



# Day 1, 2, 3
control_early_list = [

    [r"NRXN78.1A\2020_11_15_Discrimination_Imaging",  # Day 1
     r"NRXN78.1A\2020_11_15_Discrimination_Imaging",  # Day 2
     r"NRXN78.1A\2020_11_16_Discrimination_Imaging"], # Day 3

    [r"NRXN78.1D\2020_11_15_Discrimination_Imaging",  # Day 2
     r"NRXN78.1D\2020_11_16_Discrimination_Imaging"], # Day 3

    [r"NXAK4.1B\2021_02_04_Discrimination_Imaging",    # Day 1
     r"NXAK4.1B\2021_02_06_Discrimination_Imaging"],  # Day 3

    [r"NXAK7.1B\2021_02_01_Discrimination_Imaging",    # Day 1
     r"NXAK7.1B\2021_02_03_Discrimination_Imaging"],  # Day 3

    [r"NXAK14.1A\2021_04_29_Discrimination_Imaging",   # Day 1
     r"NXAK14.1A\2021_05_01_Discrimination_Imaging"], # Day 3

    [r"NXAK22.1A\2021_09_25_Discrimination_Imaging"], # Day 1

]

# Day 4, 5, 6
control_mid_list = [

    [r"NRXN78.1A\2020_11_17_Discrimination_Imaging", # NRXN78.1A Day 4
     r"NRXN78.1A\2020_11_19_Discrimination_Imaging"], # NRXN78.1A Day 6

    [r"NRXN78.1D\2020_11_17_Discrimination_Imaging", # NRXN78.1D Day 4
     r"NRXN78.1D\2020_11_19_Discrimination_Imaging"], # NRXN78.1D Day 6

    [r"NXAK4.1B\2021_02_08_Discrimination_Imaging"], # NXAK4.1B Day 5

    [r"NXAK7.1B\2021_02_05_Discrimination_Imaging"], # NXAK7.1B Day 5

    [r"NXAK14.1A\2021_05_03_Discrimination_Imaging"], # NXAK14.1A Day 5

    [r"NXAK22.1A\2021_09_29_Discrimination_Imaging"], # NXAK22.1A Day 5
]


# Day 7, 8, 9
control_late_list = [

    [r"NRXN78.1A\2020_11_21_Discrimination_Imaging"],   # Day 8

    [r"NRXN78.1D\2020_11_21_Discrimination_Imaging"],   # Day 8

    [r"NXAK4.1B\2021_02_10_Discrimination_Imaging",     # Day 7
     r"NXAK4.1B\2021_02_12_Discrimination_Imaging"],    # Day 9

    [r"NXAK7.1B\2021_02_07_Discrimination_Imaging",     # Day 7
     r"NXAK7.1B\2021_02_09_Discrimination_Imaging"],    # Day 9

    [r"NXAK14.1A\2021_05_05_Discrimination_Imaging",    # Day 7
     r"NXAK14.1A\2021_05_07_Discrimination_Imaging"],   # Day 9

    [r"NXAK22.1A\2021_10_01_Discrimination_Imaging",    # Day 7
     r"NXAK22.1A\2021_10_03_Discrimination_Imaging"],   # Day 9
]