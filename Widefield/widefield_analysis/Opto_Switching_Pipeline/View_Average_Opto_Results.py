import os
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

import Opto_GLM_Utils

"""
We get opto modulation + control modulation

def get_modulation(visual_context_control, odour_context_control, visual_context_opto, odour_context_opto):

    # Get Light Effects
    visual_opto_effect = np.subtract(visual_context_opto, visual_context_control)
    odour_opto_effect = np.subtract(odour_context_opto, odour_context_control)

    # Get Opto Context Interaction
    opto_interaction = np.subtract(visual_opto_effect, odour_opto_effect)

    return opto_interaction


"""


def compare_regressors(output_root, start_window, stop_window, experiment_name, comparison_window_start, comparison_window_stop):

    # Load Group Coefs
    control_visual_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Visual_Context_Control" + "_group_coefs.npy"))
    control_visual_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Visual_Context_Light" + "_group_coefs.npy"))
    control_odour_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Odour_Context_Control" + "_group_coefs.npy"))
    control_odour_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Control", "Group_Coefs", "Odour_Context_Light" + "_group_coefs.npy"))

    opsin_visual_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Visual_Context_Control" + "_group_coefs.npy"))
    opsin_visual_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Visual_Context_Light" + "_group_coefs.npy"))
    opsin_odour_context_no_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Odour_Context_Control" + "_group_coefs.npy"))
    opsin_odour_context_red_light = np.load(os.path.join(output_root, "Group_Results", experiment_name + "_Opsin", "Group_Coefs", "Odour_Context_Light" + "_group_coefs.npy"))

    # Get Light Effects
    control_visual_context_light_effect = np.subtract(control_visual_context_red_light, control_visual_context_no_light)
    control_odour_context_light_effect = np.subtract(control_odour_context_red_light, control_odour_context_no_light)
    opsin_visual_context_light_effect = np.subtract(opsin_visual_context_red_light, opsin_visual_context_no_light)
    opsin_odour_context_light_effect = np.subtract(opsin_odour_context_red_light, opsin_odour_context_no_light)

    # Get Contextual Modulation Of Light Effect
    control_contextual_modulation = np.subtract(control_visual_context_light_effect, control_odour_context_light_effect)
    opsin_contextual_modulation = np.subtract(opsin_visual_context_light_effect, opsin_odour_context_light_effect)

    for mouse in control_contextual_modulation:
        plt.title("Mouse Control")
        plt.imshow(mouse)
        plt.show()

    for mouse in opsin_contextual_modulation:
        plt.title("Mouse Opsin")
        plt.imshow(mouse)
        plt.show()

    save_directory = os.path.join(output_root, "Group_Results", experiment_name, "Interaction_Comparison")

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    mean_control_contextual_modulation = np.mean(control_contextual_modulation, axis=0)
    mean_opsin_contextual_modulation = np.mean(opsin_contextual_modulation, axis=0)

    plt.title("mean_control_contextual_modulation")
    plt.imshow(mean_control_contextual_modulation)
    plt.show()

    plt.title("mean_opsin_contextual_modulation")
    plt.imshow(mean_opsin_contextual_modulation)
    plt.show()

    diff = np.subtract(mean_opsin_contextual_modulation, mean_control_contextual_modulation)
    plt.title("diff")
    plt.imshow(diff, cmap="bwr", vmin=-0.1, vmax=0.1)
    plt.show()

    t_stats, p_value = stats.ttest_ind(opsin_contextual_modulation, control_contextual_modulation, axis=0)
    print("p_value", np.shape(p_value))
    plt.title("p value")
    plt.imshow(p_value)
    plt.show()

    #Plotting_Functions.compare_regressors(control_contextual_modulation, opsin_contextual_modulation, save_directory, start_window, stop_window)