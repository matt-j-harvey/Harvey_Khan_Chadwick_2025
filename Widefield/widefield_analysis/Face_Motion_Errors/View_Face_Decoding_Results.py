import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import Session_List

def get_group_scores(session_list, output_root):

    group_score_list = []
    for mouse in session_list:
        mouse_score_list = []
        for session in mouse:
            score_filepath = os.path.join(output_root, session, "Decoding_Scores.npy")
            if os.path.isfile(score_filepath):
                session_scores = np.load(score_filepath)
                mouse_score_list.append(session_scores)

        if len(mouse_score_list) > 2:
            mouse_score_list = np.array(mouse_score_list)
            mouse_mean = np.mean(mouse_score_list, axis=0)
            group_score_list.append(mouse_mean)

    group_score_list = np.array(group_score_list)
    return group_score_list




def get_mean_sd(data):
    data_mean = np.mean(data, axis=0)
    data_sem = stats.sem(data, axis=0)
    lower_bound = np.subtract(data_mean, data_sem)
    upper_bound = np.add(data_mean, data_sem)

    return data_mean, lower_bound, upper_bound



def visualise_decoding_results(control_session_list, hom_session_list, control_output_root, hom_output_root, start_window, stop_window):

    # Load Control Scores
    control_scores = get_group_scores(control_session_list, control_output_root)
    hom_scores = get_group_scores(hom_session_list, hom_output_root)
    print("hom_scores", np.shape(hom_scores))

    # Get Mean and SEMs
    control_mean, control_lower_bound, control_upper_bound = get_mean_sd(control_scores)
    hom_mean, hom_lower_bound, hom_upper_bound = get_mean_sd(hom_scores)

    # Get X Values
    x_values = list(range(start_window, stop_window))
    x_values = np.multiply(x_values, 36)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.plot(x_values, control_mean, c='cornflowerblue')
    axis_1.fill_between(x=x_values, y1=control_lower_bound, y2=control_upper_bound, color='cornflowerblue', alpha=0.5)

    neurexin_pink = (0.76, 0.17, 0.99)
    axis_1.plot(x_values, hom_mean, c=neurexin_pink)
    axis_1.fill_between(x=x_values, y1=hom_lower_bound, y2=hom_upper_bound, color=neurexin_pink, alpha=0.5)


    axis_1.axvline(0, c='k', linestyle='dashed')

    axis_1.spines[['right', 'top']].set_visible(False)

    axis_1.set_title("FA decoding from face motion")
    axis_1.set_ylabel("Decoding Perforamance")

    plt.show()














# Set Directories
control_session_list = Session_List.control_intermediate_learning
control_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Controls"
control_output_root = r"C:\Face_Decoding\Controls"

hom_session_list = Session_List.neurexin_intermediate_learning
hom_data_root = r"C:\Users\matth\Dropbox\Harvey_Khan_Chadwick_2025\Neurexin_Widefield\Homs"
hom_output_root = r"C:\Face_Decoding\Homs"


# Select Analysis Details
frame_period = 36
start_window_ms = -2500
stop_window_ms = 1500
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)


# Run Pipeline
visualise_decoding_results(control_session_list, hom_session_list, control_output_root, hom_output_root, start_window, stop_window)