import os
import numpy as np
import matplotlib.pyplot as plt

import Get_DF
import Get_Data_Tensor
import Get_Lick_CD
import Get_Vis_1_Response_CD

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def get_hits_by_rt(behaviour_matrix, rt_window_start, rt_window_stop):

    onset_list = []
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        reaction_time = trial[23]
        onset_frame = trial[18]

        if trial_type == 1:
            if correct == 1:
                reaction_time = int(reaction_time / 2)

                if reaction_time > rt_window_start and reaction_time <= rt_window_stop:
                    onset_list.append(onset_frame)

    return onset_list



def sort_psth_list(psth_list, window_start, window_stop):

    first_psth = psth_list[0]
    first_psth_window = first_psth[window_start:window_stop]
    first_psth_mean = np.mean(first_psth_window, axis=0)
    indicies = np.argsort(first_psth_mean)

    sorted_psth_list = []
    for psth in psth_list:
        sorted_psth = psth[:, indicies]
        sorted_psth_list.append(sorted_psth)
    return sorted_psth_list


def plot_psth(psth, psth_lick, x_values, magnitude=None):
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1, 1, 1)
    n_neurons = np.shape(psth)[1]
    if magnitude == None:
        magnitude = np.percentile(np.abs(psth), q=95)
    axis_1.imshow(np.transpose(psth), vmin=-magnitude, vmax=magnitude, cmap="bwr", extent=[x_values[0], x_values[-1], 0, n_neurons])
    forceAspect(axis_1)

    axis_1.axvline(0, c='k')
    axis_1.axvline(psth_lick, c='k')
    axis_1.set_title(psth_lick)




def view_lick_aligned_v_stim_aligned(data_directory_root, session_list, output_directory):
    # Split Licks By RT
    # 500 - 750
    # 750 - 1000
    # 1000 - 1250
    # 1250 - 1500

    rt_window_start_list = [500, 750, 1000, 1250]
    rt_window_stop_list = [750,1000, 1250, 1500]
    n_bins = len(rt_window_stop_list)

    start_window = -16
    stop_window = 16


    for bin_index in range(n_bins):

        # Get Bin Starts and Stops
        bin_start_window = rt_window_start_list[bin_index]
        bin_stop_window = rt_window_stop_list[bin_index]

        combined_raster = []
        combined_lick_cd = []

        for session in session_list:

            # Get Session Directory
            session_directory = os.path.join(data_directory_root, session)

            # Load dF Matrix
            df_matrix = Get_DF.load_df_matrix(session_directory)

            # Load Frame Rate
            frame_rate = np.load(os.path.join(session_directory, "Frame_Rate.npy"))

            # Load Lick CD
            lick_cd = Get_Lick_CD.get_lick_cd(session_directory, df_matrix)
            indicies = np.argsort(lick_cd)


            # Get X Values
            x_values = list(range(start_window, stop_window))
            x_values = np.multiply(x_values, 1.0 / frame_rate)

            # Load Behaviour Matrix
            behaviour_matrix = np.load(os.path.join(session_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

            # Get Vis 1 CD
            lick_cd = Get_Vis_1_Response_CD.get_vis_1_response_cd(df_matrix, behaviour_matrix)
            indicies = np.argsort(lick_cd)

            # Get Bin Onsets
            bin_onsets = get_hits_by_rt(behaviour_matrix, bin_start_window, bin_stop_window)
            print("bin onsets", len(bin_onsets))

            if len(bin_onsets) > 0:

                # Get Tensor
                baseline_correct = True
                baseline_window = 3
                tensor = Get_Data_Tensor.get_activity_tensors(df_matrix, bin_onsets, start_window, stop_window, baseline_correct, baseline_window)
                print("tensor", np.shape(tensor))

                # Get Mean
                bin_mean = np.mean(tensor, axis=0)
                print("bin mean", np.shape(bin_mean))

                # Sort Bin Mean
                bin_mean = bin_mean[:, indicies]

                # Plot Session PSTH
                save_directory = os.path.join(output_directory, session)
                print("save_directory", save_directory)
                if not os.path.exists(save_directory):
                    os.makedirs(save_directory)

                plot_psth(bin_mean, bin_start_window/1000, x_values)
                plt.savefig(os.path.join(save_directory, str(bin_start_window) + ".png"))
                #plt.show()
                plt.close()

                combined_raster.append(bin_mean)
                combined_lick_cd.append(lick_cd)

        combined_raster = np.hstack(combined_raster)
        combined_lick_cd = np.concatenate(combined_lick_cd)
        print("combined raster", np.shape(combined_raster))
        combined_indicies = np.argsort(combined_lick_cd)
        combined_raster = combined_raster[:, combined_indicies]
        #plot_psth(combined_raster, bin_start_window/1000, x_values, magnitude=0.5)

        save_directory = os.path.join(output_directory, "Group_Results")
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        #plt.savefig(os.path.join(save_directory, str(bin_start_window) + ".png"))
        #plt.show()









control_data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Controls"
control_output_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Meeting_With_Adil\2025_12_09\Lick_Aligned_Stim_Aligned\Controls"
control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]


hom_data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs"
hom_output_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Meeting_With_Adil\2025_12_09\Lick_Aligned_Stim_Aligned\Homs"
hom_session_list = [
    r"64.1B\2024_09_09_Switching",
    r"70.1A\2024_09_09_Switching",
    r"70.1B\2024_09_12_Switching",
    r"72.1E\2024_08_23_Switching",
]




#view_lick_aligned_v_stim_aligned(control_data_directory, control_session_list, control_output_directory)
view_lick_aligned_v_stim_aligned(hom_data_directory, hom_session_list, hom_output_directory)