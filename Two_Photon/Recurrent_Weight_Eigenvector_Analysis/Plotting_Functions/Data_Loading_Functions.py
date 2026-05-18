import os
import numpy as np



def load_data(session_list, output_root, filename, truncation=False):

    data_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:

            session_data = np.load(os.path.join(output_root, session, filename))

            if truncation != False:
                session_data = session_data[0:truncation]

            mouse_list.append(session_data)

        if len(mouse_list) == 1:
            data_list.append(mouse_list[0])
        else:
            mouse_mean = np.mean(np.array(mouse_list), axis=0)
            data_list.append(mouse_mean)

    return data_list



def get_hist_density(data, bin_range, bin_size):
    data = np.asarray(data)

    n_samples = len(data)
    bin_starts = np.arange(-bin_range, bin_range, bin_size)
    bin_stops = bin_starts + bin_size

    density = []
    for bin_start, bin_stop in zip(bin_starts, bin_stops):
        bin_count = np.sum((data >= bin_start) & (data < bin_stop))
        bin_density = bin_count / n_samples
        density.append(bin_density)

    return np.array(density)



def load_distributions(session_list, output_root, filename, bin_range, bin_size):

    data_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:

            session_data = np.load(os.path.join(output_root, session, filename))
            session_data = get_hist_density(session_data, bin_range, bin_size)

            mouse_list.append(session_data)

        if len(mouse_list) == 1:
            data_list.append(mouse_list[0])
        else:
            mouse_mean = np.mean(np.array(mouse_list), axis=0)
            data_list.append(mouse_mean)

    return data_list


def load_distribution_means(session_list, output_root, filename):

    data_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:

            session_data = np.load(os.path.join(output_root, session, filename))
            session_mean = np.mean(session_data)
            mouse_list.append(session_mean)

        if len(mouse_list) == 1:
            data_list.append(mouse_list[0])
        else:
            mouse_mean = np.mean(np.array(mouse_list), axis=0)
            data_list.append(mouse_mean)

    return data_list






"""
def load_eigenspectrums(session_list, output_root):

    eigenspectrum_list = []
    for session in session_list:

        session_eigenspectrum = np.load(os.path.join(output_root, session, "Sorted_Eigenvalues.npy"))

        # Take Only Real Part
        session_eigenspectrum = np.real(session_eigenspectrum)

        session_eigenspectrum = session_eigenspectrum[0:50]
        eigenspectrum_list.append(session_eigenspectrum)

    return eigenspectrum_list
"""


def load_eigenspectrums(session_list, output_root):

    eigenspectrum_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:

            # Load Session Eigenspectrum
            session_eigenspectrum = np.load(os.path.join(output_root, session, "Sorted_Eigenvalues.npy"))

            # Take Only Real Part
            session_eigenspectrum = np.real(session_eigenspectrum)

            # Take Only Top 50 Eigenvectors
            session_eigenspectrum = session_eigenspectrum[0:30]
            print("session_eigenspectrum", len(session_eigenspectrum))

            # Add To Mouse List
            mouse_list.append(session_eigenspectrum)

        if len(mouse_list) == 1:
            eigenspectrum_list.append(mouse_list[0])
        else:
            mouse_mean = np.mean(np.array(mouse_list), axis=0)
            eigenspectrum_list.append(mouse_mean)

    return eigenspectrum_list



def load_observability_eigenspectrums(session_list, output_root):

    eigenspectrum_list = []
    for mouse in session_list:

        mouse_list = []
        for session in mouse:
            # Load Session Eigenspectrum
            session_eigenspectrum = np.load(os.path.join(output_root, session, "observability_eigenvalues.npy"))

            # Take Only Real Part
            session_eigenspectrum = np.real(session_eigenspectrum)

            # Take Only Top 50 Eigenvectors
            session_eigenspectrum = session_eigenspectrum[0:30]

            # Add To Mouse List
            mouse_list.append(session_eigenspectrum)

        if len(mouse_list) == 1:
            eigenspectrum_list.append(mouse_list)
        else:
            mouse_mean = np.mean(np.array(mouse_list), axis=0)
            eigenspectrum_list.append(mouse_mean)

    return eigenspectrum_list


def load_alignment(session_list, output_root):

    alignment_list = []
    for session in session_list:

        session_alignment = np.load(os.path.join(output_root, session, "Right_Eigenvectors_Lick_Alignment.npy"))

        session_alignment = session_alignment[0:50]
        alignment_list.append(session_alignment)

    return alignment_list



def load_controlability_alignment(session_list, output_root):

    alignment_list = []
    for session in session_list:

        session_alignment = np.load(os.path.join(output_root, session, "Controlability_lick_Alignment.npy"))

        session_alignment = session_alignment[0:50]
        alignment_list.append(session_alignment)

    return alignment_list


def load_left_alignment(session_list, output_root):

    vis_1_alignment_list = []
    vis_2_alignment_list = []

    for session in session_list:

        session_vis_1_alignment = np.load(os.path.join(output_root, session, "Left_Eigenvectors_Vis_1_Alignment.npy"))
        session_vis_2_alignment = np.load(os.path.join(output_root, session, "Left_Eigenvectors_Vis_2_Alignment.npy"))

        session_vis_1_alignment = session_vis_1_alignment[0:50]
        session_vis_2_alignment = session_vis_2_alignment[0:50]

        vis_1_alignment_list.append(session_vis_1_alignment)
        vis_2_alignment_list.append(session_vis_2_alignment)

    return vis_1_alignment_list, vis_2_alignment_list





def load_left_alignment_observability(session_list, output_root):

    vis_1_alignment_list = []
    vis_2_alignment_list = []

    for session in session_list:

        session_vis_1_alignment = np.load(os.path.join(output_root, session, "Observability_Vis_1_Alignment.npy"))
        session_vis_2_alignment = np.load(os.path.join(output_root, session, "Observability_Vis_2_Alignment.npy"))

        session_vis_1_alignment = session_vis_1_alignment[0:50]
        session_vis_2_alignment = session_vis_2_alignment[0:50]

        vis_1_alignment_list.append(session_vis_1_alignment)
        vis_2_alignment_list.append(session_vis_2_alignment)

    return vis_1_alignment_list, vis_2_alignment_list






def load_non_normality(session_list, output_root):

    non_normality_list = []
    for session in session_list:
        session_non_normality = np.load(os.path.join(output_root, session, "non_normality.npy"))
        non_normality_list.append(session_non_normality)

    return non_normality_list