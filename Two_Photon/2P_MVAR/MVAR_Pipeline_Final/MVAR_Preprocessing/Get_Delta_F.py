import os
import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def lowcut_filter(X, w = 0.0033, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=20000, axis=0, padtype='constant')


def moving_average(data_matrix, window_size):

    smoothed_data = np.copy(data_matrix)

    n_timepoints = np.shape(data_matrix)[0]
    for x in tqdm(range(window_size, n_timepoints)):

        window_start = x-window_size
        window_stop = x
        data_window = data_matrix[window_start:window_stop]
        data_mean = np.mean(data_window, axis=0)
        smoothed_data[x] = data_mean

    return smoothed_data


def sort_matrix_rastermap(matrix):

    # fit rastermap#
    matrix = np.transpose(matrix)

    model = rastermap.Rastermap(n_PCs=200, n_clusters=100,
                      locality=0.75, time_lag_window=5).fit(matrix)

    isort = model.isort

    sorted_matrix = matrix[isort]
    sorted_matrix = np.transpose(sorted_matrix)
    return sorted_matrix


def get_moving_baseline(data_matrix, window_size):

    """
    Did i forget this axis!?!?!

    """

    moving_baseline = []

    for x in range(window_size):
        window_baseline = np.percentile(data_matrix[0:window_size], q=5, axis=0)
        moving_baseline.append(window_baseline)

    n_timepoints = np.shape(data_matrix)[0]
    for x in tqdm(range(window_size, n_timepoints)):
        window_start = x - window_size
        window_stop = x
        data_window = data_matrix[window_start:window_stop]
        window_baseline = np.percentile(data_window, q=5, axis=0)
        #print("window_baseline", np.shape(window_baseline))
        moving_baseline.append(window_baseline)

    moving_baseline = np.array(moving_baseline)
    #moving_baseline = np.expand_dims(moving_baseline, axis=1)
    print("moving_baseline", np.shape(moving_baseline))
    return moving_baseline


def get_delta_f(data_root, session, mvar_output_root, plot_rastermap=False):

    # Get Directories
    data_directory = os.path.join(data_root, session)
    output_directory = os.path.join(mvar_output_root, session)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Load F Matrix
    f_matrix = np.load(os.path.join(data_directory, "F_Matrix.npy"))
    print("f_matrix", np.shape(f_matrix))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_directory, "Frame_Rate.npy"))
    print("frame_rate", frame_rate)

    # Lowcut Filter
    mean = np.mean(f_matrix, axis=0)
    f_matrix = np.subtract(f_matrix, mean)
    f_matrix = lowcut_filter(f_matrix, w=0.0006, fs=frame_rate)
    f_matrix = np.add(f_matrix, mean)

    # Get Baseline
    f_zero = np.percentile(f_matrix, q=5, axis=0)

    # Get dF/F
    df = np.subtract(f_matrix, f_zero)
    df_matrix = np.divide(df, f_zero)

    # Save Matrix
    np.save(os.path.join(output_directory, "df_over_f_matrix.npy"), df_matrix)

    # Plot Matrix
    if plot_rastermap == True:
        import rastermap
        sorted_matrix = sort_matrix_rastermap(df_matrix)
        plt.imshow(np.transpose(sorted_matrix), vmin=np.percentile(sorted_matrix, q=10), vmax=np.percentile(sorted_matrix, q=99), cmap='Greys')
        forceAspect(plt.gca())
        plt.savefig(os.path.join(output_directory, "df_over_f_rastermap.png"))
        plt.close()