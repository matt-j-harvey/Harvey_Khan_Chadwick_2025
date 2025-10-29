import numpy as np
import os

def lowcut_filter(X, w = 0.0033, fs = 30.):
    from scipy.signal import butter, filtfilt
    b, a = butter(2,w/(fs/2.), btype='highpass');
    return filtfilt(b, a, X, padlen=20000, axis=0, padtype='constant')


def load_df_matrix(data_directory):

    # Load F Matrix
    f_matrix = np.load(os.path.join(data_directory, "F_Matrix.npy"))

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_directory, "Frame_Rate.npy"))

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

    return df_matrix
