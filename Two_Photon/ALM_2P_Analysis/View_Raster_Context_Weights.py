import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def load_df_matrix(data_directory, session):

    # Load DF Matrix
    df_matrix = np.load(os.path.join(data_directory, session, "df_matrix.npy"))
    df_matrix = np.transpose(df_matrix)

    # Z Score
    df_matrix = stats.zscore(df_matrix, axis=0)

    # Remove NaN
    df_matrix = np.nan_to_num(df_matrix)

    # Return
    return df_matrix


def view_raster_sorted_context_weights(data_directory, session):

    # Load DF Matrix
    df_matrix = load_df_matrix(data_directory, session)
    print("df_matrix", np.shape(df_matrix))

    # Load Context CD
    context_cd = np.load(os.path.join(data_directory, session, "Context_Decoding", "Decoding_Coefs.npy"))
    context_cd = np.mean(context_cd[0:18], axis=0)
    context_cd = np.squeeze(context_cd)

    context_indicies = np.argsort(context_cd)
    df_matrix = df_matrix[:, context_indicies]

    plt.imshow(np.transpose(df_matrix))
    forceAspect(plt.gca())
    plt.show()




data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Lick_Coding_Dimension_Comparison"

session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
    ]

for session in session_list:
    view_raster_sorted_context_weights(data_root, session)