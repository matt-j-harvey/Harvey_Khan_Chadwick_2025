import Matrix_Analysis_Functions
import os
import numpy as np

from Matrix_Analysis_Functions import Matrix_Analysis_Functions

def get_covariance_alignment_lick_cd(df_matrix, lick_cd):

    # Get Covariance Matrix
    covariance_matrix = np.cov(df_matrix, rowvar=False)
    covariance_matrix = np.nan_to_num(covariance_matrix, 0)

    # Get Eigenvectors Of Covariance Matrix
    eigenvalues, left_vecs, right_vecs = Matrix_Analysis_Functions.matrix_eigendecomposition(covariance_matrix, absolute=True)

    # Get Absolute Cosine Similarity With Lick CD
    eigenvector_alignment = Matrix_Analysis_Functions.get_vector_alignment(left_vecs, lick_cd, absolute=True)

    return eigenvector_alignment

