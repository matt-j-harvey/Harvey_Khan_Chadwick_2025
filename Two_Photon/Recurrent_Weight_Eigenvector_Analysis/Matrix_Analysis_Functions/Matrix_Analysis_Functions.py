import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eig, solve_discrete_lyapunov


def get_observability_gramian(matrix):
    n = matrix.shape[0]
    identity = np.eye(n)
    W_o = solve_discrete_lyapunov(matrix.T, identity)
    return W_o

def get_controllability_gramian(matrix):
    n = matrix.shape[0]
    identity = np.eye(n)
    W_c = solve_discrete_lyapunov(matrix, identity @ identity.T)
    return W_c



def get_cosine_simmilarity(a, b):
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    return cos_sim

def norm_vector(vector):
    norm = np.linalg.norm(vector)
    vector = np.divide(vector, norm)
    return vector

def matrix_eigendecomposition(matrix, absolute=True):
    matrix = np.asarray(matrix, dtype=np.complex128)
    eigenvalues, left_mat, right_mat = eig(matrix, left=True, right=True)
    eigenvalues = np.real(eigenvalues)

    if absolute == True:
        eigenvalues = np.abs(eigenvalues)

    sort_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idx]
    left_vecs = left_mat[:, sort_idx].T
    right_vecs = right_mat[:, sort_idx].T
    return eigenvalues, left_vecs, right_vecs



def henrici_non_normality(matrix):

    # Get Matrix Dimension
    n = matrix.shape[0]

    # Get Eigenvalues
    eigvals = np.linalg.eigvals(matrix)

    # Get Sum of Sqaured absolute eigenvalues
    eig_sq_sum = np.sum(np.abs(eigvals) ** 2)

    # Get Sqaured Frobenium Norm
    fro_sq = np.linalg.norm(matrix, ord="fro") ** 2

    # Get The Difference
    diff = np.real(fro_sq - eig_sq_sum)

    # Take Sqrt
    sqrt_diff =  np.sqrt(diff)

    # Normalise By N
    normalised_sqrt_diff = sqrt_diff / n

    return normalised_sqrt_diff



def get_vector_alignment(vector_list, comparison_vector, absolute=False):

    vector_simmilarities = []
    for vector in vector_list:
        vector = vector.real
        simmilarity = get_cosine_simmilarity(vector, comparison_vector)
        if absolute == True:
            simmilarity = np.abs(simmilarity)
        vector_simmilarities.append(simmilarity)

    return vector_simmilarities


def compute_dimension_reachability(dimension, controllability_gramian):

    # Convert inputs to numpy arrays
    w = np.asarray(dimension, dtype=float).reshape(-1)
    W_c = np.asarray(controllability_gramian, dtype=float)

    # Optionally normalise lick coding dimension
    norm = np.linalg.norm(w)
    w = w / norm

    # Reachability / ease of driving along this direction
    reachability = float(w.T @ W_c @ w)

    return reachability



