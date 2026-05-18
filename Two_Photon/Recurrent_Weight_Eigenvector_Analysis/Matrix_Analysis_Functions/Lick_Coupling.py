import numpy as np
from tqdm import tqdm

def sample_axes_according_to_covariance(covariance_matrix, n_axes):

    # Get Dimensions of Data
    n_dimensions, _ = np.shape(covariance_matrix)

    # Perform Eigendecomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    eigenvalues = np.clip(eigenvalues, 0, None)

    # Create a matrix with sqrt eigenvalues down diagonal
    eigenvalues_sqrt = np.sqrt(eigenvalues)
    eigenvalues_sqrt = np.diag(eigenvalues_sqrt)

    # Generate Random Axes From Normal Distribution
    axes = np.random.normal(loc=0, scale=1, size=(n_axes, n_dimensions))

    # Transform These Axes By The Eigenvectors and Eigenvalues of the covariance matrix
    axes = np.matmul(axes, eigenvalues_sqrt)
    axes = np.matmul(axes, np.transpose(eigenvectors))

    # Renormalise To unit length
    norms = np.linalg.norm(axes, axis=1, keepdims=True)
    axes = axes / (norms + 1e-12)

    return axes


def analyse_effect_of_orthogonal_dimensions_on_lick_cd(recurrent_matrix, df_matrix, lick_cd):
    n_vectors = 10000
    n_timepoints = 13

    # Normalise lick_cd once
    lick_cd = np.asarray(lick_cd, dtype=np.float64).reshape(-1)
    lick_cd = lick_cd / (np.linalg.norm(lick_cd) + 1e-12)

    # Covariance
    covariance_matrix = np.cov(df_matrix, rowvar=False)
    covariance_matrix = np.nan_to_num(covariance_matrix, nan=0.0)

    # Sample random vectors
    random_vectors = sample_axes_according_to_covariance(covariance_matrix, n_vectors)

    # Remove component along lick_cd for all vectors at once
    projections = random_vectors @ lick_cd
    random_vectors = random_vectors - np.outer(projections, lick_cd)

    # Renormalise
    norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
    valid_mask = norms[:, 0] > 1e-12
    random_vectors = random_vectors[valid_mask] / norms[valid_mask]

    # Propagate all vectors together
    current_states = random_vectors.copy()

    total_effect = np.zeros(current_states.shape[0], dtype=np.float64)
    single_step_effect = None

    for t in tqdm(range(n_timepoints)):
        current_projection = current_states @ lick_cd
        total_effect += current_projection

        if t == 1:
            single_step_effect = current_projection.copy()

        # Row-wise batch update:
        # original single-vector code was recurrent_matrix @ current_state
        # so for rows of vectors, use @ recurrent_matrix.T
        current_states = current_states @ recurrent_matrix.T
    
    return total_effect, single_step_effect