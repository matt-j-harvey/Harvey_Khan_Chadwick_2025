import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#import Sample_Random_Vectors_Preserve_Covariance




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


def get_peturbation_effect(vector, lick_cd, recurrent_matrix, n_timepoints):

    # Norm Lick CD
    lick_cd = lick_cd / np.linalg.norm(lick_cd)

    # Take Component Orthogonal To Lick CD
    projection = np.dot(vector, lick_cd)
    vector = np.subtract(vector, lick_cd * projection)

    # Norm Vector
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-12:
        return np.nan
    vector = vector / vector_norm

    trajectory = []
    current_state = vector
    for timepoint_index in range(n_timepoints):

        current_projection = np.dot(current_state, lick_cd)
        trajectory.append(current_projection)

        current_state = recurrent_matrix @ current_state

    # Get Integral
    total_effect = np.sum(trajectory)

    return total_effect, trajectory[1]


def analyse_effect_of_orthogonal_dimensions_on_lick_cd(recurrent_matrix, df_matrix, lick_cd):

    n_vectors = 10000
    n_timepoints = 10

    # Get Covariance Matrix
    covariance_matrix = np.cov(df_matrix, rowvar=False)
    covariance_matrix = np.nan_to_num(covariance_matrix, 0)

    # Sample Random Vectors
    random_vectors = sample_axes_according_to_covariance(covariance_matrix, n_vectors)

    # Iterate Through Each Vector and See What effect it has on the Lick CD
    effect_distribution = []
    single_step_distribution = []

    for vector in tqdm(random_vectors):
        effect, single_step_effect = get_peturbation_effect(vector, lick_cd, recurrent_matrix, n_timepoints)
        if not np.isnan(effect):
            effect_distribution.append(effect)
            single_step_distribution.append(single_step_effect)

    #plt.hist(effect_distribution)
    #plt.show()

    return effect_distribution, single_step_distribution


