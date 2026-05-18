import numpy as np

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
