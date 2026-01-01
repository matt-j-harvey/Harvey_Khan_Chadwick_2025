import numpy as np
from scipy.linalg import null_space
from sympy.physics.units import length

"""
Null space:
The set of all possible solutions to the equation Ax = 0
A = Matrix
x = Vector

"""

def get_magnitude_of_orthogonal_projection(matrix, coding_dimension):

    """
    Matrix of shape (n_timepoints x n_neurons)
    Vector of shape n_neurons - vector you want the orthogonal subspace to
    """

    # Reshape Vector to be shape (1, n_neurons)
    coding_dimension = coding_dimension.reshape(1, -1)

    # Get Orthonormal Basis
    orthonormal_basis = null_space(coding_dimension)

    # Create Projection Matrix for Orthonormal basis
    projection_matrix = np.dot(orthonormal_basis, orthonormal_basis.T)

    # Project Each Timepoint Onto This and Get the length
    length_list = []
    for datapoint in matrix:

        # Project Into Subspace
        projection = np.dot(projection_matrix, datapoint)

        # Get Vector Length
        projection_length = np.linalg.norm(projection)

        # Add To List
        length_list.append(projection_length)

    return length_list



def get_positive_only_activity(matrix, lick_coding_dimension):

    proj_coefs = np.dot(matrix, lick_coding_dimension)

    projection = np.outer(proj_coefs, lick_coding_dimension)

    orthogonal_component = matrix - projection
    orthogonal_component = np.clip(orthogonal_component, a_min=0, a_max=None)
    print("orthogonal component", np.shape(orthogonal_component))
    mean_activity = np.mean(orthogonal_component, axis=1)
    #magnitude = np.linalg.norm(orthogonal_component, axis=1)
    return mean_activity




v = np.array([0, 0, 1], dtype=float)
v = v.reshape(1, -1)
print("v", np.shape(v))

# treat v as a row vector; its null space gives vectors orthogonal to v
orth_basis = null_space(v)
orth_projection_matrix = np.dot(orth_basis, orth_basis.T)
print("orth_projection_matrix", orth_projection_matrix)


test_vector = np.array([1,0,1])
projection = np.dot(orth_projection_matrix, test_vector)
projection_length = np.linalg.norm(projection)
print("projection", projection)
print("projection_length", projection_length)

