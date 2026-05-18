import numpy as np

def analyse_lick_decay(lick_coding_dimension, recurrent_matrix):

    # Make lick coding dimension 1D and normalise
    lick_coding_dimension = np.asarray(lick_coding_dimension).squeeze()
    lick_coding_dimension = lick_coding_dimension / np.linalg.norm(lick_coding_dimension)

    # Create Empty List To Hold Lick CD Projection
    projection = []
    state_norm = []

    # Set Initial State To 0.5 along the Lick Coding Dimension
    current_state = np.multiply(lick_coding_dimension,0.5)
    print("lick_coding_dimension", np.shape(lick_coding_dimension))

    # Iterate Through 20 Timepoints
    for x in range(20):

        # A Project Current State Onto Lick CD and Add To Projection List
        current_state_projection = np.dot(current_state, lick_coding_dimension)
        projection.append(current_state_projection)

        # Get Orthogonal Component
        x_parallel = current_state_projection * lick_coding_dimension
        x_orthogonal = current_state - x_parallel
        state_norm.append(np.linalg.norm(x_orthogonal))

        # Get New State By Multiplying this with reucrrent matrix
        current_state = np.matmul(recurrent_matrix, current_state)

    return projection, state_norm

