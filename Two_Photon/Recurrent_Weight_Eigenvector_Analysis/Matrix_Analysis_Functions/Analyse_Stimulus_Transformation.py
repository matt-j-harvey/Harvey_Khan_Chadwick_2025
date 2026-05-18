import numpy as np
from Matrix_Analysis_Functions import Matrix_Analysis_Functions

def compute_stimulus_transformation_function(stim_vector, recurrent_matrix, lick_cd):

    # Normalise Both
    stim_vector = np.squeeze(stim_vector)
    stim_norm = np.linalg.norm(stim_vector)
    stim_vector = stim_vector / stim_norm

    lick_cd = np.squeeze(lick_cd)
    lick_norm = np.linalg.norm(lick_cd)
    lick_cd = lick_cd / lick_norm

    # Calculate Cosine Simmilarity
    cosine_similarity = Matrix_Analysis_Functions.get_cosine_simmilarity(stim_vector, lick_cd)

    # Compute Direct Overlap
    direct_projection = np.dot(stim_vector, lick_cd)

    # Get Orthogonal Components
    parallel = direct_projection * lick_cd
    orthogonal_component = stim_vector - parallel


    trajectory = []

    current_state = orthogonal_component
    for x in range(12):

        current_projection = np.dot(current_state, lick_cd)
        trajectory.append(current_projection)

        current_state = recurrent_matrix @ current_state

    return direct_projection, trajectory, cosine_similarity