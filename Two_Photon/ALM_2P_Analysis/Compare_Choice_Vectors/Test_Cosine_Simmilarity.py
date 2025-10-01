import os
from numpy.linalg import norm
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm




def shuffle_groups(group_1_vectors, group_2_vectors):

    # Take two lists of vectors, and arbitrailty Swap their values
    n_group_1_samples = len(group_1_vectors)
    n_group_2_samples = len(group_2_vectors)

    # Create ID List
    group_1_ids = np.zeros(n_group_1_samples)
    group_2_ids = np.ones(n_group_2_samples)
    combined_ids = np.concatenate([group_1_ids, group_2_ids])

    # Shuffle
    combined_data = np.concatenate([group_1_vectors, group_2_vectors])

    # Get Combined List
    shuffled_ids = np.copy(combined_ids)
    np.random.shuffle(shuffled_ids)

    shuffled_group_1_vectors = combined_data[np.argwhere(shuffled_ids == 0)]
    shuffled_group_2_vectors = combined_data[np.argwhere(shuffled_ids == 1)]

    shuffled_group_1_vectors = np.squeeze(shuffled_group_1_vectors)
    shuffled_group_2_vectors = np.squeeze(shuffled_group_2_vectors)

    return shuffled_group_1_vectors, shuffled_group_2_vectors


def test_cosine_simmilarity_two_tensors(tensor_1, tensor_2, mean_period, save_directory):

    #Get Means
    group_1_vectors = np.mean(tensor_1[:, mean_period[0]:mean_period[1]], axis=1)
    group_2_vectors = np.mean(tensor_2[:, mean_period[0]:mean_period[1]], axis=1)

    # Get Average Cosine Similarity
    group_1_mean_vector = np.mean(group_1_vectors, axis=0)
    group_2_mean_vector = np.mean(group_2_vectors, axis=0)
    real_distance = np.dot(group_1_mean_vector,group_2_mean_vector)/(norm(group_1_mean_vector)*norm(group_2_mean_vector))

    shuffled_distribution = []
    n_iterations = 100000
    for iteration in tqdm(range(n_iterations)):

        # Shuffle Data
        shuffled_group_1_data, shuffled_group_2_data = shuffle_groups(group_1_vectors, group_2_vectors)

        # Get Distance
        shuffled_group_1_mean_vector = np.squeeze(np.mean(shuffled_group_1_data, axis=0))
        shuffled_group_2_mean_vector = np.squeeze(np.mean(shuffled_group_2_data, axis=0))
        shuffle_distance = np.dot(shuffled_group_1_mean_vector, shuffled_group_2_mean_vector) / (norm(shuffled_group_1_mean_vector) * norm(shuffled_group_2_mean_vector))
        shuffled_distribution.append(shuffle_distance)


    # Plot Distance
    print("real distance", real_distance)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.hist(shuffled_distribution, bins=60)
    axis_1.axvline(real_distance, c='k', linestyle='dashed')
    axis_1.set_xlim([0, 1])
    axis_1.set_xlabel("Cosine Simmilarity")
    axis_1.set_ylabel("Frequency")
    axis_1.spines[['right', 'top']].set_visible(False)
    plt.savefig(os.path.join(save_directory, "Coding_Dimension_Cosine_Simmilarity.png"))
    plt.close()

    # Save Values
    mean_shuffled_distance = np.mean(shuffled_distribution)
    np.save(os.path.join(save_directory, "Real_Distance.npy"), real_distance)
    np.save(os.path.join(save_directory, "Shuffled_Distance.npy"), mean_shuffled_distance)






def get_paired_distance(visual_1_vectors, visual_2_vectors, odour_1_vectors, odour_2_vectors):

    # Get Means
    visual_1_mean = np.mean(visual_1_vectors, axis=0)
    visual_2_mean = np.mean(visual_2_vectors, axis=0)
    odour_1_mean = np.mean(odour_1_vectors, axis=0)
    odour_2_mean = np.mean(odour_2_vectors, axis=0)

    # Get Choice Dimensions
    visual_choice_dimension = np.subtract(visual_1_mean, visual_2_mean)
    odour_choice_dimension = np.subtract(odour_1_mean, odour_2_mean)

    # Get Distance
    distance = np.dot(visual_choice_dimension,odour_choice_dimension)/(norm(visual_choice_dimension)*norm(odour_choice_dimension))

    return distance




def test_output_dimension_cosine_simmilarity(visual_1, visual_2, odour_1, odour_2, mean_period, save_directory):

    # Get Means
    visual_1_vectors = np.mean(visual_1[:, mean_period[0]:mean_period[1]], axis=1)
    visual_2_vectors = np.mean(visual_2[:, mean_period[0]:mean_period[1]], axis=1)
    odour_1_vectors = np.mean(odour_1[:, mean_period[0]:mean_period[1]], axis=1)
    odour_2_vectors = np.mean(odour_2[:, mean_period[0]:mean_period[1]], axis=1)

    # Get Real Distance
    real_distance = get_paired_distance(visual_1_vectors, visual_2_vectors, odour_1_vectors, odour_2_vectors)

    # Shuffle
    n_iterations = 100000

    shuffled_distribution = []
    for iteration in tqdm(range(n_iterations)):

        # Shuffle Contexts
        shuffled_visual_1_vectors, shuffled_odour_1_vectors = shuffle_groups(visual_1_vectors, odour_1_vectors)
        shuffled_visual_2_vectors, shuffled_odour_2_vectors = shuffle_groups(visual_2_vectors, odour_2_vectors)

        # Get Shuffled Distance
        distance = get_paired_distance(shuffled_visual_1_vectors, shuffled_visual_2_vectors, shuffled_odour_1_vectors, shuffled_odour_2_vectors)
        shuffled_distribution.append(distance)


    # Plot Distance
    print("real distance", real_distance)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.hist(shuffled_distribution, bins=60)
    axis_1.axvline(real_distance, c='k', linestyle='dashed')
    axis_1.set_xlim([0, 1])
    axis_1.set_xlabel("Cosine Simmilarity")
    axis_1.set_ylabel("Frequency")
    axis_1.spines[['right', 'top']].set_visible(False)
    plt.savefig(os.path.join(save_directory, "Coding_Dimension_Cosine_Simmilarity.png"))
    plt.close()

    # Save Values
    mean_shuffled_distance = np.mean(shuffled_distribution)
    np.save(os.path.join(save_directory, "Real_Distance.npy"), real_distance)
    np.save(os.path.join(save_directory, "Shuffled_Distance.npy"), mean_shuffled_distance)




def perform_similarity_checks(data_directory, output_directory):

    # Load Data Tensors
    tensor_directory = os.path.join(output_directory, "Activity_Tensors")
    visual_lick_tensor = np.load(os.path.join(tensor_directory, "visual_lick_tensor.npy"))
    odour_lick_tensor = np.load(os.path.join(tensor_directory, "odour_lick_tensor.npy"))
    vis_context_stable_vis_1_tensor = np.load(os.path.join(tensor_directory, "vis_context_stable_vis_1_tensor.npy"))
    vis_context_stable_vis_2_tensor = np.load(os.path.join(tensor_directory, "vis_context_stable_vis_2_tensor.npy"))
    odour_1_tensor = np.load(os.path.join(tensor_directory, "odour_1_tensor.npy"))
    odour_2_tensor = np.load(os.path.join(tensor_directory, "odour_2_tensor.npy"))

    # Test Lick Cosine Simmilarity
    lick_save_directory = os.path.join(output_directory, "Cosine_Simmilarity", "Lick")
    if not os.path.exists(lick_save_directory):
        os.makedirs(lick_save_directory)

    test_cosine_simmilarity_two_tensors(visual_lick_tensor, odour_lick_tensor, mean_period=[0, 8], save_directory=lick_save_directory)


    # Test Choice Dimension Cosine Simmilarity
    choice_save_directory = os.path.join(output_directory, "Cosine_Simmilarity", "Choice_Dimensions")
    if not os.path.exists(choice_save_directory):
        os.makedirs(choice_save_directory)

    test_output_dimension_cosine_simmilarity(vis_context_stable_vis_1_tensor,
                                             vis_context_stable_vis_2_tensor,
                                             odour_1_tensor,
                                             odour_2_tensor,
                                             mean_period=[6, -1],
                                             save_directory=choice_save_directory)
