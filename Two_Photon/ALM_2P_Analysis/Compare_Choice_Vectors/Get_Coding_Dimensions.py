import numpy as np
import os
import matplotlib.pyplot as plt


def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def visualise_coding_dimensions(combined_lick_dimension, visual_lick_dimension, odour_lick_dimension, visual_context_choice_dimension, odour_context_choice_dimension, save_directory):

    # Get Sorting Indicies
    sorted_indicies = np.argsort(combined_lick_dimension)
    combined_lick_dimension = combined_lick_dimension[sorted_indicies]
    visual_lick_dimension = visual_lick_dimension[sorted_indicies]
    odour_lick_dimension = odour_lick_dimension[sorted_indicies]
    visual_context_choice_dimension = visual_context_choice_dimension[sorted_indicies]
    odour_context_choice_dimension = odour_context_choice_dimension[sorted_indicies]

    # Concatenate
    combined_matrix = np.vstack([combined_lick_dimension, visual_lick_dimension, odour_lick_dimension, visual_context_choice_dimension, odour_context_choice_dimension])

    weight_magnitude = np.percentile(np.abs(combined_matrix), q=95)

    figure_1  = plt.figure(figsize=(15,10))
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.imshow(np.transpose(combined_matrix), cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude, interpolation='none')
    axis_1.set_xticks(list(range(0,5)), ["Combined_lick", "visual_lick", "odour_lick", "visual_choice", "odour_choice"])

    forceAspect(plt.gca(), aspect=2)
    #plt.show()
    plt.savefig(os.path.join(save_directory, "Coding_Dimensions.png"))
    plt.close()




def norm_vector(vector):
    norm = np.linalg.norm(vector)
    vector = np.divide(vector, norm)
    return vector

def get_coding_dimensions(output_directory):

    # Get Coding Dimensions
    """
    Combined Lick CD
    Visual Lick CD
    Odour Lick CD
    Visual Choice Dimension
    Odour Choice Dimension
    """

    # Load Tensors
    tensor_directory = os.path.join(output_directory, "Activity_Tensors")
    visual_lick_tensor = np.load(os.path.join(tensor_directory, "visual_lick_tensor.npy"))
    odour_lick_tensor = np.load(os.path.join(tensor_directory, "odour_lick_tensor.npy"))
    vis_context_stable_vis_1_tensor = np.load(os.path.join(tensor_directory, "vis_context_stable_vis_1_tensor.npy"))
    vis_context_stable_vis_2_tensor = np.load(os.path.join(tensor_directory, "vis_context_stable_vis_2_tensor.npy"))
    odour_1_tensor = np.load(os.path.join(tensor_directory, "odour_1_tensor.npy"))
    odour_2_tensor = np.load(os.path.join(tensor_directory, "odour_2_tensor.npy"))

    # Get Means Across Time
    visual_lick_tensor = np.mean(visual_lick_tensor, axis=1)
    odour_lick_tensor = np.mean(odour_lick_tensor, axis = 1)
    vis_context_stable_vis_1_tensor = np.mean(vis_context_stable_vis_1_tensor[6:-1], axis=1)
    vis_context_stable_vis_2_tensor = np.mean(vis_context_stable_vis_2_tensor[6:-1], axis=1)
    odour_1_tensor = np.mean(odour_1_tensor[6:-1], axis=1)
    odour_2_tensor = np.mean(odour_2_tensor[6:-1], axis=1)

    # Get Mean Across Trials
    visual_lick_tensor = np.mean(visual_lick_tensor, axis=0)
    odour_lick_tensor = np.mean(odour_lick_tensor, axis=0)
    vis_context_stable_vis_1_tensor = np.mean(vis_context_stable_vis_1_tensor, axis=0)
    vis_context_stable_vis_2_tensor = np.mean(vis_context_stable_vis_2_tensor, axis=0)
    odour_1_tensor = np.mean(odour_1_tensor, axis=0)
    odour_2_tensor = np.mean(odour_2_tensor, axis=0)

    # Get Combined Lick Tensor
    combined_lick_tensor = np.vstack([visual_lick_tensor, odour_lick_tensor])
    print("combined_lick_tensor", np.shape(combined_lick_tensor))
    combined_lick_tensor = np.mean(combined_lick_tensor, axis=0)
    print("combined_lick_tensor", np.shape(combined_lick_tensor))

    # Get Choice Dimensions
    visual_context_choice_dimension = np.subtract(vis_context_stable_vis_1_tensor, vis_context_stable_vis_2_tensor)
    odour_context_choice_dimension = np.subtract(odour_1_tensor, odour_2_tensor)

    # Norm
    visual_lick_dimension = norm_vector(visual_lick_tensor)
    odour_lick_dimension = norm_vector(odour_lick_tensor)
    combined_lick_dimension = norm_vector(combined_lick_tensor)
    visual_context_choice_dimension = norm_vector(visual_context_choice_dimension)
    odour_context_choice_dimension = norm_vector(odour_context_choice_dimension)

    # Save
    save_directory = os.path.join(output_directory, "Coding_Dimensions")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "combined_lick_dimension.npy"), combined_lick_dimension)
    np.save(os.path.join(save_directory, "visual_lick_dimension.npy"), visual_lick_dimension)
    np.save(os.path.join(save_directory, "odour_lick_dimension.npy"), odour_lick_dimension)
    np.save(os.path.join(save_directory, "visual_context_choice_dimension.npy"), visual_context_choice_dimension)
    np.save(os.path.join(save_directory, "odour_context_choice_dimension.npy"), odour_context_choice_dimension)

    visualise_coding_dimensions(combined_lick_dimension, visual_lick_dimension, odour_lick_dimension, visual_context_choice_dimension, odour_context_choice_dimension, save_directory)



