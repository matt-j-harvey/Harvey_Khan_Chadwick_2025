import os
import numpy as np
import matplotlib.pyplot as plt

def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)



def norm_vector_preserve_excitation(vector):

    """
    Just FOr Positive Only Check!

    """

    positive_vector = np.clip(vector, a_max=None, a_min=0)
    negative_vector = np.clip(vector, a_max=0, a_min=None)

    positive_vector = norm_vector(positive_vector)
    negative_vector = norm_vector(negative_vector)

    combined_vector = np.add(positive_vector, negative_vector)
    return combined_vector



def norm_vector(vector):
    norm = np.linalg.norm(vector)
    vector = np.divide(vector, norm)
    return vector



def visualise_stimuli_weights(visual_context_vis_1, visual_context_vis_2, odour_context_vis_1, odour_context_vis_2, save_directory):

    # Sort By Vis 1
    sorted_indicies = np.argsort(visual_context_vis_1)
    visual_context_vis_1 = visual_context_vis_1[sorted_indicies]
    visual_context_vis_2 = visual_context_vis_2[sorted_indicies]
    odour_context_vis_1 = odour_context_vis_1[sorted_indicies]
    odour_context_vis_2 = odour_context_vis_2[sorted_indicies]

    # Concatenate
    combined_matrix = np.vstack([visual_context_vis_1, visual_context_vis_2, odour_context_vis_1, odour_context_vis_2])

    weight_magnitude = np.percentile(np.abs(combined_matrix), q=95)

    figure_1  = plt.figure(figsize=(15,10))
    axis_1 = figure_1.add_subplot(1,1,1)

    axis_1.imshow(np.transpose(combined_matrix), cmap='bwr', vmin=-weight_magnitude, vmax=weight_magnitude, interpolation='none')
    axis_1.set_xticks(list(range(0,4)), ["visual_context_rewarded", "visual_context_unrewarded", "odour_context_rewarded", "odour_context_unrewarded"])

    forceAspect(plt.gca(), aspect=2)
    plt.savefig(os.path.join(save_directory, "Stimuli_Weights.png"))
    plt.close()




def get_stimuli_weights(model_dict, output_directory):

    model_params = model_dict['MVAR_Parameters']
    n_neurons = model_dict['Nvar']
    Nt = model_dict['Nt']
    preceeding_window = int(Nt/2)

    # Load Stimuli Weights
    stimulus_weight_list = []
    for stimulus_index in range(6):
        stimulus_weight_start = n_neurons + (stimulus_index * Nt)
        stimulus_weight_stop = stimulus_weight_start + Nt
        stimulus_weight_list.append(model_params[:, stimulus_weight_start:stimulus_weight_stop])

    # Extract Vis 1 and 2 Stimuli
    visual_context_vis_1 = stimulus_weight_list[0]
    visual_context_vis_2 = stimulus_weight_list[1]
    odour_context_vis_1 = stimulus_weight_list[2]
    odour_context_vis_2 = stimulus_weight_list[3]

    # Get Mean Stimuli Response
    response_window_size = 6
    visual_context_vis_1 = np.mean(visual_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    visual_context_vis_2 = np.mean(visual_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_1 = np.mean(odour_context_vis_1[:, preceeding_window:preceeding_window + response_window_size], axis=1)
    odour_context_vis_2 = np.mean(odour_context_vis_2[:, preceeding_window:preceeding_window + response_window_size], axis=1)


    # Normalise
    visual_context_vis_1 = norm_vector(visual_context_vis_1)
    visual_context_vis_2 = norm_vector(visual_context_vis_2)
    odour_context_vis_1 = norm_vector(odour_context_vis_1)
    odour_context_vis_2 = norm_vector(odour_context_vis_2)


    # Normalise
    """
    visual_context_vis_1 = norm_vector_preserve_excitation(visual_context_vis_1)
    visual_context_vis_2 = norm_vector_preserve_excitation(visual_context_vis_2)
    odour_context_vis_1 = norm_vector_preserve_excitation(odour_context_vis_1)
    odour_context_vis_2 = norm_vector_preserve_excitation(odour_context_vis_2)
    """

    # Save These
    save_directory = os.path.join(output_directory, "Stimuli Vectors")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "visual_context_vis_1.npy"), visual_context_vis_1)
    np.save(os.path.join(save_directory, "visual_context_vis_2.npy"), visual_context_vis_2)
    np.save(os.path.join(save_directory, "odour_context_vis_1.npy"), odour_context_vis_1)
    np.save(os.path.join(save_directory, "odour_context_vis_2.npy"), odour_context_vis_2)

    visualise_stimuli_weights(visual_context_vis_1, visual_context_vis_2, odour_context_vis_1, odour_context_vis_2, save_directory)