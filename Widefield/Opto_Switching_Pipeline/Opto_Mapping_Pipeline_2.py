import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import Ridge
import statsmodels.api as sm
from tqdm import tqdm

from Widefield_Utils import widefield_utils
import Session_List




def reconstruct_trials_into_pixel_space(data_directory_root, session, tensor):

    # Load Registered SVD
    reg_u = np.load(os.path.join(data_directory_root, session, "Preprocessed_Data", "Registered_U.npy"))

    # Flatten Reg U
    image_height, image_width, n_components = np.shape(reg_u)
    reg_u = np.reshape(reg_u, (image_height * image_width, n_components))

    # load mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Mask Reg U
    reg_u = reg_u[indicies]

    # Recosntruct
    tensor = np.dot(tensor, reg_u.T)

    return tensor





def z_score_trials(data_directory, session, tensor):

    # Load Mean and SD
    session_mean = np.load(os.path.join(data_directory, session, "Z_Scoring", "pixel_means.npy"))
    session_std = np.load(os.path.join(data_directory, session, "Z_Scoring", "pixel_stds.npy"))

    # Subtract Mean
    tensor = np.subtract(tensor, session_mean)

    # Divide By STD
    tensor = np.divide(tensor, session_std)
    tensor = np.nan_to_num(tensor)

    return tensor


def baseline_correct_tensor(tensor, baseline_start=0, baseline_stop=14):

    baseline_corrected_tensor = []
    for trial in tensor:
        trial_baseline = trial[baseline_start:baseline_stop]
        trial_baseline = np.mean(trial_baseline, axis=0)
        trial = np.subtract(trial, trial_baseline)
        baseline_corrected_tensor.append(trial)

    baseline_corrected_tensor = np.array(baseline_corrected_tensor)
    return baseline_corrected_tensor




def load_data(data_directory, output_directory, session, selected_timepoint):

    # Load Activity Tensors
    visual_context_control_activity_tensor = np.load(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_control_activity_tensor.npy"))
    odour_context_control_activity_tensor = np.load(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_control_activity_tensor.npy"))
    visual_context_opto_activity_tensor = np.load(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_opto_activity_tensor.npy"))
    odour_context_opto_activity_tensor = np.load(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_opto_activity_tensor.npy"))
    #print("visual_context_control_activity_tensor", np.shape(visual_context_control_activity_tensor))

    # Baseline Correct
    visual_context_control_activity_tensor = baseline_correct_tensor(visual_context_control_activity_tensor)
    odour_context_control_activity_tensor = baseline_correct_tensor(odour_context_control_activity_tensor)
    visual_context_opto_activity_tensor = baseline_correct_tensor(visual_context_opto_activity_tensor)
    odour_context_opto_activity_tensor = baseline_correct_tensor(odour_context_opto_activity_tensor)
    #print("visual_context_control_activity_tensor", np.shape(visual_context_control_activity_tensor))

    # Take Select Timepoint
    visual_context_control_activity_tensor = visual_context_control_activity_tensor[:, selected_timepoint]
    odour_context_control_activity_tensor = odour_context_control_activity_tensor[:, selected_timepoint]
    visual_context_opto_activity_tensor = visual_context_opto_activity_tensor[:, selected_timepoint]
    odour_context_opto_activity_tensor = odour_context_opto_activity_tensor[:, selected_timepoint]
    #print("visual_context_control_activity_tensor", np.shape(visual_context_control_activity_tensor))

    # Reconstruct Into Pixel Space
    visual_context_control_activity_tensor = reconstruct_trials_into_pixel_space(data_directory, session, visual_context_control_activity_tensor)
    odour_context_control_activity_tensor = reconstruct_trials_into_pixel_space(data_directory, session, odour_context_control_activity_tensor)
    visual_context_opto_activity_tensor = reconstruct_trials_into_pixel_space(data_directory, session, visual_context_opto_activity_tensor)
    odour_context_opto_activity_tensor = reconstruct_trials_into_pixel_space(data_directory, session, odour_context_opto_activity_tensor)
    #print("visual_context_control_activity_tensor", np.shape(visual_context_control_activity_tensor))

    # Z Score
    visual_context_control_activity_tensor = z_score_trials(data_directory, session, visual_context_control_activity_tensor)
    odour_context_control_activity_tensor = z_score_trials(data_directory, session, odour_context_control_activity_tensor)
    visual_context_opto_activity_tensor = z_score_trials(data_directory, session, visual_context_opto_activity_tensor)
    odour_context_opto_activity_tensor = z_score_trials(data_directory, session, odour_context_opto_activity_tensor)
    #print("visual_context_control_activity_tensor", np.shape(visual_context_control_activity_tensor))

    return visual_context_control_activity_tensor, odour_context_control_activity_tensor, visual_context_opto_activity_tensor, odour_context_opto_activity_tensor



def view_regressor(coef_vector, p_value_vector, index):
    selected_coefs = coef_vector[:, index]
    select_p_values = p_value_vector[:, index]
    sig_coefs = np.where(select_p_values < 0.05, selected_coefs, 0)

    # Load Mask
    indicies, image_height, image_width = widefield_utils.load_tight_mask()

    # Reconstruct Images
    sig_coefs = widefield_utils.create_image_from_data(sig_coefs, indicies, image_height, image_width)

    plt.imshow(sig_coefs)
    plt.show()


def fit_linear_model(data_directory, glm_directory, roi_name, opto_session_list, control_session_list, start_window, stop_window, mean_window_start, mean_window_stop):

    """
    Regressors
    0 = Trial
    1 = Context
    2 = light
    3 = Optogenetics
    4 = red light x context interaction
    5 = optogenetics x context interaction
    """

    # Create Regressor Table
                                            #  T  C  L  O
    odour_context_no_light_control_mouse    = [1, 0, 0, 0]
    visual_context_no_light_control_mouse   = [1, 1, 0, 0]
    odour_context_light_control_mouse       = [1, 0, 1, 0]
    visual_context_light_control_mouse      = [1, 1, 1, 0]

    odour_context_no_light_opsin_mouse      = [1, 0, 0, 0]
    visual_context_no_light_opsin_mouse     = [1, 1, 0, 0]
    odour_context_light_opsin_mouse         = [1, 0, 1, 1]
    visual_context_light_opsin_mouse        = [1, 1, 1, 1]


    n_timepoints = stop_window - start_window
    for timepoint_index in range(n_timepoints):

        design_matrix = []
        delta_f_matrix = []
        for session in tqdm(control_session_list):

            vis_context_control, odr_context_control, vis_context_opto, odr_context_opto = load_data(data_directory, glm_directory, session, timepoint_index)

            # Create Session Regressors
            vis_context_control_regressor = np.empty((np.shape(vis_context_control)[0], 4))
            vis_context_control_regressor[:] = visual_context_no_light_control_mouse

            odr_context_control_regressor = np.empty((np.shape(odr_context_control)[0], 4))
            odr_context_control_regressor[:] = odour_context_no_light_control_mouse

            vis_context_opto_regressor = np.empty((np.shape(vis_context_opto)[0], 4))
            vis_context_opto_regressor[:] = visual_context_light_control_mouse

            odr_context_opto_regressor = np.empty((np.shape(odr_context_opto)[0], 4))
            odr_context_opto_regressor[:] = odour_context_light_control_mouse

            # Add Df Data
            delta_f_matrix.append(vis_context_control)
            delta_f_matrix.append(odr_context_control)
            delta_f_matrix.append(vis_context_opto)
            delta_f_matrix.append(odr_context_opto)

            # Add Regressors To Design MAtrix
            design_matrix.append(vis_context_control_regressor)
            design_matrix.append(odr_context_control_regressor)
            design_matrix.append(vis_context_opto_regressor)
            design_matrix.append(odr_context_opto_regressor)


        for session in opto_session_list:

            vis_context_control, odr_context_control, vis_context_opto, odr_context_opto = load_data(data_directory, glm_directory, session, timepoint_index)

            # Create Session Regressors
            vis_context_control_regressor = np.empty((np.shape(vis_context_control)[0], 4))
            vis_context_control_regressor[:] = visual_context_no_light_opsin_mouse

            odr_context_control_regressor = np.empty((np.shape(odr_context_control)[0], 4))
            odr_context_control_regressor[:] = odour_context_no_light_opsin_mouse

            vis_context_opto_regressor = np.empty((np.shape(vis_context_opto)[0], 4))
            vis_context_opto_regressor[:] = visual_context_light_opsin_mouse

            odr_context_opto_regressor = np.empty((np.shape(odr_context_opto)[0], 4))
            odr_context_opto_regressor[:] = odour_context_light_opsin_mouse

            # Add Df Data
            delta_f_matrix.append(vis_context_control)
            delta_f_matrix.append(odr_context_control)
            delta_f_matrix.append(vis_context_opto)
            delta_f_matrix.append(odr_context_opto)

            design_matrix.append(vis_context_control_regressor)
            design_matrix.append(odr_context_control_regressor)
            design_matrix.append(vis_context_opto_regressor)
            design_matrix.append(odr_context_opto_regressor)

        # Combine Into Arrays
        design_matrix = np.vstack(design_matrix)
        delta_f_matrix = np.vstack(delta_f_matrix)

        # Create Interaction Terms
        light_context_interaction = np.multiply(design_matrix[:, 1], design_matrix[:, 2])
        opsin_context_interaction = np.multiply(design_matrix[:, 1], design_matrix[:, 3])
        light_context_interaction = np.expand_dims(light_context_interaction, axis=1)
        opsin_context_interaction = np.expand_dims(opsin_context_interaction, axis=1)
        interaction_regressors = np.hstack([light_context_interaction, opsin_context_interaction])
        design_matrix = np.hstack([design_matrix, interaction_regressors])

        # Create  Model
        model = Ridge(alpha=2, fit_intercept=False)
        model.fit(X=design_matrix, y=delta_f_matrix)

        #Extract Coefs
        model_coefs = model.coef_
        context_coefs = model_coefs[:, 1]
        light_coefs = model_coefs[:, 2]
        opsin_coefs = model_coefs[:, 3]
        light_interaction_coefs = model_coefs[:, 4]
        opsin_interaction_coefs = model_coefs[:, 5]

        # Load Mask
        indicies, image_height, image_width = widefield_utils.load_tight_mask()

        # Reconstruct Images
        context_coefs = widefield_utils.create_image_from_data(context_coefs, indicies, image_height, image_width)
        light_coefs = widefield_utils.create_image_from_data(light_coefs, indicies, image_height, image_width)
        opsin_coefs = widefield_utils.create_image_from_data(opsin_coefs, indicies, image_height, image_width)
        light_interaction_coefs = widefield_utils.create_image_from_data(light_interaction_coefs, indicies, image_height, image_width)
        opsin_interaction_coefs = widefield_utils.create_image_from_data(opsin_interaction_coefs, indicies, image_height, image_width)

        # Save These
        save_root = r"/media/matthew/29D46574463D2856/BNA_Poster_Results/Combined_Opto_GLM_Seperate_Timepoints"
        save_directory = os.path.join(save_root, str(timepoint_index).zfill(3), roi_name)
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        np.save(os.path.join(save_directory, "context_coefs.npy"), context_coefs)
        np.save(os.path.join(save_directory, "light_coefs.npy"), light_coefs)
        np.save(os.path.join(save_directory, "opsin_coefs.npy"), opsin_coefs)
        np.save(os.path.join(save_directory, "light_interaction_coefs.npy"), light_interaction_coefs)
        np.save(os.path.join(save_directory, "opsin_interaction_coefs.npy"), opsin_interaction_coefs)

    """
    # Load COlournmap
    cmap = widefield_utils.get_musall_cmap()
    magntiude = 0.4

    plt.title("context_coefs")
    plt.imshow(context_coefs, cmap=cmap, vmin=-magntiude, vmax=magntiude)
    plt.show()

    plt.title("light_coefs")
    plt.imshow(light_coefs, cmap=cmap, vmin=-magntiude, vmax=magntiude)
    plt.show()

    plt.title("opsin_coefs")
    plt.imshow(opsin_coefs, cmap=cmap, vmin=-magntiude, vmax=magntiude)
    plt.show()

    plt.title("light_interaction_coefs")
    plt.imshow(light_interaction_coefs, cmap=cmap, vmin=-magntiude, vmax=magntiude)
    plt.show()

    plt.title("opsin_interaction_coefs")
    plt.imshow(opsin_interaction_coefs, cmap=cmap, vmin=-0.2, vmax=0.2)
    plt.show()

    print("design_matrix", np.shape(design_matrix))
    print("delta_f_matrix", np.shape(delta_f_matrix))
    """



frame_period = 36
start_window_ms = -2500
stop_window_ms = 0
start_window = int(start_window_ms/frame_period)
stop_window = int(stop_window_ms/frame_period)

mean_window_start = 14
mean_window_stop = 41

data_directory = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"
glm_directory = r"/media/matthew/29D46574463D2856/Paper_Results/Opto_Mapping_GLM_Baseline_Correct"

experiment_list = [

    ["V1",
     Session_List.v1_opto_session_list,
     Session_List.v1_control_session_list],

    ["PPC",
     Session_List.ppc_opto_session_list,
     Session_List.ppc_control_session_list],

    ["SS",
     Session_List.ss_opto_session_list,
     Session_List.ss_control_session_list],

    ["MM",
     Session_List.mm_opto_session_list,
     Session_List.mm_control_session_list],

    ["ALM",
     Session_List.alm_opto_session_list,
     Session_List.alm_control_session_list],

    ["RSC",
     Session_List.rsc_opto_session_list,
     Session_List.rsc_control_session_list],

    ["PM",
     Session_List.pm_opto_session_list,
     Session_List.pm_control_session_list],

]


"""

    [
    mm_opto_session_list 
    alm_opto_session_list
    rsc_opto_session_list 
    pm_opto_session_list 


"""

for experiment in experiment_list:
    fit_linear_model(data_directory, glm_directory, experiment[0], experiment[1], experiment[2], start_window, stop_window, mean_window_start, mean_window_stop)