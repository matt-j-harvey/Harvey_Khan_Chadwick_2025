import numpy as np
import os
import matplotlib.pyplot as plt
import pickle


def open_tensor(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        activity_tensor = session_trial_tensor_dict["tensor"]
    return activity_tensor

def get_start_stop_windows(file_location):
    with open(file_location + ".pickle", 'rb') as handle:
        session_trial_tensor_dict = pickle.load(handle)
        start_window = session_trial_tensor_dict["start_window"]
        stop_window = session_trial_tensor_dict["stop_window"]

    return start_window, stop_window


def get_trajectory_coordinates(data_directory, session, output_directory):

    # Load Tensor Start Windows
    tensor_start_window, tensor_stop_window = get_start_stop_windows(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_1"))
    print("tensor_start_window", tensor_start_window, "tensor_stop_window", tensor_stop_window)

    # Get Lick CD
    lick_cd_directory = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Results\Visual_Lick_CD"
    lick_cd = np.load(os.path.join(lick_cd_directory, session, "Visual Context Lick CD", "vis_context_lick_cd.npy"))

    # Load Context CD
    context_cd = np.load(os.path.join(data_directory, session, "Context_Decoding", "Decoding_Coefs.npy"))
    context_cd = np.mean(context_cd[0:18], axis=0)
    context_cd = np.squeeze(context_cd)

    # Load Activity Tensors
    vis_context_vis_1_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_1"))
    vis_context_vis_2_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "visual_context_stable_vis_2"))
    odr_context_vis_1_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_stable_vis_1"))
    odr_context_vis_2_tensor = open_tensor(os.path.join(output_directory, session, "Activity_Tensors", "odour_context_stable_vis_2"))

    # Get Means
    mean_vis_context_vis_1 = np.mean(vis_context_vis_1_tensor, axis=0)
    mean_vis_context_vis_2 = np.mean(vis_context_vis_2_tensor, axis=0)
    mean_odr_context_vis_1 = np.mean(odr_context_vis_1_tensor, axis=0)
    mean_odr_context_vis_2 = np.mean(odr_context_vis_2_tensor, axis=0)

    # Get Projections
    vis_context_vis_1_lick_projection = np.dot(mean_vis_context_vis_1, lick_cd)
    vis_context_vis_2_lick_projection = np.dot(mean_vis_context_vis_2, lick_cd)
    odr_context_vis_1_lick_projection = np.dot(mean_odr_context_vis_1, lick_cd)
    odr_context_vis_2_lick_projection = np.dot(mean_odr_context_vis_2, lick_cd)

    vis_context_vis_1_context_projection = np.dot(mean_vis_context_vis_1, context_cd)
    vis_context_vis_2_context_projection = np.dot(mean_vis_context_vis_2, context_cd)
    odr_context_vis_1_context_projection = np.dot(mean_odr_context_vis_1, context_cd)
    odr_context_vis_2_context_projection = np.dot(mean_odr_context_vis_2, context_cd)

    lick_projections = [vis_context_vis_1_lick_projection,
                          vis_context_vis_2_lick_projection,
                          odr_context_vis_1_lick_projection,
                          odr_context_vis_2_lick_projection]

    context_projections = [vis_context_vis_1_context_projection,
                           vis_context_vis_2_context_projection,
                           odr_context_vis_1_context_projection,
                           odr_context_vis_2_context_projection]

    return lick_projections, context_projections





# File Directory Info
data_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls"
mvar_output_root = r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\2P_MVAR_Results_Shared_Weights_No_Preceeding_Lick_Regressor"

control_session_list = [
    r"65.2a\2024_08_05_Switching",
    r"65.2b\2024_07_31_Switching",
    r"67.3b\2024_08_09_Switching",
    r"67.3C\2024_08_20_Switching",
    r"69.2a\2024_08_12_Switching",
    r"72.3C\2024_09_10_Switching",
]

group_lick_projections = []
group_context_projections = []

for session in control_session_list:
    lick_projections, context_projections = get_trajectory_coordinates(data_root, session, mvar_output_root)
    group_lick_projections.append(lick_projections)
    group_context_projections.append(context_projections)

group_lick_projections = np.array(group_lick_projections)
group_context_projections = np.array(group_context_projections)

print("group_lick_projections", np.shape(group_lick_projections))
print("group_context_projections", np.shape(group_context_projections))


mean_lick_projections = np.mean(group_lick_projections, axis=0)
mean_context_projection = np.mean(group_context_projections, axis=0)

length = 6
mean_lick_projections = mean_lick_projections[:, 17:17+length]
mean_context_projection = mean_context_projection[:, 17:17+length]

figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)

axis_1.plot(mean_lick_projections[0], mean_context_projection[0], c='b')
axis_1.plot(mean_lick_projections[1], mean_context_projection[1], c='r')
axis_1.plot(mean_lick_projections[2], mean_context_projection[2], c='g')
axis_1.plot(mean_lick_projections[3], mean_context_projection[3], c='m')

axis_1.scatter([mean_lick_projections[0, 0]], [mean_context_projection[0, 0]], c='k',  zorder=2)
axis_1.scatter([mean_lick_projections[1, 0]], [mean_context_projection[1, 0]], c='k',  zorder=2)
axis_1.scatter([mean_lick_projections[2, 0]], [mean_context_projection[2, 0]], c='k',  zorder=2)
axis_1.scatter([mean_lick_projections[3, 0]], [mean_context_projection[3, 0]], c='k',  zorder=2)

# Remove Spines
axis_1.spines[['right', 'top']].set_visible(False)

axis_1.set_xlabel("Lick Dimension")
axis_1.set_ylabel("Context Dimension")

plt.show()


