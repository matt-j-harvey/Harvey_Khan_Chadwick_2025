import os
import numpy as np
import matplotlib.pyplot as plt
import Session_List

from tqdm import tqdm




def place_roi_into_mousecam(roi_pixels, image_height, image_width, vector):

    number_pixels = np.shape(roi_pixels)[0]
    template = np.zeros((image_height, image_width))

    for pixel_index in range(number_pixels):
        pixel_value = vector[pixel_index]
        pixel_x = roi_pixels[pixel_index, 1]
        pixel_y = roi_pixels[pixel_index, 0]
        template[pixel_y, pixel_x] = pixel_value

    return template

def view_face_components(base_directory):

    image_width = 640
    image_height = 480

    # Get Mousecam Directory
    mousecam_directory = os.path.join(base_directory, "Mousecam_Analysis")

    # Get Mousecam Components
    mousecam_components = np.load(os.path.join(mousecam_directory, "Face_Motion_Components.npy"))
    print("Mousecam Components", np.shape(mousecam_components))
    number_of_components = np.shape(mousecam_components)[0]

    # Load Face Pixels
    face_pixels = np.load(os.path.join(mousecam_directory, "Face_Pixels.npy"))
    print("Face Pixel Shape", np.shape(face_pixels))
    face_pixels = np.transpose(face_pixels)

    # Get Face Pixel Extent
    face_y_min = np.min(face_pixels[:, 0])
    face_y_max = np.max(face_pixels[:, 0])
    face_x_min = np.min(face_pixels[:, 1])
    face_x_max = np.max(face_pixels[:, 1])

    # Create Figure
    mousecam_figure_save_directory = os.path.join(mousecam_directory, "Component_Images")
    if not os.path.exists(mousecam_figure_save_directory):
        os.mkdir(mousecam_figure_save_directory)


    for component_index in range(5):

        # Create Axis
        figure_1 = plt.figure()
        component_axis = figure_1.add_subplot(1,2,1)

        # Create Component Image
        component = mousecam_components[component_index]
        component_magnitude = np.max(np.abs(component))
        component_image = place_roi_into_mousecam(face_pixels, image_height, image_width, component)
        component_image = component_image[face_y_min:face_y_max, face_x_min:face_x_max]

        # Plot Component Image
        component_axis.set_title("Component " + str(component_index + 1).zfill(3))
        component_axis.axis('off')
        component_axis.imshow(component_image, cmap='seismic', vmin=-component_magnitude, vmax=component_magnitude)

        plt.show()



# Load Session List
session_list = Session_List.nested_session_list
session_list = Session_List.flatten_nested_list(session_list)

data_root = r"/media/matthew/Expansion1/Cortex_Wide_Opto/Experimental_Mice"
save_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"

for session in tqdm(session_list):
    view_face_components(os.path.join(save_root, session))