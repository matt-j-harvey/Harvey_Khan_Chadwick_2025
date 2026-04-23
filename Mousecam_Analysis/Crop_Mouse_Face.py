import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from matplotlib.path import Path

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import pyqtgraph

import os
import sys

from tqdm import tqdm

import Session_List


pyqtgraph.setConfigOptions(imageAxisOrder='row-major')



def get_video_name(base_directory):

    file_list = os.listdir(base_directory)
    for file in file_list:
        if "_cam_1.mp4" in file:
            return file


def load_image_still(video_file, n_frames=100):

    # Open Video File
    cap = cv2.VideoCapture(video_file)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print("Frame count", frameCount)
    # Get Evenly Spread Frames
    selected_frames = np.linspace(start=0, stop=frameCount-1, num=n_frames, dtype=int)
    selected_frames_list = []
    """
    frame_index = 0
    ret = True
    while (frame_index < frameCount and ret):
        ret, frame = cap.read()
        if frame_index in selected_frames:
            selected_frames_list.append(frame)

        frame_index += 1
    """

    # Extract Selected Frames
    for frame in selected_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ret, frame = cap.read()
        selected_frames_list.append(frame)

    cap.release()

    return selected_frames_list


class face_selection_window(QWidget):

    def __init__(self, data_root, session_list, parent=None):
        super(face_selection_window, self).__init__(parent)

        # Setup Window
        self.setWindowTitle("ROI Selector")
        self.setGeometry(0, 0, 1900, 500)

        # Setup Internal Variables
        self.data_root = data_root
        self.session_list = session_list
        self.number_of_sessions = len(self.session_list)
        self.current_session_index = 0

        # Regressor Display View
        self.mousecam_display_view_widget = QWidget()
        self.mousecam_display_view_widget_layout = QGridLayout()
        self.mousecam_display_view = pyqtgraph.ImageView()
        self.mousecam_display_view.ui.histogram.hide()
        self.mousecam_display_view.ui.roiBtn.hide()
        self.mousecam_display_view.ui.menuBtn.hide()
        self.mousecam_display_view_widget_layout.addWidget(self.mousecam_display_view, 0, 0)
        self.mousecam_display_view_widget.setLayout(self.mousecam_display_view_widget_layout)
        self.mousecam_display_view_widget.setMinimumWidth(1500)
        self.mousecam_display_view_widget.setMinimumHeight(800)
        cm = pyqtgraph.colormap.get('CET-R4')
        self.mousecam_display_view.setColorMap(cm)

        # Create Session List Views
        self.session_list_widget = QListWidget()
        self.session_list_widget.currentItemChanged.connect(self.change_session)
        self.image_list = []

        # Load Session Data
        for session in tqdm(session_list, desc="Loading Sessions"):
            print(session)

            # Add Session To List
            self.session_list_widget.addItem(session)

            # Get Video Name
            video_name = get_video_name(os.path.join(self.data_root, session))

            # Load Video Frame
            frame_list = load_image_still(os.path.join(self.data_root, session, video_name))

            # Add To List
            self.image_list.append(frame_list)

        # Set Current Images
        self.current_image = self.image_list[0][0]
        self.mousecam_display_view.setImage(self.current_image)

        # Create Map Button
        self.map_button = QPushButton("Map Region")
        self.map_button.clicked.connect(self.map_region)

        # Create Frame Slider
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(len(self.image_list[self.current_session_index])-1)
        self.frame_slider_label = QLabel("Frame: 0")
        self.frame_slider.valueChanged.connect(self.change_frame)

        self.session_list_widget.setCurrentRow(self.current_session_index)

        # Create and Set Layout
        self.layout = QGridLayout()
        self.setLayout(self.layout)

        # Add Display Views
        self.layout.addWidget(self.mousecam_display_view_widget, 0, 0, 1, 10)
        self.layout.addWidget(self.frame_slider, 1, 0, 1, 8)
        self.layout.addWidget(self.frame_slider_label, 1, 8, 1, 1)
        self.layout.addWidget(self.map_button, 1, 9, 1, 1)


        # Add List Views
        self.layout.addWidget(self.session_list_widget, 0, 11, 2, 1)

        # Add ROI
        self.whisker_roi = pyqtgraph.PolyLineROI(positions=[[500, 100], [550, 100], [550, 150], [500, 150]], closed=True)
        self.mousecam_display_view.addItem(self.whisker_roi)

    def change_frame(self):
        current_frame = int(self.frame_slider.value())
        self.current_image = self.image_list[self.current_session_index][current_frame]
        self.mousecam_display_view.setImage(self.current_image)

    def change_session(self):
        self.current_session_index = int(self.session_list_widget.currentRow())
        self.change_frame()

    def map_region(self):

        print("Current image Shaoe", self.current_image.shape)
        cols, rows, colours = self.current_image.shape
        m = np.mgrid[:cols, :rows]
        possx = m[0, :, :]  # make the x pos array
        possy = m[1, :, :]  # make the y pos array
        possx.shape = cols, rows
        possy.shape = cols, rows
        mpossx = self.whisker_roi.getArrayRegion(possx, self.mousecam_display_view.imageItem).astype(int)
        mpossx = mpossx[np.nonzero(mpossx)]  # get the x pos from ROI
        mpossy = self.whisker_roi.getArrayRegion(possy, self.mousecam_display_view.imageItem).astype(int)
        mpossy = mpossy[np.nonzero(mpossy)]  # get the y pos from ROI


        face_pixels = np.array([mpossx, mpossy])
        print("Face Pixels", np.shape(face_pixels))

        # Save These
        save_directory = os.path.join(self.data_root, self.session_list[self.current_session_index], "Mousecam_Analysis")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
      
        np.save(os.path.join(save_directory, "Face_Pixels.npy"), face_pixels)
        print("Saved!", os.path.join(save_directory, "Face_Pixels.npy"))

    """
    def map_region(self):

        roi_handles = self.whisker_roi.getLocalHandlePositions()

        polygon_verticies = []
        for handle in roi_handles:
            handle = handle[1]

            if type(handle) == QPointF:
                handle_coords = [int(handle.x()), int(handle.y())]

            else:
                handle = np.array(handle)
                handle_coords = [int(handle[0]), int(handle[1])]
            polygon_verticies.append(handle_coords)

        image_height, image_width, rgb_depth = np.shape(self.current_image)
        x, y = np.meshgrid(np.arange(image_width), np.arange(image_height))  # make a canvas with coordinates
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x, y)).T

        p = Path(polygon_verticies)  # make a polygon
        grid = p.contains_points(points)
        grid = np.reshape(grid, (image_height, image_width))

        whisker_coords = np.nonzero(grid)

        # Save These
        save_directory = os.path.join(self.session_list[self.current_session_index], "Mousecam_Analysis")
        if not os.path.exists(save_directory):
            os.mkdir(save_directory)
  
        np.save(os.path.join(save_directory, "Face_Pixels.npy"), whisker_coords)
        """


if __name__ == '__main__':

    app = QApplication(sys.argv)

    # Load Session List
    #nested_session_list = Session_List.nested_session_list
    #flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    #data_root = r"/media/matthew/29D46574463D2856/Harvey_Khan_Data/Widefield_Opto"

    nested_session_list = Session_List.control_all_post_learning_list
    flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    data_root = r"/media/matthew/Expansion/Control_Data"

    nested_session_list = Session_List.neurexin_all_post_learning
    flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

    nested_session_list = Session_List.neurexin_pre_learning_list
    flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

    nested_session_list = Session_List.neurexin_intermediate_learning_sessions
    flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    data_root = r"/media/matthew/External_Harddrive_1/Neurexin_Data"

    nested_session_list = Session_List.control_pre_learning
    flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    data_root = r"/media/matthew/Expansion/Control_Data"

    nested_session_list = Session_List.control_intermediate_learning_sessions
    flat_session_list = Session_List.flatten_nested_list(nested_session_list)
    data_root = r"/media/matthew/Expansion/Control_Data"

    selection_window = face_selection_window(data_root, flat_session_list)
    selection_window.show()

    app.exec_()

