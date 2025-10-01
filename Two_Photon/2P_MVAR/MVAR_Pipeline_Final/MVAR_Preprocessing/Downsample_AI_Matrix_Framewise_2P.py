import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import tables
import random


def forceAspect(ax, aspect=1):
    im = ax.get_images()
    extent = im[0].get_extent()
    ax.set_aspect(abs((extent[1] - extent[0]) / (extent[3] - extent[2])) / aspect)



def normalise_trace(trace):
    trace = np.subtract(trace, np.min(trace))
    trace = np.divide(trace, np.max(trace))
    return trace



def downsammple_trace_framewise(trace, frame_times):

    # Get Average Frame Duration
    frame_duration_list = []
    for frame_index in range(1000):
        frame_duaration = frame_times[frame_index + 1] - frame_times[frame_index]
        frame_duration_list.append(frame_duaration)
    average_duration = int(np.mean(frame_duration_list))

    downsampled_trace = []
    number_of_frames = len(frame_times)
    for frame_index in range(number_of_frames-1):
        frame_start = frame_times[frame_index]
        frame_end = frame_times[frame_index + 1]
        frame_data = trace[frame_start:frame_end]
        frame_data_mean = np.mean(frame_data)
        downsampled_trace.append(frame_data_mean)

    # Add Final Frame
    final_frame_start = frame_times[number_of_frames-1]
    final_frame_end = final_frame_start + average_duration
    final_frame_data = trace[final_frame_start:final_frame_end]
    final_frame_mean = np.mean(final_frame_data)
    downsampled_trace.append(final_frame_mean)

    return downsampled_trace



def visualise_downsampling(original_trace, downsampled_trace):

    figure_1 = plt.figure()
    rows = 2
    columns = 1
    original_axis = figure_1.add_subplot(rows, columns, 1)
    downsample_axis = figure_1.add_subplot(rows, columns, 2)

    original_axis.plot(original_trace)
    downsample_axis.plot(downsampled_trace)

    plt.show()



def downsample_ai_matrix(data_root_directory, base_directory, mvar_output_directory):

    # Load Times of each 2photon Z-Stack
    frame_times = np.load(os.path.join(data_root_directory, base_directory, "Behaviour", "Stack_Onsets.npy"), allow_pickle=True)[()]

    # Load AI Recorder File
    ai_data = np.load(os.path.join(data_root_directory, base_directory, "Behaviour", "AI_Matrix.npy"))
    print("AI Data", np.shape(ai_data))

    # Extract Relevant Traces
    number_of_traces = np.shape(ai_data)[1]
    print("Number of traces", number_of_traces)

    # Create Downsampled AI Matrix
    downsampled_ai_matrix = []
    for trace_index in range(number_of_traces):
        full_trace = ai_data[:, trace_index]
        downsampled_trace = downsammple_trace_framewise(full_trace, frame_times)

        """
        figure_1 = plt.figure()
        axis_1 = figure_1.add_subplot(2, 1, 1)
        axis_2 = figure_1.add_subplot(2, 1, 2)
        axis_1.plot(full_trace)
        axis_2.plot(downsampled_trace)
        plt.show()
        """

        normalised_trace = downsampled_trace

        downsampled_ai_matrix.append(normalised_trace)


    downsampled_ai_matrix = np.array(downsampled_ai_matrix)
    print("Downsampled AI Matrix Shape", np.shape(downsampled_ai_matrix))

    # Save Downsampled AI Matrix
    save_directory = os.path.join(mvar_output_directory, base_directory, "Behaviour")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Downsampled_AI_Matrix_Framewise.npy"), downsampled_ai_matrix)

