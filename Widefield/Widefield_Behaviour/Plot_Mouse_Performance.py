import numpy as np
import matplotlib.pyplot as plt




def plot_performance(visual_list, odour_list):

    # Get X Values
    visual_x_values = [0, 1, 2, 3]
    odour_x_values = [1, 3]

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    # Plot Visual
    n_mice = len(visual_list)
    for mouse_index in range(n_mice):
        axis_1.plot(visual_x_values, visual_list[mouse_index])
        axis_1.scatter(visual_x_values, visual_list[mouse_index])
        axis_1.scatter(odour_x_values, odour_list[mouse_index])

    axis_1.set_ylim([0, 3.5])
    axis_1.plot()

    plt.show()