import numpy as np
import matplotlib.pyplot as plt



def get_trajectory(x, y, stim):

    trajectory = []
    current_state = x
    for x in range(10):
        trajectory.append(current_state)
        current_state = current_state * y
        current_state = current_state + stim

    plt.plot(trajectory)
    plt.show()



y = 0.3
initial_state = 0
stim = 0.8

get_trajectory(initial_state, y, stim)
