import numpy as np
import os
import matplotlib.pyplot as plt



def saddle_form(x, r):
    return r + x**2



x_values = np.linspace(start=-5, stop=5, num=100)
y_values = []

for value in x_values:
    y_values.append(saddle_form(x_values))


