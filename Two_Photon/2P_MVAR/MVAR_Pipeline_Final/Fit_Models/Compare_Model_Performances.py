import numpy as np
import matplotlib.pyplot as plt
import os



"""
1 - Previous Step Only
2 - Just stimuli and behaviour regressors
3 - With recurrent weights
4 - Different recurrent weights for each context
5 - Time varying recurrent weights
"""

def plot_model_performances(mvar_directory, session_list):


    for session in session_list:

        standard_scores = np.load(os.path.join())


