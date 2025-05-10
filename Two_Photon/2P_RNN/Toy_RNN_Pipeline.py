import os
import matplotlib.pyplot as plt
import numpy as np
import torch

import RNN_Model
import Train_RNN

# Set Save Directory
save_directory = r"C:\Users\matth\Documents\RNN_Play\RNN_Weights"
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

# Load Data
data_directory = r"C:\Users\matth\Documents\RNN_Play\Test_Data"
input_data = np.load(os.path.join(data_directory, "Input_Data.npy"))
output_data = np.load(os.path.join(data_directory, "Output_Data.npy"))

# Create RNN
n_inputs = np.shape(input_data)[1]
n_outputs = np.shape(output_data)[1]
n_neurons = 250
device = torch.device('cpu')
rnn = RNN_Model.custom_rnn_model(n_inputs, n_neurons, n_outputs, device)
#rnn.load_state_dict(torch.load(os.path.join(save_directory, 'model.pth')))


# Train Model
Train_RNN.fit_model(rnn, input_data, output_data, save_directory)
