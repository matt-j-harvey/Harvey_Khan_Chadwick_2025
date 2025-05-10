import torch
import os
import matplotlib.pyplot as plt
import numpy as np


class rnn_model(torch.nn.Module):

    def __init__(self, n_inputs, n_neurons, n_outputs, device):
        super(rnn_model, self).__init__()

        # Save Parameters
        self.device = device
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.hidden_state = torch.zeros(size=[n_neurons], device=device)
        #self.hidden_state = torch.(np.random.uniform(low=-0.01, high=0.01, size=(n_neurons)), dtype=torch.float, device=device)

        self.rnn_cell = torch.nn.RNNCell(self.n_inputs, self.n_neurons, bias=True, device=device, dtype=torch.float32, nonlinearity='relu')

        output_weights = np.random.uniform(low=-0.01, high=0.01, size=(self.n_neurons, self.n_outputs))
        output_biases = np.random.uniform(low=-0.01, high=0.01, size=(self.n_outputs))

        # Initialise Weights
        self.output_weights = torch.nn.Parameter(torch.tensor(output_weights, dtype=torch.float, device=device))
        self.output_biases = torch.nn.Parameter(torch.tensor(output_biases, dtype=torch.float, device=device))


    def forward(self, external_input_tensor):

        number_of_samples = external_input_tensor.size(dim=0)
        output_tensor = torch.zeros(size=(number_of_samples, self.n_outputs), device=self.device)

        # Initialise Random Hidden State
        for sample_index in range(number_of_samples):

            # Put Through GRU
            hidden_state = self.rnn_cell(external_input_tensor[sample_index], self.hidden_state)

            # Get Output
            output_tensor[sample_index] = torch.matmul(hidden_state, self.output_weights) + self.output_biases


        return output_tensor




class custom_rnn_model(torch.nn.Module):

    def __init__(self, n_inputs, n_neurons, n_outputs, device):
        super(custom_rnn_model, self).__init__()

        # Save Parameters
        self.device = device
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs
        self.hidden_state = torch.zeros(size=[n_neurons], device=device)
        #self.hidden_state = torch.(np.random.uniform(low=-0.01, high=0.01, size=(n_neurons)), dtype=torch.float, device=device)
        #self.rnn_cell = torch.nn.RNNCell(self.n_inputs, self.n_neurons, bias=True, device=device, dtype=torch.float32, nonlinearity='relu')

        input_weights = np.random.uniform(low=-0.01, high=0.01, size=(self.n_inputs, self.n_neurons))
        input_biases = np.random.uniform(low=-0.01, high=0.01, size=(self.n_inputs))

        recurrent_weights = np.random.uniform(low=-0.01, high=0.01, size=(self.n_neurons, self.n_neurons))
        recurrent_biases = np.random.uniform(low=-0.01, high=0.01, size=(self.n_neurons))

        output_weights = np.random.uniform(low=-0.01, high=0.01, size=(self.n_neurons, self.n_outputs))
        output_biases = np.random.uniform(low=-0.01, high=0.01, size=(self.n_outputs))

        # Initialise Weights
        self.input_weights = torch.nn.Parameter(torch.tensor(input_weights, dtype=torch.float, device=device))
        self.input_biases = torch.nn.Parameter(torch.tensor(input_biases, dtype=torch.float, device=device))

        self.recurrent_weights = torch.nn.Parameter(torch.tensor(recurrent_weights, dtype=torch.float, device=device))
        self.recurrent_biases = torch.nn.Parameter(torch.tensor(recurrent_biases, dtype=torch.float, device=device))

        self.output_weights = torch.nn.Parameter(torch.tensor(output_weights, dtype=torch.float, device=device))
        self.output_biases = torch.nn.Parameter(torch.tensor(output_biases, dtype=torch.float, device=device))


    def forward(self, external_input_tensor):

        number_of_samples = external_input_tensor.size(dim=0)
        output_tensor = torch.zeros(size=(number_of_samples, self.n_outputs), device=self.device)

        # Initialise Random Hidden State
        for sample_index in range(number_of_samples):

            # Get Input
            input = torch.matmul(external_input_tensor[sample_index]  + self.input_biases, self.input_weights)

            # Put Through Recurrent Matrix
            hidden_state = torch.matmul(input, self.recurrent_weights) + self.recurrent_biases

            # Through Non-Linearity
            hidden_state = torch.relu(hidden_state)

            # Get Output
            output_tensor[sample_index] = torch.matmul(hidden_state, self.output_weights) + self.output_biases


        return output_tensor




class custom_rnn_model_2(torch.nn.Module):

    def __init__(self, n_inputs, n_neurons, n_outputs, device):
        super(custom_rnn_model_2, self).__init__()

        # Save Parameters
        self.device = device
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.n_outputs = n_outputs

        self.input_layer = torch.nn.Linear(self.n_inputs, self.n_neurons, bias=True)
        self.recurrent_layer = torch.nn.Linear(self.n_neurons, self.n_neurons, bias=True)
        self.activation = torch.nn.Tanh()
        self.output_layer = torch.nn.Linear(self.n_neurons, self.n_outputs, bias=True)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.recurrent_layer(x)
        x = self.activation(x)
        x = self.output_layer(x)
        return x