import numpy as np
import os
import torch

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score

import RNN_Model
import Train_RNN


def get_r2(model, input_matrix, output_matrix):

    # Convert To Tensor
    input_matrix = torch.tensor(input_matrix, dtype=torch.float32)
    output_matrix = torch.tensor(output_matrix, dtype=torch.float32)

    # Get Model Prediction
    with torch.no_grad():
        y_pred = model(input_matrix).detach().numpy()
        y_real = output_matrix.detach().numpy()
        r2 = r2_score(y_true=y_real, y_pred=y_pred)
        return r2


def perform_k_fold_validation(n_model_units, device, input_matrix, output_matrix, save_directory, n_folds=5):

    # Create Model
    n_inputs = np.shape(input_matrix)[1]
    n_outputs = np.shape(output_matrix)[1]

    # Create K Fold Object
    k_fold_object = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Test and Train each Fold
    r2_list = []
    for i, (train_index, test_index) in enumerate(k_fold_object.split(input_matrix)):

        # Split Data
        x_train = input_matrix[train_index]
        x_test = input_matrix[test_index]
        y_train = output_matrix[train_index]
        y_test = output_matrix[test_index]

        # Create Model
        model = RNN_Model.custom_rnn_model_2(n_inputs, n_model_units, n_outputs, device)

        # Create Fold Save Directory
        fold_save_directory = os.path.join(save_directory, "Fold_" + str(i))
        if not os.path.exists(fold_save_directory):
            os.makedirs(fold_save_directory)

        # Train Model
        model, train_loss_list, validation_loss_list = Train_RNN.train_model(model, x_train, y_train)

        # Get Test Score
        r2 = get_r2(model, x_test, y_test)
        print(r2)
        r2_list.append(r2)

    # Get Mean R2
    mean_r2 = np.mean(r2_list)

    return mean_r2
