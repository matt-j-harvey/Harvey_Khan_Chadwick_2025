import torch
import numpy as np
from tqdm import tqdm
import os

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import r2_score


def train_epoch_batched(model, input_matrix, output_matrix, crtierion, optimiser, batch_size=500):

    n_samples = np.shape(input_matrix)[0]
    n_batches = int(np.divide(n_samples, batch_size))
    loss_list = []

    for batch_index in range(n_batches):

        # Get Input Batch
        batch_start = batch_index * batch_size
        batch_stop = batch_start + batch_size
        batch_input = input_matrix[batch_start:batch_stop]
        batch_output = output_matrix[batch_start:batch_stop]

        # Clear Gradients
        optimiser.zero_grad()

        # Get Model Prediction
        prediction = model(batch_input)

        # Get Loss
        estimation_loss = crtierion(prediction, batch_output)

        # Get Gradients
        estimation_loss.backward()

        # Update Weights
        optimiser.step()

        batch_loss = estimation_loss.detach().numpy()
        loss_list.append(batch_loss)

    mean_loss = np.mean(loss_list)
    return mean_loss


def train_epoch(model, input_matrix, output_matrix, crtierion, optimiser):

    # Clear Gradients
    optimiser.zero_grad()

    # Get Model Prediction
    prediction = model(input_matrix)

    # Get Loss
    estimation_loss = crtierion(prediction, output_matrix)

    # Get Gradients
    estimation_loss.backward()

    # Update Weights
    optimiser.step()

    mean_loss = estimation_loss.detach().numpy()

    return mean_loss


def get_validation_loss(model, input_matrix, output_matrix, crtierion):

    with torch.no_grad():

        # Get Prediction
        prediction = model(input_matrix)

        # Get Loss
        estimation_loss = crtierion(prediction, output_matrix)

    return estimation_loss


def training_stopping_criteria(validation_loss_list, patience=3):

    if len(validation_loss_list) <= patience:
        return False

    else:
        for x in range(1, patience+1):
            if validation_loss_list[-x] < validation_loss_list[-(patience+1)]:
                return False

        return True



def train_model(model, input_matrix, output_matrix):

    # Set Training Parameters
    criterion = torch.nn.MSELoss()
    learning_rate = 0.001
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.001)

    # Split Train Into Validation
    x_train, x_validation, y_train, y_validation = train_test_split(input_matrix, output_matrix, train_size=0.8, shuffle=True, random_state=42)

    # Convert To Tensors
    x_train = torch.tensor(x_train, dtype=torch.float32)
    x_validation = torch.tensor(x_validation, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    y_validation = torch.tensor(y_validation, dtype=torch.float32)

    # Training Loop
    epoch = 1
    validation_loss_list = []
    train_loss_list = []

    stopping = False
    while stopping == False:

        # Train Network
        train_loss = train_epoch(model, x_train, y_train, criterion, optimiser)

        # Check Validation Loss
        validation_loss = get_validation_loss(model, x_validation, y_validation, criterion)

        # Add To Lists
        train_loss_list.append(train_loss)
        validation_loss_list.append(validation_loss)
        #print("Epoch:", str(epoch).zfill(5), "train_loss:", np.around(train_loss, 6), "validation_loss:", validation_loss)

        # Check Stopping Criteria
        stopping = training_stopping_criteria(validation_loss_list, patience=5)

        # Increment Epoch
        epoch += 1

    return model, train_loss_list, validation_loss_list

