import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from scipy.stats import zscore


from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


def balance_data(input_data, labels):

    # Get Class Sizes
    unique_labels, unique_counts = np.unique(labels, return_counts=True)

    # Select Smallest Class
    smallest_class_size = np.min(unique_counts)

    # Get Condition Indicies
    condition_1_indicies = list(np.where(labels == 0)[0])
    condition_2_indicies = list(np.where(labels == 1)[0])

    # Get Smaple
    condition_1_sample_indicies = random.sample(condition_1_indicies, smallest_class_size)
    condition_2_sample_indicies = random.sample(condition_2_indicies, smallest_class_size)

    combined_indicies = np.concatenate([condition_1_sample_indicies, condition_2_sample_indicies])

    balanced_data = input_data[combined_indicies]
    balanced_labels = labels[combined_indicies]

    return balanced_data, balanced_labels



def perform_cv(model, x_all, y_all, n_balance_iterations=20, n_folds=5):

    balance_score_list = []
    balance_coef_list = []
    for balance_interation in range(n_balance_iterations):

        # Balance Classes By Randomly subsampling From Larger Class
        x_balanced, y_balanced = balance_data(x_all, y_all)

        # Get N Fold Structure
        skf = StratifiedKFold(n_splits=n_folds, random_state=42, shuffle=True)

        # Iterate Through Each Fold
        fold_score_list = []
        fold_coef_list = []
        for fold_index, (train_index, test_index) in enumerate(skf.split(x_balanced, y_balanced)):

            # Split Into Test and Train
            x_train = x_balanced[train_index]
            y_train = y_balanced[train_index]
            x_test = x_balanced[test_index]
            y_test = y_balanced[test_index]

            # Z Score Seperately
            x_train = zscore(x_train, axis=0)
            x_test = zscore(x_test, axis=0)
            x_train = np.nan_to_num(x_train)
            x_test = np.nan_to_num(x_test)

            # Train Model
            model.fit(X=x_train, y=y_train)

            # Get prediction
            y_pred = model.predict(x_test)

            # Score Prediction
            score = accuracy_score(y_true=y_test, y_pred=y_pred)
            fold_score_list.append(score)

            # Get Coefs
            coefs = model.coef_
            fold_coef_list.append(coefs)

        # Get Average Score Across Folds
        fold_average_score = np.mean(fold_score_list)
        fold_average_coefs = np.mean(np.array(fold_coef_list), axis=0)

        balance_score_list.append(fold_average_score)
        balance_coef_list.append(fold_average_coefs)

    # Get Average Across Balance Subsamples
    average_score = np.mean(balance_score_list)
    average_coefs = np.mean(np.array(balance_coef_list), axis=0)

    return average_score, average_coefs



def perform_shuffled_decoding(model, x_all, y_all, n_balance_iterations=1, n_folds=5, n_shuffles=20):

    shuffled_result_list = []
    for shuffle_index in range(n_shuffles):
        y_shuffle = np.copy(y_all)
        np.random.shuffle(y_shuffle)
        average_score, average_coefs = perform_cv(model, x_all, y_shuffle, n_balance_iterations=n_balance_iterations, n_folds=n_folds)
        shuffled_result_list.append(average_score)

    mean_shuffled_result = np.mean(shuffled_result_list, axis=0)
    return mean_shuffled_result