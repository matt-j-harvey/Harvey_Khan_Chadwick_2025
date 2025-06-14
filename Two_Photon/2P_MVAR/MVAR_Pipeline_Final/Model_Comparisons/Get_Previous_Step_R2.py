import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



def visualise_matricies(preceeding_df, delta_f_matrix):

    # Caulcuate Difference
    difference_matrix = np.subtract(delta_f_matrix, preceeding_df)

    # Create Figure
    figure_1 = plt.figure()
    preceeding_axis = figure_1.add_subplot(3,1,1)
    df_axis = figure_1.add_subplot(3,1,2)
    difference_axis = figure_1.add_subplot(3,1,3)

    preceeding_axis.imshow(np.transpose(preceeding_df))
    df_axis.imshow(np.transpose(delta_f_matrix))
    difference_axis.imshow(np.transpose(difference_matrix))

    plt.show()




def get_previous_step_r2(mvar_directory, session):

    # Load Regression Matrix
    regression_matrix = np.load(os.path.join(mvar_directory, session, "Design_Matricies", "Combined_Design_Matrix_Dictionary.npy"), allow_pickle=True)[()]

    # Df Negshift is first regressor in design matrix
    design_matrix = regression_matrix['DesignMatrix']
    n_neurons = regression_matrix['Nvar']
    preceeding_df = design_matrix[:, 0:n_neurons]

    # Extract Df (y)
    delta_f_matrix = regression_matrix['dFtot']
    delta_f_matrix = np.transpose(delta_f_matrix) # Swap So Its In Form (n_timepoints x n_neurons)

    print("delta_f_matrix", np.shape(delta_f_matrix))
    print("preceeding_df", np.shape(preceeding_df))

    #Calculate R2 - R2 score takes matricies of shape (n_samples, n_outputs)
    preceeding_only_r2 = r2_score(y_true=delta_f_matrix, y_pred=preceeding_df)
    print("preceeding_only_r2", preceeding_only_r2)

    # Get Multi Values
    r2_distribution = r2_score(y_true=delta_f_matrix, y_pred=preceeding_df, multioutput='raw_values')
    plt.hist(r2_distribution)
    plt.show()

    visualise_matricies(preceeding_df, delta_f_matrix)