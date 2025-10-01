import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score



def forceAspect(ax,aspect=1):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


def visualise_matricies(preceeding_df, delta_f_matrix):

    # Caulcuate Difference
    difference_matrix = np.subtract(delta_f_matrix, preceeding_df)

    vmin = np.percentile(np.abs(delta_f_matrix), q=5)
    vmax = np.percentile(np.abs(delta_f_matrix), q=95)
    diff_magnitude = np.percentile(np.abs(difference_matrix), q=95)

    # Create Figure
    figure_1 = plt.figure()
    preceeding_axis = figure_1.add_subplot(3,1,1)
    df_axis = figure_1.add_subplot(3,1,2)
    difference_axis = figure_1.add_subplot(3,1,3)

    preceeding_axis.imshow(np.transpose(preceeding_df), vmin=vmin, vmax=vmax)
    df_axis.imshow(np.transpose(delta_f_matrix),  vmin=vmin, vmax=vmax)
    difference_axis.imshow(np.transpose(difference_matrix),  vmin=-diff_magnitude, vmax=diff_magnitude, cmap="bwr")

    forceAspect(preceeding_axis)
    forceAspect(df_axis)
    forceAspect(difference_axis)

    plt.show()


def plot_traces(df_matrix, preceeding_df, selected_index):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    real_mean = np.mean(df_matrix[:, selected_index])
    axis_1.plot(preceeding_df[:, selected_index], c='g')
    axis_1.plot(df_matrix[:, selected_index], c='b')
    axis_1.axhline(real_mean, c='k', linestyle='dashed')
    plt.show()


def view_single_trace(r2_list, preceeding_df, df_matrix):

    min_r2 = np.min(r2_list)
    max_r2 = np.max(r2_list)

    min_index = list(r2_list).index(min_r2)
    max_index = list(r2_list).index(max_r2)

    print("min r2", min_r2)
    plot_traces(df_matrix, preceeding_df, min_index)
    plot_traces(df_matrix, preceeding_df, max_index)


def shift_matrix_sanity_check(df_matrix, preceeding_matrix, n_trials, n_t, n_neurons):

    df_matrix = np.reshape(df_matrix, (n_trials, n_t, n_neurons))
    preceeding_matrix = np.reshape(preceeding_matrix, (n_trials, n_t, n_neurons))

    df_matrix = df_matrix[:, 0:-1]
    preceeding_matrix = preceeding_matrix[:, 1:]

    df_matrix = np.reshape(df_matrix, ((n_trials * (n_t-1), n_neurons)))
    preceeding_matrix = np.reshape(preceeding_matrix, ((n_trials * (n_t-1), n_neurons)))

    return df_matrix, preceeding_matrix



def get_previous_step_r2(mvar_directory, session):

    # Load Regression Matrix
    regression_matrix = np.load(os.path.join(mvar_directory, session, "Design_Matricies", "Standard_Design_Matrix_Dict.npy"), allow_pickle=True)[()]
    n_trials = np.sum(regression_matrix["N_trials"])
    n_t = regression_matrix["Nt"]
    n_neurons = regression_matrix['Nvar']

    # Df Negshift is first regressor in design matrix
    design_matrix = regression_matrix['DesignMatrix']
    preceeding_df = design_matrix[:, 0:n_neurons]

    # Extract Df (y)
    delta_f_matrix = regression_matrix['dFtot']
    delta_f_matrix = np.transpose(delta_f_matrix) # Swap So Its In Form (n_timepoints x n_neurons)

    # Sanity Check
    #delta_f_matrix, preceeding_df = shift_matrix_sanity_check(delta_f_matrix, preceeding_df, n_trials, n_t, n_neurons)

    #Calculate R2 - R2 score takes matricies of shape (n_samples, n_outputs)
    preceeding_only_r2 = r2_score(y_true=delta_f_matrix, y_pred=preceeding_df)
    #print("preceeding_only_r2", str(np.around(preceeding_only_r2*100, 1)) + "%")

    # Save This
    save_directory = os.path.join(mvar_directory, session, "Model_Comparisons")
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    np.save(os.path.join(save_directory, "Previous_Step_R2.npy"), preceeding_only_r2)

    return preceeding_only_r2

    # Get Multi Values
    #r2_distribution = r2_score(y_true=delta_f_matrix, y_pred=preceeding_df, multioutput='raw_values')
