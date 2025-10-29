import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


def get_correct_go_trials(behaviour_matrix):

    correct_goal_trial_list = []


    n_trials = np.shape(behaviour_matrix)[0]
    for trial_index in range(n_trials):
        trial_data = behaviour_matrix[trial_index]
        trial_type = trial_data[1]
        trial_outcome = trial_data[6]

        if trial_type == 0 and trial_outcome == 1:
            correct_goal_trial_list.append(trial_index)

    return correct_goal_trial_list

    correct_go_reaction_times = matrix[correct_goal_trial_list, 4]


def calculate_delay_performance(matrix):

    # Get List of Trial Types
    trial_types = matrix[:, 1]

    # Get Indicies of Delay Trials
    delay_indicies = np.where(trial_types == 1)

    # Calculate Number Of Delay Trials
    number_of_delay_trials = np.shape(delay_indicies)[1]
    print("number_of_delay_trials", number_of_delay_trials)

    # Get Outcomes of the delay trials
    delay_outcomes = matrix[delay_indicies, 6]

    # Sum up Correct Trials
    delay_correct_trials = np.where(delay_outcomes == 1, 1, 0)
    number_of_delay_correct = np.sum(delay_correct_trials)

    percentage_correct = (number_of_delay_correct / number_of_delay_trials) * 100

    return percentage_correct



def plot_performance_graph(go_percentage, delay_percentage, stop_percentage):

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)  # N rows, N columns, which graph this is

    axis_1.plot([1, 2, 3],[go_percentage, delay_percentage, stop_percentage])
    axis_1.set_ylim([0, 110])
    axis_1.set_xticks([1, 2, 3], labels=["go percentage", "delay percentage", "stop percentage"])
    axis_1.set_title("Mouse Performance")
    axis_1.spines[['right', 'top']].set_visible(False)


    plt.show()



# Pipeline

matrix_file = r"C:\Local_data\Behaviour_Data\Cohort_2\Blue\2025_09_18\Behaviour_Matrix.npy"
matrix = np.load(matrix_file)

# First Get Go Trials Correct
percentage_correct = calculate_delay_performance(matrix)

plot_performance_graph(100, 80, 90)
# Then Get Delay Trials

# Then Get Stop Trials


#



histogram_data = [1,1,1,2,4,5,6,3,2,6,6,3,2,6,3,4,5,2,3,3,4,2,3,4,3,2,4]

figure_1 = plt.figure()
axis_1 = figure_1.add_subplot(1,1,1)
axis_1.hist(histogram_data, alpha=0.5)
plt.show()

print("percentage_correct", percentage_correct)


