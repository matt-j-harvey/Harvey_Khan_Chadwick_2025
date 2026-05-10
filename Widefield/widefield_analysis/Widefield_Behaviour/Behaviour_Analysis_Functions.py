import numpy as np
from scipy.stats import norm

def extreme_value_corrections(selected_value, number_of_trials):

    if selected_value == 0:
        selected_value = float(1) / number_of_trials

    elif selected_value == 1:
        selected_value = float((number_of_trials - 1)) / number_of_trials

    return selected_value


def calculate_d_prime(hits, misses, false_alarms, correct_rejections):

    # Calculate Hit Rates and False Alarm Rates
    number_of_rewarded_trials = hits + misses
    number_of_unrewarded_trials = false_alarms + correct_rejections

    if number_of_unrewarded_trials == 0 or number_of_rewarded_trials == 0:
        return np.nan
    else:

        hit_rate = float(hits) / number_of_rewarded_trials
        false_alarm_rate = float(false_alarms) / number_of_unrewarded_trials

        # Ensure Either Value Does Not Equal Zero or One
        hit_rate = extreme_value_corrections(hit_rate, number_of_rewarded_trials)
        false_alarm_rate = extreme_value_corrections(false_alarm_rate, number_of_unrewarded_trials)

        # Get The Standard Normal Distribution
        Z = norm.ppf

        # Z Transform Both The Hit Rates And The False Alarm Rates
        hit_rate_z_transform = Z(hit_rate)
        false_alarm_rate_z_transform = Z(false_alarm_rate)

        # Calculate D Prime
        d_prime = hit_rate_z_transform - false_alarm_rate_z_transform

    d_prime = np.around(d_prime, 2)
    return d_prime




def get_session_visual_performance(behaviour_matrix):

    # Create Variables To Hold Trial Outcomes
    n_hits = 0
    n_misses = 0
    n_fas = 0
    n_crs = 0

    # Iterate Through Each Trial
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        opto_status = trial[22]
        first_in_block = trial[9]

        if opto_status == False and first_in_block == 0:
            if trial_type == 1:

                if correct == 1:
                    n_hits += 1
                elif correct == 0:
                    n_misses += 1

            elif trial_type == 2:
                if correct == 1:
                    n_crs += 1
                elif correct == 0:
                    n_fas += 1

    # Get d Prime
    d_prime = calculate_d_prime(n_hits, n_misses, n_fas, n_crs)

    return d_prime



def get_session_odour_performance(behaviour_matrix):

    # Create Variables To Hold Trial Outcomes
    n_hits = 0
    n_misses = 0
    n_fas = 0
    n_crs = 0

    # Iterate Through Each Trial
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        trial_block = trial[8]
        opto_status = trial[22]
        first_in_block = trial[9]


        if opto_status == False and first_in_block == 0:

            if trial_type == 3:

                if correct == 1:
                    n_hits += 1
                elif correct == 0:
                    n_misses += 1

            elif trial_type == 4:
                if correct == 1:
                    n_crs += 1
                elif correct == 0:
                    n_fas += 1

    # Get d Prime
    d_prime = calculate_d_prime(n_hits, n_misses, n_fas, n_crs)

    return d_prime


def get_session_irrel_performance(behaviour_matrix):

    # Create Variables To Hold Trial Outcomes
    n_hits = 0
    n_misses = 0
    n_fas = 0
    n_crs = 0

    # Iterate Through Each Trial
    for trial in behaviour_matrix:
        preceeded_by_irrel = trial[5]
        irrel_type = trial[6]
        ignore_irrel = trial[7]
        trial_block = trial[8]
        opto_status = trial[22]
        first_in_block = trial[9]

        if opto_status == False and first_in_block == 0:

            if preceeded_by_irrel == 1:
                if irrel_type == 1:

                    if ignore_irrel == 0:
                        n_hits += 1
                    elif ignore_irrel == 1:
                        n_misses += 1

                elif irrel_type == 2:
                    if ignore_irrel == 1:
                        n_crs += 1
                    elif ignore_irrel == 0:
                        n_fas += 1

    # Get d Prime
    d_prime = calculate_d_prime(n_hits, n_misses, n_fas, n_crs)

    return d_prime




def get_visual_block_performance(behaviour_matrix, block_number):

    # Create Variables To Hold Trial Outcomes
    n_hits = 0
    n_misses = 0
    n_fas = 0
    n_crs = 0

    # Iterate Through Each Trial
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        trial_block = trial[8]
        opto_status = trial[22]
        first_in_block = trial[9]

        if trial_block == block_number:
            if opto_status == False and first_in_block == 0:
                if trial_type == 1:

                    if correct == 1:
                        n_hits += 1
                    elif correct == 0:
                        n_misses += 1

                elif trial_type == 2:
                    if correct == 1:
                        n_crs += 1
                    elif correct == 0:
                        n_fas += 1

    # Get d Prime
    d_prime = calculate_d_prime(n_hits, n_misses, n_fas, n_crs)

    return d_prime



def get_odour_block_performance(behaviour_matrix, block_number):

    # Create Variables To Hold Trial Outcomes
    n_hits = 0
    n_misses = 0
    n_fas = 0
    n_crs = 0

    # Iterate Through Each Trial
    for trial in behaviour_matrix:
        trial_type = trial[1]
        correct = trial[3]
        trial_block = trial[8]
        opto_status = trial[22]
        first_in_block = trial[9]

        if trial_block == block_number:
            if opto_status == False and first_in_block == 0:

                if trial_type == 3:

                    if correct == 1:
                        n_hits += 1
                    elif correct == 0:
                        n_misses += 1

                elif trial_type == 4:
                    if correct == 1:
                        n_crs += 1
                    elif correct == 0:
                        n_fas += 1

    # Get d Prime
    d_prime = calculate_d_prime(n_hits, n_misses, n_fas, n_crs)

    return d_prime


def get_odour_block_irrel_performance(behaviour_matrix, block_number):

    # Create Variables To Hold Trial Outcomes
    n_hits = 0
    n_misses = 0
    n_fas = 0
    n_crs = 0

    # Iterate Through Each Trial
    for trial in behaviour_matrix:
        irrel_type = trial[6]
        preceeded_by_irrel = trial[5]
        ignore_irrel = trial[7]
        trial_block = trial[8]
        opto_status = trial[22]
        first_in_block = trial[9]

        if trial_block == block_number:
            if opto_status == False and first_in_block == 0:

                if preceeded_by_irrel == 1:
                    if irrel_type == 1:

                        if ignore_irrel == 0:
                            n_hits += 1
                        elif ignore_irrel == 1:
                            n_misses += 1

                    elif irrel_type == 2:
                        if ignore_irrel == 1:
                            n_crs += 1
                        elif ignore_irrel == 0:
                            n_fas += 1

    # Get d Prime
    d_prime = calculate_d_prime(n_hits, n_misses, n_fas, n_crs)

    return d_prime

