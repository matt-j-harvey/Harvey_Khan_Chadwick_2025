import numpy as np
import pandas as pd

def sample_evenly_across_mice(trial_df,
                              n_to_sample,
                              mouse_column="mouse",
                              random_state=None):

    """
    Sample trials while trying to balance contributions across mice.
    """

    # -----------------------------------------
    # Step 0 — Handle edge cases
    # -----------------------------------------
    if n_to_sample == 0 or len(trial_df) == 0:
        return trial_df.iloc[[]]

    # -----------------------------------------
    # Step 1 — Get list of mice
    # -----------------------------------------
    mouse_ids = trial_df[mouse_column].unique()
    n_mice = len(mouse_ids)

    # -----------------------------------------
    # Step 2 — Decide target number per mouse
    # -----------------------------------------
    base_trials_per_mouse = n_to_sample // n_mice
    extra_trials = n_to_sample % n_mice

    # Example:
    # n_to_sample = 10, n_mice = 3
    # → base = 3, extra = 1
    # → one mouse gets 4, others get 3

    # -----------------------------------------
    # Step 3 — Shuffle mouse order
    # (so the "extra" trials are random)
    # -----------------------------------------
    rng = np.random.default_rng(random_state)
    mouse_ids = rng.permutation(mouse_ids)

    sampled_trials_list = []

    # -----------------------------------------
    # Step 4 — Sample from each mouse
    # -----------------------------------------
    for i, mouse_id in enumerate(mouse_ids):

        # Get trials for this mouse
        mouse_trials = trial_df[trial_df[mouse_column] == mouse_id]

        # Decide how many we WANT from this mouse
        target_n = base_trials_per_mouse

        # Give extra trial to first few mice
        if i < extra_trials:
            target_n += 1

        # But cannot take more than available
        actual_n = min(target_n, len(mouse_trials))

        # Sample from this mouse
        if actual_n > 0:
            sampled_mouse_trials = mouse_trials.sample(
                n=actual_n,
                random_state=random_state,
            )

            sampled_trials_list.append(sampled_mouse_trials)

    # -----------------------------------------
    # Step 5 — Combine all sampled trials
    # -----------------------------------------
    if len(sampled_trials_list) > 0:
        sampled_df = pd.concat(sampled_trials_list)
    else:
        sampled_df = trial_df.iloc[[]]

    # -----------------------------------------
    # Step 6 — Check if we still need more trials
    # (because some mice had too few)
    # -----------------------------------------
    current_n = len(sampled_df)
    n_missing = n_to_sample - current_n

    if n_missing > 0:

        # Get trials not yet used
        remaining_trials = trial_df.drop(index=sampled_df.index)

        # Take extra from remaining pool
        n_extra = min(n_missing, len(remaining_trials))

        if n_extra > 0:
            extra_trials_df = remaining_trials.sample(
                n=n_extra,
                random_state=random_state,
            )

            sampled_df = pd.concat([sampled_df, extra_trials_df])

    return sampled_df




def match_rt_distributions_across_groups(
    trial_df,
    rt_bin_starts,
    rt_bin_stops,
    group_column="group",
    mouse_column="mouse",
    rt_column="reaction_time",
    group_a=0,
    group_b=1,
):
    """
    Match reaction time distributions between two groups.

    Parameters
    ----------
    rt_bin_starts : list or array
    rt_bin_stops  : list or array

    Each (start, stop) pair defines one RT bin.
    """

    matched_dataframes = []
    summary_rows = []

    # Iterate through bins
    for bin_index in range(len(rt_bin_starts)):
        bin_start = rt_bin_starts[bin_index]
        bin_stop = rt_bin_stops[bin_index]


        # Select trials in this bin for each group
        group_a_trials = trial_df[(trial_df[group_column] == group_a) & (trial_df[rt_column] >= bin_start) & (trial_df[rt_column] < bin_stop)]
        group_b_trials = trial_df[(trial_df[group_column] == group_b) & (trial_df[rt_column] >= bin_start) & (trial_df[rt_column] < bin_stop)]
        n_a_available = len(group_a_trials)
        n_b_available = len(group_b_trials)

        # Determine how many to sample
        n_to_sample = min(n_a_available, n_b_available)


        if n_to_sample == 0:

            summary_rows.append({
                "bin_start": bin_start,
                "bin_stop": bin_stop,
                "n_group_a_available": n_a_available,
                "n_group_b_available": n_b_available,
                "n_sampled_per_group": 0,
            })

            continue


        # Sample evenly across mice
        sampled_group_a = sample_evenly_across_mice(group_a_trials, n_to_sample=n_to_sample)
        sampled_group_b = sample_evenly_across_mice(group_b_trials, n_to_sample=n_to_sample)

        # Store matched data
        matched_dataframes.append(sampled_group_a)
        matched_dataframes.append(sampled_group_b)

        # Store summary info
        summary_rows.append({
            "bin_start": bin_start,
            "bin_stop": bin_stop,
            "n_group_a_available": n_a_available,
            "n_group_b_available": n_b_available,
            "n_sampled_per_group": n_to_sample,
            "n_group_a_sampled": len(sampled_group_a),
            "n_group_b_sampled": len(sampled_group_b),
            "n_group_a_mice": sampled_group_a[mouse_column].nunique(),
            "n_group_b_mice": sampled_group_b[mouse_column].nunique(),
        })

    # Combine all matched data
    if len(matched_dataframes) > 0:
        matched_df = pd.concat(matched_dataframes).reset_index(drop=True)
    else:
        matched_df = trial_df.iloc[[]].copy()

    summary_df = pd.DataFrame(summary_rows)

    return matched_df, summary_df