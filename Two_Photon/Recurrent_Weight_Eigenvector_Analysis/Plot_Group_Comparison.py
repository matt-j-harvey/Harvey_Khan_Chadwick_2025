import Plotting_Functions.Plotting_Functions as Plotting_Functions



def plot_group_comparison(wt_session_list, wt_output_root, hom_session_list, hom_output_root):

    """
    #Plotting_Functions.compare_transfer_trajectories(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    """


    # Stimulus Alignment
    #Plotting_Functions.compare_stimuli_lick_cd_alignment(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_left_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_observability_stim_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

    # Compare eigenspectrums
    #Plotting_Functions.compare_eigenspectrum_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

    # Output Alignment
    #Plotting_Functions.compare_controlability_lick_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_right_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_non_normality(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_lick_reachability_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

    # Analyse Lick CD Decay
    #Plotting_Functions.compare_lick_cd_decay_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_lick_cd_decay_orthogonal_norm_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

    # Preparatory activity
    Plotting_Functions.compare_preparatory_coupling_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_preparatory_lick_alignment(wt_session_list, wt_output_root, hom_session_list, hom_output_root)

    # Random Vectors To Lick CD
    #Plotting_Functions.compare_random_to_lick_coupling_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)
    #Plotting_Functions.compare_covariance_to_lick_alignment_groups(wt_session_list, wt_output_root, hom_session_list, hom_output_root)