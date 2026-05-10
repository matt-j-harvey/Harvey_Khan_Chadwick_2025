import numpy as np
import os
import matplotlib.pyplot as plt


def get_pattern_lick_rate(behaviour_matrix, selected_pattern):

  # Load Stimuli Trials
  visual_context_vis_1_trials = np.where((behaviour_matrix[:, 1] == 1) & (behaviour_matrix[:, 23] == selected_pattern))
  visual_context_vis_2_trials = np.where((behaviour_matrix[:, 1] == 2) & (behaviour_matrix[:, 23] == selected_pattern))
  odour_context_vis_1_trials = np.where((behaviour_matrix[:, 6] == 1) & (behaviour_matrix[:, 23] == selected_pattern))
  odour_context_vis_2_trials = np.where((behaviour_matrix[:, 6] == 2) & (behaviour_matrix[:, 23] == selected_pattern))

  visual_context_vis_1_lick_rate = np.mean(behaviour_matrix[visual_context_vis_1_trials, 2])
  visual_context_vis_2_lick_rate = np.mean(behaviour_matrix[visual_context_vis_2_trials, 2])
  odour_context_vis_1_lick_rate = 1 - np.mean(behaviour_matrix[odour_context_vis_1_trials, 7])
  odour_context_vis_2_lick_rate = 1 - np.mean(behaviour_matrix[odour_context_vis_2_trials, 7])

  return [visual_context_vis_1_lick_rate, visual_context_vis_2_lick_rate, odour_context_vis_1_lick_rate, odour_context_vis_2_lick_rate]


def get_baseline_lick_rates(behaviour_matrix):

  # Load Stimuli Trials
  visual_context_vis_1_trials = np.where((behaviour_matrix[:, 1] == 1) & (behaviour_matrix[:, 22] == False))
  visual_context_vis_2_trials = np.where((behaviour_matrix[:, 1] == 2) & (behaviour_matrix[:, 22] == False))
  odour_context_vis_1_trials = np.where( (behaviour_matrix[:, 6] == 1) & (behaviour_matrix[:, 22] == False))
  odour_context_vis_2_trials = np.where( (behaviour_matrix[:, 6] == 2) & (behaviour_matrix[:, 22] == False))

  visual_context_vis_1_lick_rate = np.mean(behaviour_matrix[visual_context_vis_1_trials, 2])
  visual_context_vis_2_lick_rate = np.mean(behaviour_matrix[visual_context_vis_2_trials, 2])

  odour_context_vis_1_lick_rate = 1 - np.mean(behaviour_matrix[odour_context_vis_1_trials, 7])
  odour_context_vis_2_lick_rate = 1 - np.mean(behaviour_matrix[odour_context_vis_2_trials, 7])

  #print("visual_context_vis_1_trials", visual_context_vis_1_trials)
  print("visual_context_vis_1_lick_rate", visual_context_vis_1_lick_rate)

  #print("visual_context_vis_2_trials", visual_context_vis_2_trials)
  print("visual_context_vis_2_lick_rate", visual_context_vis_2_lick_rate)

  #print("odour_context_vis_1_trials", odour_context_vis_1_trials)
  print("odour_context_vis_1_lick_rate", odour_context_vis_1_lick_rate)

  #print("odour_context_vis_2_trials", odour_context_vis_2_trials)
  print("odour_context_vis_2_lick_rate", odour_context_vis_2_lick_rate)

  return [visual_context_vis_1_lick_rate, visual_context_vis_2_lick_rate, odour_context_vis_1_lick_rate, odour_context_vis_2_lick_rate]



def plot_lick_rate_change(baseline_lick_rate, pattern_lick_rate, pattern_save_directory, stim_name):

    plt.title(stim_name)
    plt.plot([0, 1], [baseline_lick_rate, pattern_lick_rate])
    plt.scatter([0, 1], [baseline_lick_rate, pattern_lick_rate])
    plt.ylim([-0.1, 1.1])
    plt.xticks(ticks=[0, 1], labels=['baseline', 'inhibition'])
    plt.xlim([-0.2, 1.2])
    plt.savefig(os.path.join(pattern_save_directory, stim_name + ".png"))
    plt.close()

def create_opto_reports(base_directory):

  # Load Behaviour Matrix
  behaviour_matrix = np.load(os.path.join(base_directory, "Stimuli_Onsets", "Behaviour_Matrix.npy"), allow_pickle=True)

  # Get Pattern Labels
  pattern_labels = behaviour_matrix[:, 23]

  # Get Baseline Lick Rates
  [baseline_visual_context_vis_1_lick_rate,
   baseline_visual_context_vis_2_lick_rate,
   baseline_odour_context_vis_1_lick_rate,
   baseline_odour_context_vis_2_lick_rate] = get_baseline_lick_rates(behaviour_matrix)

  # Get Unique Patterns
  print(pattern_labels)
  pattern_labels_none_removed = pattern_labels[pattern_labels != None]
  unique_patterns = np.unique(pattern_labels_none_removed)
  n_unique_patterns = len(unique_patterns)

  # Get Pattern lick Rate
  for pattern_index in range(n_unique_patterns):

    # reate Pattern Save Directory
    pattern_save_directory = os.path.join(base_directory, "Opto_Stim_Reports", "Stim_" + str(pattern_index).zfill(3))
    if not os.path.exists(pattern_save_directory):
      os.makedirs(pattern_save_directory)

    [pattern_visual_context_vis_1_lick_rate,
     pattern_visual_context_vis_2_lick_rate,
     pattern_odour_context_vis_1_lick_rate,
     pattern_odour_context_vis_2_lick_rate] = get_pattern_lick_rate(behaviour_matrix, pattern_index)

    print("pattern: ", str(pattern_index), "baseline_visual_context_vis_1_lick_rate", baseline_visual_context_vis_1_lick_rate, "pattern_visual_context_vis_1_lick_rate", pattern_visual_context_vis_1_lick_rate)
    print("pattern: ", str(pattern_index), "baseline_visual_context_vis_2_lick_rate", baseline_visual_context_vis_2_lick_rate, "pattern_visual_context_vis_2_lick_rate", pattern_visual_context_vis_2_lick_rate)
    print("pattern: ", str(pattern_index), "baseline_odour_context_vis_1_lick_rate", baseline_odour_context_vis_1_lick_rate, "pattern_odour_context_vis_1_lick_rate", pattern_odour_context_vis_1_lick_rate)
    print("pattern: ", str(pattern_index), "baseline_odour_context_vis_2_lick_rate", baseline_odour_context_vis_2_lick_rate, "pattern_odour_context_vis_2_lick_rate", pattern_odour_context_vis_2_lick_rate)

    plot_lick_rate_change(baseline_visual_context_vis_1_lick_rate, pattern_visual_context_vis_1_lick_rate, pattern_save_directory, "visual_context_vis_1")
    plot_lick_rate_change(baseline_visual_context_vis_2_lick_rate, pattern_visual_context_vis_2_lick_rate, pattern_save_directory, "visual_context_vis_2")
    plot_lick_rate_change(baseline_odour_context_vis_1_lick_rate, pattern_odour_context_vis_1_lick_rate, pattern_save_directory, "odour_context_vis_1")
    plot_lick_rate_change(baseline_odour_context_vis_2_lick_rate, pattern_odour_context_vis_2_lick_rate, pattern_save_directory, "odour_context_vis_2")


"""

base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_12_Switching_M2_1_Filter"
base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_20_Switching_MM_1_Filter"
base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_06_21_Switch_MM_2F"
base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_22_Switching_MM_2_Filter"
base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC1.3A/2023_06_21_Switch_MM_2F"
base_directory = "/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3E/2023_06_22_Switch_V1_1F_03"
base_directory = r"/media/matthew/External_Harddrive_2/Cortex_Wide_Opto/KPGC3.3A/2023_06_02_Switching_V1_1_Filter/ROI_0_Opto_Stim_Trial_Averages.npy"


create_opto_reports(base_directory)
"""