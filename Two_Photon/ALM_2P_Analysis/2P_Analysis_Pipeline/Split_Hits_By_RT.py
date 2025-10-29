import os

import matplotlib.pyplot as plt
import numpy as np

import ALM_Analysis_Utils



def get_rt_time(onset, downsampled_lick_trace, period, lick_threshold):

    n_timepoints = len(downsampled_lick_trace)
    licked = False
    count = 0
    while licked == False:

        if downsampled_lick_trace[onset + count] > lick_threshold:
            return count * period

        else:
            count += 1
            if count+onset == n_timepoints:
                return None





def split_hits_by_rt(data_root, session, output_root, lick_threshold=600):

    # Load Frame Rate
    frame_rate = np.load(os.path.join(data_root, session, "Frame_Rate.npy"))
    period = float(1)/frame_rate
    period = np.multiply(period, 1000)

    # Load Vis 1 Hits
    vis_1_hits = np.load(os.path.join(data_root, session,"Stimuli_Onsets", "visual_context_stable_vis_1_onsets.npy"))

    # Load AI Data
    ai_data = np.load(os.path.join(output_root, session, "Behaviour", "Downsampled_AI_Matrix_Framewise.npy"))
    print("ai_data", np.shape(ai_data))

    # Load Stimuli Dict
    stimuli_dict = ALM_Analysis_Utils.load_rig_1_channel_dict()

    # Get Lick Trace
    lick_trace = ai_data[stimuli_dict["Lick"]]

    hit_rt_list = []
    rt_dist = []
    for onset in vis_1_hits:
        trial_rt = get_rt_time(onset, lick_trace, period, lick_threshold)
        hit_rt_list.append([onset, trial_rt])
        rt_dist.append(trial_rt)

    hit_rt_list = np.array(hit_rt_list)
    print("hit_rt_list", np.shape(hit_rt_list))

    print("rt_dist", rt_dist)

    np.save(os.path.join(output_root, session, "Behaviour", "Hits_By_RT.npy"), hit_rt_list)




