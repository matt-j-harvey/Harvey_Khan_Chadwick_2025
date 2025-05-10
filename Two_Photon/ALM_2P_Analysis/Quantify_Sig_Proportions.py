import numpy as np
import os
import matplotlib.pyplot as plt
from scipy import stats
import Plot_Swarmplot





def quantify_signficance(session_list):

    total_negative_cells = 0
    total_positive_cells = 0
    total_cells = 0
    total_unmodulated_cells = 0

    positive_fraction_list = []
    negative_fraction_list = []
    unmodulated_fraction_list = []
    for session in session_list:

        negative_celL_indicies = np.load(os.path.join(session, "Cell Significance Testing", "negative_cell_indexes.npy"))
        positive_cell_indicies = np.load(os.path.join(session, "Cell Significance Testing", "positive_cell_indexes.npy"))
        all_cell_vector = np.load(os.path.join(session, "Cell Significance Testing", "significance_vector.npy"))

        session_negative_cells = len(negative_celL_indicies)
        session_positive_cells = len(positive_cell_indicies)
        session_n_cells = len(all_cell_vector)
        session_unmodulated_cells = session_n_cells - (session_positive_cells + session_negative_cells)

        print("session_negative_cells", session_negative_cells)
        print("session_positive_cells", session_positive_cells)
        print("session_n_cells", session_n_cells)

        # Add Cells to Pool
        total_cells += session_n_cells
        total_positive_cells += session_positive_cells
        total_negative_cells += session_negative_cells
        total_unmodulated_cells += session_unmodulated_cells

        # Calculate Session Fractions
        session_positive_fraction = (float(session_positive_cells) / session_n_cells) * 100
        session_negative_fraction = (float(session_negative_cells) / session_n_cells) * 100
        session_unmodulated_fraction = (float(session_unmodulated_cells) / session_n_cells) * 100
        positive_fraction_list.append(session_positive_fraction)
        negative_fraction_list.append(session_negative_fraction)
        unmodulated_fraction_list.append(session_unmodulated_fraction)


    # Pi Chart of Pooled Cells
    print("total_cells", total_cells)
    print("total_positive_cells", total_positive_cells, float(total_positive_cells)/total_cells*100)
    print("total_negative_cells", total_negative_cells, float(total_negative_cells)/total_cells*100)
    print("total_unmodulated_cells", total_unmodulated_cells, float(total_unmodulated_cells)/total_cells*100)

    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.pie([total_unmodulated_cells,
                   total_positive_cells,
                   total_negative_cells],
                   labels=["Unmodulated Cells",
                           "Positive Cells",
                           "Negative Cells"],
               colors=['Grey','crimson','dodgerblue'],
               autopct='%1.1f%%')
    axis_1.set_title("Pooled Cell Fractions")
    plt.show()


    # Mean Across Mice
    mean_positive_fraction = np.mean(positive_fraction_list)
    mean_negative_fraction = np.mean(negative_fraction_list)
    mean_unmodulated_fraction = np.mean(unmodulated_fraction_list)

    print("Across mouse positive mean", mean_positive_fraction)
    print("Across mouse negative mean", mean_negative_fraction)
    print("Across mouse unmodulated mean", mean_unmodulated_fraction)
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)
    axis_1.pie([mean_unmodulated_fraction,
                   mean_positive_fraction,
                   mean_negative_fraction],
                   labels=["Unmodulated",
                           "Positive",
                           "Negative"],
               colors=['Grey','crimson','dodgerblue'],
               autopct='%1.1f%%')
    axis_1.set_title("Mean fractions across mice")
    plt.show()

    # Plot Swarmplots
    negative_cmap = plt.get_cmap("Blues")
    positive_cmap = plt.get_cmap("Reds")



    Plot_Swarmplot.swarmplot(negative_fraction_list,
                             positive_fraction_list,
                             negative_cmap,
                             positive_cmap,
                             save_directory=None,
                             plot_name="Across Mouse Fractions",
                             y_lim=[0,100],
                             x_labels=["Negatively Modulated","Positively Modulated"],
                             y_label="Fraction Modulated",
                             plot_significance=False,
                             plot_confidence_interval=True)




sig_window_start = 8
sig_window_stop = 15
data_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Results\Lick_Modulation"

#quantify_grand_raster(data_directory, sig_window_start, sig_window_stop)



session_list = [
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2a\2024_08_05_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\65.2b\2024_07_31_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\67.3b\2024_08_09_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\67.3C\2024_08_20_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\69.2a\2024_08_12_Switching",
    r"C:\Users\matth\OneDrive - The Francis Crick Institute\Documents\Neurexin_Paper\ALM 2P\Data\Controls\72.3C\2024_09_10_Switching",
]



save_directory = r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Hom_Results"

quantify_signficance(session_list)



session_list = [
   #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK64.1B\2024_09_09_Switching",
    #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1A\2024_09_09_Switching",
    #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK70.1B\2024_09_12_Switching",
    #r"C:\Users\matth\Documents\PhD Docs\ALM 2P\Data\Homs\NXAK72.1E\2024_08_23_Switching",
]
