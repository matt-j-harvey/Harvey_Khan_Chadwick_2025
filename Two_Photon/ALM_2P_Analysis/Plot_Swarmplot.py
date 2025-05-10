import numpy as np
import matplotlib.pyplot as plt
from scipy import  stats


def add_confidence_interval_bars(axis, x, values, color='#000000', horizontal_line_width=0.25):

    # Get Mean
    mean = np.mean(values)

    # Get CI
    confidence_interval = stats.t.interval(confidence=0.95, df=len(values) - 1, loc=mean, scale=stats.sem(values))

    left = x - horizontal_line_width / 2
    top = confidence_interval[1]
    right = x + horizontal_line_width / 2
    bottom = confidence_interval[0]

    axis.plot([x, x], [top, bottom], color=color, zorder=0)
    axis.plot([left, right], [top, top], color=color, zorder=0)
    axis.plot([left, right], [bottom, bottom], color=color, zorder=0)
    axis.plot(x, mean, 'o', color='000000', zorder=0)

    return mean, confidence_interval


def get_signficance_mark(p_value):

    if p_value > 0.05:
        sig_mark = "n.s."

    elif p_value < 0.05 and p_value >= 0.005:
        sig_mark = "*"

    elif p_value < 0.005 and p_value >= 0.0005:
        sig_mark = "**"

    elif p_value < 0.0005:
        sig_mark = "***"

    return sig_mark



def add_signfificance_brackets(axis, group_1_x, group_2_x, ylim, p_value):


    # Add Bar
    height  = ylim[1]
    distance = (ylim[1] - ylim[0]) * 0.25
    bar_height = height - distance
    arrow_properties = {'connectionstyle': 'bar', 'arrowstyle': '-', 'shrinkA': 48, 'shrinkB': 48, 'linewidth': 2}
    axis.annotate('', xy=(group_1_x, bar_height), xytext=(group_2_x, bar_height), arrowprops=arrow_properties)

    #Add Text
    text = get_signficance_mark(p_value)
    text_x = group_1_x + (group_2_x - group_1_x)/2
    text_y = height #bar_height + bar_height*0.01
    axis.annotate(text, xy=(text_x,text_y), zorder=10, fontsize='xx-large')






def swarmplot(group_1_values, group_2_values, group_1_cmap, group_2_cmap, save_directory, plot_name, y_lim, x_labels=["",""], y_label="", plot_significance=False, plot_confidence_interval=False):

    # Get Colours
    group_1_color = group_1_cmap(0.5)
    group_2_color = group_2_cmap(0.5)

    """
    # Test Signficance
    t_stat, p_value = stats.ttest_ind(control_values, neurexin_values)
    print("t stat", t_stat, "p value", p_value)
    """

    # Create Figure
    figure_1 = plt.figure()
    axis_1 = figure_1.add_subplot(1,1,1)

    # Get X Values
    n_group_1 = len(group_1_values)
    n_group_2 = len(group_2_values)

    spread = 0.05
    group_1_center = 1
    group_2_center = 2

    group_1_x_values = np.linspace(start=np.subtract(group_1_center, spread), stop=np.add(group_1_center, spread), num=n_group_1)
    group_2_x_values = np.linspace(start=np.subtract(group_2_center, spread), stop=np.add(group_2_center, spread), num=n_group_2)

    # Scatter points
    axis_1.scatter(group_1_x_values, group_1_values, color=group_1_color,  edgecolors='k', alpha=0.8)
    axis_1.scatter(group_2_x_values, group_2_values, color=group_2_color,  edgecolors='k', alpha=0.8)

    axis_1.set_xlim(0.5, 2.5)
    axis_1.set_ylim(y_lim)
    axis_1.set_xticks([1,2])
    axis_1.set_xticklabels([x_labels[0], x_labels[1]])

    axis_1.set_ylabel(y_label)
    axis_1.spines[['right', 'top']].set_visible(False)
    #plt.savefig(os.path.join(save_directory, plot_name + ".svg"))
    #plt.close()

    """
    if plot_significance == True:
        add_signfificance_brackets(axis_1, control_center, neurexin_center, y_lim, p_value)
    """

    if plot_confidence_interval == True:
        add_confidence_interval_bars(axis_1, 1, group_1_values)
        add_confidence_interval_bars(axis_1, 2, group_2_values)


    plt.show()