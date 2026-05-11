import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from matplotlib.colors import LinearSegmentedColormap


def get_nx_colourmap():

    colours = [
        (0.00, "#fff8e8"),  # near white
        (0.18, "#ffe4ad"),  # very pale amber
        (0.36, "#f6bc5c"),  # soft orange
        (0.55, "#ea8a00"),  # saturated orange
        (0.72, "#b85a00"),  # medium burnt orange
        (0.86, "#6b2d00"),  # dark brown-orange
        (1.00, "#000000"),  # black
    ]

    colourmap = LinearSegmentedColormap.from_list(
        "nx_orange_balanced",
        colours,
        N=256
    )

    return colourmap

def get_wt_colourmap():

    colours = [

        # Hold near-white slightly longer
        (0.00, "#f7f9fb"),

        # Very light blue-grey
        (0.13, "#dbe5ec"),

        # Pale desaturated blue
        (0.31, "#b7cede"),

        # Main blue region
        (0.55, "#6ea6cf"),

        # Darkening begins slightly later
        (0.72, "#3473a6"),

        # Rapid compression into dark navy
        (0.88, "#12304a"),

        # Near black-blue
        (1.00, "#081018"),
    ]

    colourmap = LinearSegmentedColormap.from_list(
        "wt_blue_harmonised",
        colours,
        N=256
    )

    return colourmap

def visualise_colourmap(colourmap, title="Colourmap"):

    # Create vertical gradient from 0 → 1
    gradient = np.linspace(0, 1, 512)
    gradient = np.tile(gradient[:, np.newaxis], (1, 50))

    plt.figure(figsize=(2, 8))

    plt.imshow(
        gradient,
        aspect='auto',
        cmap=colourmap,
        origin='lower'
    )

    plt.yticks([0, 128, 256, 384, 511], [0, 0.25, 0.5, 0.75, 1.0])
    plt.xticks([])

    plt.ylabel("Value")
    plt.title(title)

    plt.tight_layout()
    plt.show()




def compare_colourmaps():

    gradient = np.linspace(0, 1, 512)
    gradient = np.tile(gradient[:, np.newaxis], (1, 50))

    fig, axes = plt.subplots(1, 2, figsize=(3, 6))

    axes[0].imshow(gradient, aspect="auto", cmap=get_nx_colourmap(), origin="lower")
    axes[0].set_title("NX")
    axes[0].set_xticks([])

    axes[1].imshow(gradient, aspect="auto", cmap=get_wt_colourmap(), origin="lower")
    axes[1].set_title("WT")
    axes[1].set_xticks([])

    for ax in axes:
        ax.set_yticks([0, 128, 256, 384, 511])
        ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1.0])

    plt.tight_layout()
    plt.show()

