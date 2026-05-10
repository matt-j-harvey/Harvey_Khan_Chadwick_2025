from matplotlib.colors import LinearSegmentedColormap

def get_nx_colourmap():

    colours = [
        (0.00, "#000000"),  # black
        (0.12, "#2a1200"),  # very dark brown
        (0.28, "#5a2400"),  # dark burnt orange
        (0.45, "#9b4300"),  # brown-orange
        (0.62, "#d66a00"),  # strong orange
        (0.78, "#f3a43a"),  # light orange
        (0.92, "#ffd98a"),  # pale yellow-orange
        (1.00, "#fff8e8"),  # near white
    ]

    colourmap = LinearSegmentedColormap.from_list("nx_orange", colours, N=256)

    return colourmap



def get_wt_colourmap():

    colours = [
        (0.00, "#081018"),   # near black-blue
        (0.15, "#12304a"),   # dark navy
        (0.35, "#2d6f9e"),   # mid blue
        (0.55, "#6aaed6"),   # light blue
        (0.75, "#b7d8ee"),   # pale blue
        (1.00, "#ffffff"),   # white
    ]

    colourmap = LinearSegmentedColormap.from_list("wt_blue", colours, N=256)

    return colourmap
