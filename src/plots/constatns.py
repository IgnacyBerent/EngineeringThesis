import matplotlib.pyplot as plt

DPI = 600
SQUARE_FIG_SIZE = (8, 8)
RECTANGLE_FIG_SIZE = (15, 6)

DEFAULT_SIGNAL_COLOR = 'blue'
DEFAULT_MARKER_COLOR = 'red'

TITLE_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 14
AXIS_TICKS_FONT_SIZE = 12
ANNOTATION_FONT_SZIE = 12
LEGEND_FONT_SIZE = 12

plt.rcParams.update(
    {
        'figure.dpi': DPI,
        'axes.titlesize': TITLE_FONT_SIZE,
        'ytick.labelsize': AXIS_TICKS_FONT_SIZE,
        'xtick.labelsize': AXIS_TICKS_FONT_SIZE,
        'axes.labelsize': AXIS_LABEL_FONT_SIZE,
        'legend.fontsize': LEGEND_FONT_SIZE,
        'font.size': ANNOTATION_FONT_SZIE,
    }
)
