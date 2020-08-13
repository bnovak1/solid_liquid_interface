import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator #MultipleLocator
import subprocess

# Don't output plots to screen
plt.ioff()

# Figure size in inches
mpl.rcParams['figure.figsize'] = 3.54, 2.655

# Font size in plots
mpl.rcParams['font.size'] = 10

# legend font size
mpl.rcParams['legend.fontsize'] = 8.5

# Font size for added text
text_font_size = 9

# font
p = subprocess.run('which latex', shell=True, stdout=subprocess.PIPE)
if len(p.stdout.decode()) == 0:
    pass
else:
    mpl.rcParams['text.usetex'] = True

mpl.rcParams['font.family'] = 'serif'

# Line width in plots
mpl.rcParams['lines.linewidth'] = 1.0

# Marker size in plots
mpl.rcParams['lines.markersize'] = 4.0

# Errorbar width (points)
errbarW = 4

# Only one point in legends
mpl.rcParams['legend.numpoints'] = 1

# No box around legends, best location, change spacings
mpl.rcParams['legend.frameon'] = False
mpl.rcParams['legend.loc'] = 'best'
mpl.rcParams['legend.fontsize'] = 7.5
mpl.rcParams['legend.columnspacing'] = 0.5
mpl.rcParams['legend.labelspacing'] = 0.25
mpl.rcParams['legend.handletextpad'] = 0.2
mpl.rcParams['legend.handlelength'] = 1
mpl.rcParams['legend.handleheight'] = 0.5

# Space between tick labels and axes labels
axeslabelpad = 7

mpl_version = (mpl.__version__).split('.')[0]

def axis_setup(axis_type):

    if axis_type == 'x':

        plt.margins(x=0.02)

        # xmajticks = plt.xticks()[0]
        # xmintick_len = (xmajticks[1] - xmajticks[0])/4.0
        # ml = MultipleLocator(xmintick_len); plt.axes().xaxis.set_minor_locator(ml)
        plt.axes().xaxis.set_minor_locator(AutoMinorLocator())

    elif axis_type == 'y':

        plt.margins(y=0.02)

        # ymajticks = plt.yticks()[0]
        # ymintick_len = (ymajticks[1] - ymajticks[0])/4.0
        # ml = MultipleLocator(ymintick_len); plt.axes().yaxis.set_minor_locator(ml)
        plt.axes().yaxis.set_minor_locator(AutoMinorLocator())

def save_figure(file_name, res):
    plt.savefig(file_name, bbox_inches='tight', pad_inches=0.02, dpi=res)
