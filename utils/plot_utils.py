# plotting_utils.py

import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import rc

cb_palette = ["#1a80bb", "#ea801c"]


def setup_plotting(scale=2.5):
    sns.set_style("white")
    rc("font", **{"family": "serif", "serif": ["Times New Roman"]})
    rc("text", usetex=True)
    rc("xtick", labelsize=17 * scale)
    rc("ytick", labelsize=17 * scale)
    rc("lines", linewidth=2.5)
    mpl.rcParams["lines.markersize"] = 8 * scale
    mpl.rcParams["axes.titlesize"] = 20 * scale
    mpl.rcParams["axes.labelsize"] = 20 * scale
    mpl.rcParams["legend.fontsize"] = 16 * scale
    mpl.rcParams["font.size"] = 24 * scale
    plt.rcParams["errorbar.capsize"] = 3 * scale
    plt.rcParams["lines.linewidth"] = 2 * scale

    plt.clf()
    plt.figure(constrained_layout=True)

    return plt


def get_asterisks(p_val):
    return (
        "****"
        if p_val < 0.0001
        else (
            "***"
            if p_val < 0.001
            else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else ""))
        )
    )
