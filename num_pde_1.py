import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns

# Style settings for the plots
sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def plot_function(
    x: NDArray,
    y: NDArray,
    function_string: str,
    figurename: str,
    y_lim: bool = False,
):
    """Plots the adjusted function

    Arguments:
        x: NDArray containing values for the x ax
        y: NDArray containing the values for the y as
        function_string: string with the function plotted (for display)
        figurename: name to save the figure as
        y_lim: optional parameter to set plot y limit
    """
    _, ax = plt.subplots()
    ax.semilogx(x, y)

    ax.set_xlabel("x", fontsize=14, fontweight="bold")
    ax.set_ylabel("y", fontsize=14, fontweight="bold")

    tick_locations = [1e-9, 1e-7, 1e-6]
    ax.set_xticks(tick_locations)
    ax.set_xticklabels(["$10^{-9}$", "$10^{-7}$", "$10^{-6}$"])

    ax.text(
        0.25,
        0.95,
        function_string,
        transform=ax.transAxes,
        fontsize=14,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )
    if y_lim:
        ax.set_ylim(0.499, 0.501)

    plt.tight_layout()
    plt.savefig(figurename, dpi=150, bbox_inches="tight")


def given_function(x: float) -> float:
    """The function that was given in the exercise"""
    return np.sin(x)


def derivative(x: float) -> float:
    """Derivative of the function that was given in the exercise"""
    return np.cos(x)


def one_sided_approximation(a, b, c, x: float, h: float) -> float:
    """Computes the one sided approximation given in the exercise"""
    return (
        a * given_function(x)
        + b * given_function(x - h)
        + c * given_function(x - 2 * h)
    )


def compute_error(a, b, c, x: float, h: float) -> float:
    """Computes the error of the one sided approximation to the exact value"""
    return np.abs(one_sided_approximation(a, b, c, x, h) - derivative(x))


if __name__ == "__main__":
    h = np.logspace(1, 0.00001)
    x = 1.0
    for nudge in h:
        print(compute_error(1.0, 1.0, 1.0, x, nudge))
    # y = given_function(x)
    # plot_function(x, y, r"$$", "output_plot.png")
