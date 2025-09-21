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
        y: NDArray containing the valuies for the y as
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


def given_function(x: NDArray) -> NDArray:
    "The function that was given in the exercise"
    return (1 - np.cos(x)) / np.power(x, 2)


def adjusted_function(x: NDArray) -> NDArray:
    "Adjusted function that sidesteps the numerical cancellation that happens for the original function"
    sin_term = np.sin(x / 2)
    numerator = 2 * np.power(sin_term, 2)
    denominator = np.power(x, 2)
    result = numerator / denominator
    return result


if __name__ == "__main__":

    x = np.logspace(-9, -6, 50000)
    y = given_function(x)
    plot_function(x, y, r"$f(x) = \frac{1 - \cos(x)}{x^2}$", "output_plot.png")

    y = adjusted_function(x)
    plot_function(
        x,
        y,
        r"$f(x) = \frac{2\sin^2(\frac{x}{2})}{x^2}$",
        "output_plot_adjusted.png",
        True,
    )
