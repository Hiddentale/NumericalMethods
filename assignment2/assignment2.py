import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
import time

# Style settings for the plots
sns.set_style("darkgrid")
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]


def plot_function(x_axis: NDArray, y_axis: NDArray, figure_name: str):
    """
    Plots the given function

    Arguments:
        x: NDArray containing values for the x axes
        y: NDArray containing the valuies for the y axes
        figure_name: name to save the figure as
    """
    _, ax = plt.subplots()
    matrix_sizes = [f"{2}^{x}" for x in x_axis]
    ax.set_title("The relation between the size of a matrix and its condition number")
    ax.set_xlabel("size of matrix", fontsize=14, fontweight="bold")
    ax.set_ylabel("condition number", fontsize=14, fontweight="bold")

    ax.plot(matrix_sizes, y_axis)
    plt.savefig(figure_name, dpi=150, bbox_inches="tight")


def estimate_quadratic_exponent(x_axis: NDArray, y_axis: NDArray) -> list:
    """
    Estimates the quadratic exponent, alpha, by converting the known equation
    to a first-order polynomial through the log function. Then fits a line to this equation
    to approximate the coefficient of this polynomial.

    Arguments:
        x: NDArray containing values for the x axes
        y: NDArray containing the valuies for the y axes
    """
    logarithmic_condition_numbers = [np.log(element) for element in y_axis]
    x_values = [np.log(2**x) for x in x_axis]

    coefficients = np.polyfit(x_values, logarithmic_condition_numbers, 1)
    quadratic_exponent = np.exp(coefficients[0])
    return quadratic_exponent


def calculate_condition_numbers() -> list:
    """
    For every matrix of size 2^4 to 2^13 constructs a tridiagonal matrix
    where the diagonal is all 2's and then other values of the tridiagonal are -1's.
    Then calculates the condition number for the constructed matrix.
    """
    condition_numbers = []

    for i in range(4, 13):
        print(f"Starting on matrix of size 2^{i}")
        start = time.time()

        # Construct tridiagonal matrix for given matrix size
        diagonal = [2] * np.power(2, i)
        upper_diagonal = [-1] * (np.power(2, i) - 1)
        tridiagonal_matrix_of_size_i = (
            np.diag(diagonal)
            + np.diag(upper_diagonal, k=1)
            + np.diag(upper_diagonal, k=-1)
        )
        # Calculate condition number for given matrix
        condition_number = np.linalg.cond(tridiagonal_matrix_of_size_i)
        condition_numbers.append(condition_number)

        # Keeping track of how long a single iteration takes to make sure that the code is not stuck/actually runs.
        print(f"Current iteration took {time.time() - start:0.6f} seconds.\n")
    return condition_numbers


# If the file that you are running, is the current file, run the code, otherwise do not run the code.
# If you are working with multiply python files and this file is imported into that file, if __name__ == "__main__"
# is not in this file, the code automatically gets run at compilation, creating weird bugs.
# Good habit to always use, even if working with just one file.
if __name__ == "__main__":
    # Opdracht (a)
    x_axis = [number for number in range(4, 13)]
    y_axis = calculate_condition_numbers()
    plot_function(x_axis, y_axis, "output.png")

    # Opdracht (b)
    quadratic_exponent = estimate_quadratic_exponent(x_axis, y_axis)
    print(f"The estimated scaling factor alpha is: {quadratic_exponent}")
