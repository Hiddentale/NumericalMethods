import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags

MESH_FOURIER_NUMBER = 1 / 6


def approximate_solution(final_time, time_steps, x_values, number_of_interior_points):
    """Approximates the diffusion equation through the use of the Crank-Nicolson scheme."""
    return None


def calculate_error(x_values, approximated_values):
    """Calculates the maximum error between components of two matrices."""
    # difference = abs(exact_solution(x_values, 0.2) - approximated_values)
    return None


def convergence_order(space_step_vector, error):
    """Estimates the local order of convergence, using p = log(error1/error2) / log(h1/h2)."""

    order_of_convergence = np.log(error[:-1] / error[1:]) / np.log(
        space_step_vector[:-1] / space_step_vector[1:]
    )
    return order_of_convergence


def exact_solution(x, t):
    """The calculated exact solution."""
    return None


def crank_nicolson_algorithm(tridiagonal, approximated_values):
    """The given matrix-based Crank Nicolson algorithm, as mentioned in (4.9) of the lecture notes."""
    return None


def initial_condition(x: np.ndarray):
    """The given initial condition."""
    return np.sin((np.pi) * x)


def plot_error(space_step_vector, error):
    """Plots the given error on a loglog graph."""
    plt.loglog(space_step_vector, error)
    plt.grid()
    plt.show()


def tridiagonal(matrix_width):
    """Constructs a tridagonal matrix given a matrix width."""
    diagonal = np.full(shape=matrix_width, fill_value=1 - 2 * MESH_FOURIER_NUMBER)
    lower_diagonal = np.full(shape=matrix_width - 1, fill_value=MESH_FOURIER_NUMBER)
    tridiagonal = diags([diagonal, lower_diagonal, lower_diagonal], [0, -1, 1])
    return tridiagonal


if __name__ == "__main__":
    None
