import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, linalg

MESH_FOURIER_NUMBER = 2


def approximate_solution(final_time, time_steps, x_values, number_of_interior_points):
    """Approximates the diffusion equation through the use of the Crank-Nicolson scheme."""
    number_of_time_steps = round(final_time / time_steps[i])
    initial_values = initial_condition(x_values)

    approximated_values = initial_values
    S = construct_S(number_of_interior_points[i])
    M = construct_M(number_of_interior_points[i])
    for _ in range(number_of_time_steps):
        approximated_values = crank_nicolson_algorithm(M, S, approximated_values)
    return approximated_values


def calculate_error(x_values, space_step, approximated_values):
    """Calculates the maximum error in the l_2 norm between components of two matrices."""
    difference = exact_solution(x_values, 0.2) - approximated_values
    return np.sqrt(space_step * np.sum(np.pow(difference, 2)))


def convergence_order(space_step_vector, error):
    """Estimates the local order of convergence, using p = log(error1/error2) / log(h1/h2)."""

    order_of_convergence = np.log(error[:-1] / error[1:]) / np.log(
        space_step_vector[:-1] / space_step_vector[1:]
    )
    return order_of_convergence


def exact_solution(x, t):
    """The calculated exact solution."""
    return np.pow(np.e, (-(np.pow(np.pi, 2) + 1) * t)) * np.sin(np.pi * x)


def crank_nicolson_algorithm(M, S, approximated_values):
    """The given matrix-based Crank Nicolson algorithm, as mentioned in (4.9) of the lecture notes."""
    b = S @ approximated_values
    return linalg.spsolve(M, b)


def initial_condition(x: np.ndarray):
    """The given initial condition."""
    return np.sin((np.pi) * x)


def plot_error(space_step_vector, error):
    """Plots the given error on a loglog graph."""
    plt.loglog(space_step_vector, error)
    plt.grid()
    plt.show()


def construct_M(matrix_width):
    """Constructs the tridagonal matrix M given a matrix width."""
    diagonal = np.full(shape=matrix_width, fill_value=1 + MESH_FOURIER_NUMBER)
    lower_diagonal = np.full(
        shape=matrix_width - 1, fill_value=-0.5 * MESH_FOURIER_NUMBER
    )
    M = diags([diagonal, lower_diagonal, lower_diagonal], [0, -1, 1])
    return M


def construct_S(matrix_width):
    """Constructs the tridagonal matrix S given a matrix width."""
    diagonal = np.full(shape=matrix_width, fill_value=1 - MESH_FOURIER_NUMBER)
    lower_diagonal = np.full(
        shape=matrix_width - 1, fill_value=0.5 * MESH_FOURIER_NUMBER
    )
    S = diags([diagonal, lower_diagonal, lower_diagonal], [0, -1, 1])
    return S


if __name__ == "__main__":
    final_time = 0.2
    number_of_iterations = 8
    initial_number_of_spatial_points_J = 5
    initial_space_step_h = 1 / initial_number_of_spatial_points_J

    space_steps = [
        initial_space_step_h * (2 ** (-i)) for i in range(0, number_of_iterations)
    ]
    time_steps = space_steps
    number_of_interior_points = [
        int((1 / space_step) - 1) for space_step in space_steps
    ]

    errors = []
    for i in range(number_of_iterations):
        x_values = space_steps[i] * np.arange(1, number_of_interior_points[i] + 1)
        approximated_values = approximate_solution(
            final_time,
            time_steps,
            x_values,
            number_of_interior_points,
        )
        errors.append(calculate_error(x_values, space_steps[i], approximated_values))
    print(
        f"convergence_orders: {convergence_order(np.asarray(space_steps), np.asarray(errors))}"
    )
    plot_error(space_steps, errors)
