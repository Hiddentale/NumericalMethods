import numpy as np
import matplotlib.pyplot as plt

MESH_FOURIER_NUMBER = 1 / 6


def approximate_solution(final_time, time_steps, x_values, number_of_interior_points):
    """Approximates the heat equation through the use of a FCTS algorithm."""
    number_of_time_steps = round(final_time / time_steps[i])
    initial_values = initial_condition(x_values)

    approximated_values = initial_values
    tridiagonal = tridiagonal_S(number_of_interior_points[i])
    for _ in range(number_of_time_steps):
        approximated_values = ftcs_algorithm(tridiagonal, approximated_values)
    return approximated_values


def calculate_error(x_values, approximated_values):
    """Calculates the maximum error between components of two matrices."""
    difference = abs(exact_solution(x_values, 0.2) - approximated_values)
    return np.max(difference)


def convergence_order(space_step_vector, error):
    """Estimates the local order of convergence, using p = log(error1/error2) / log(h1/h2)."""

    order_of_convergence = np.log(error[:-1] / error[1:]) / np.log(
        space_step_vector[:-1] / space_step_vector[1:]
    )
    return order_of_convergence


def exact_solution(x, t):
    """The given exact solution."""
    return np.pow(np.e, (-4 * np.pow(np.pi, 2) * t)) * np.sin(2 * np.pi * x)


def ftcs_algorithm(tridiagonal, approximated_values):
    """The given matrix-based Forward Time Centered Space algorithm in the case that b^n = 0,
    as mentioned in (3.25) of the lecture notes."""
    return approximated_values @ tridiagonal


def initial_condition(x: np.ndarray):
    """The given initial condition."""
    return np.sin(2 * (np.pi) * x)


def plot_error(space_step_vector, error):
    """Plots the given error on a loglog graph."""
    plt.loglog(space_step_vector, error)
    plt.grid()
    plt.show()


def tridiagonal_S(matrix_width):
    """Constructs a tridagonal matrix given a matrix width."""
    diagonal = np.full(shape=matrix_width, fill_value=1 - 2 * MESH_FOURIER_NUMBER)
    lower_diagonal = np.full(shape=matrix_width - 1, fill_value=MESH_FOURIER_NUMBER)
    tridiagonal = (
        np.diag(diagonal, k=0)
        + np.diag(lower_diagonal, k=1)
        + np.diag(lower_diagonal, k=-1)
    )
    return tridiagonal


if __name__ == "__main__":
    final_time = 0.2
    number_of_iterations = 8
    initial_number_of_spatial_points_J = 5
    initial_space_step_h = 1 / initial_number_of_spatial_points_J

    space_steps = [
        initial_space_step_h * (2 ** (-i)) for i in range(0, number_of_iterations)
    ]
    time_steps = [
        MESH_FOURIER_NUMBER * pow(space_step, 2) for space_step in space_steps
    ]
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
        errors.append(calculate_error(x_values, approximated_values))
    print(
        f"convergence_orders: {convergence_order(np.asarray(space_steps), np.asarray(errors))}"
    )
    plot_error(space_steps, errors)
