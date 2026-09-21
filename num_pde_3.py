import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x: np.ndarray):
    return np.sin(2 * (np.pi) * x)


def exact_solution(x, t):
    return np.pow(np.e, (-4 * np.pow(np.pi, 2) * t)) * np.sin(2 * np.pi * x)


def ftcs_algorithm(x, t):
    return None


def plot_error(space_step_vector, error_inf):
    plt.loglog(space_step_vector, error_inf)
    plt.grid()
    plt.show()


def tridiagonal_S(matrix_width, mesh_fourier_number):
    diagonal = np.full(shape=matrix_width, fill_value=1 - 2 * mesh_fourier_number)
    lower_diagonal = np.full(shape=matrix_width - 1, fill_value=mesh_fourier_number)
    tridiagonal = (
        np.diag(diagonal, k=0)
        + np.diag(lower_diagonal, k=1)
        + np.diag(lower_diagonal, k=-1)
    )
    return tridiagonal


def initial_values(matrix_width):
    pass


if __name__ == "__main__":
    mesh_fourier_number = 1 / 6
    final_time = 0.2
    number_of_iterations = 8
    initial_number_of_spatial_points_J = 5
    initial_space_step_h = 1 / initial_number_of_spatial_points_J
    space_steps = [
        initial_space_step_h * (2 ** (-i)) for i in range(0, number_of_iterations)
    ]
    print(f"space_steps: {space_steps}")
    time_steps = [
        mesh_fourier_number * pow(space_step, 2) for space_step in space_steps
    ]
    print(f"time_steps: {time_steps}")
    number_of_interior_points = [
        int((1 / space_step) - 1) for space_step in space_steps
    ]

    x_values = space_steps[0] * np.arange(1, number_of_interior_points[0] + 1)
    print(f"x_values: {x_values}")
    number_of_time_steps = round(final_time / time_steps[0])
    initial_values = initial_condition(x_values)

    approximated_values = initial_values
    tridiagonal = tridiagonal_S(number_of_interior_points[0], mesh_fourier_number)
    for _ in range(number_of_time_steps):
        approximated_values = tridiagonal @ approximated_values
    print(f"approx_values: {approximated_values}")
    difference = exact_solution(x_values, 0.2) - approximated_values
    print(f"difference: {difference}")
    print(np.max(difference))
