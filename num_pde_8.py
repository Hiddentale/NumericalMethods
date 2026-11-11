import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from numpy.typing import NDArray


def plot_function(
    x: NDArray,
    approximate_list: list[NDArray],
    exact_list: list[NDArray],
    figurename: str,
):
    """Plots the adjusted function

    Arguments:
        x: NDArray containing values for the x axes
        approximate_list: NDArray containing the values for the approximate solution
        exact_list: NDArray containing the values for the exact solution
        figurename: name to save the figure as
    """
    _, ax = plt.subplots()
    for approximate, exact in zip(approximate_list, exact_list):
        (line,) = ax.plot(x, approximate)
        ax.plot(x, exact, "--", color=line.get_color())

    ax.plot([], [], "k-", label="Approximate")
    ax.plot([], [], "k--", label="Exact")

    ax.set_xlabel("x", fontsize=14, fontweight="bold")
    ax.set_ylabel("u", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    plt.savefig(figurename, dpi=150, bbox_inches="tight")


def approximate_solution(
    initial_condition,
    boundary_condition,
    time_steps,
    x_values,
    number_of_interior_points,
    cfl_number,
):
    """Approximates the transport equation through the use of the FTFS scheme."""
    number_of_time_steps = len(time_steps)
    initial_values = initial_condition(x_values)
    boundary_vector = np.zeros(number_of_interior_points)
    boundary_vector[-1] = -cfl_number * boundary_condition
    approximated_values = initial_values
    solution_values = [approximated_values]
    tridiagonal = construct_S(number_of_interior_points, cfl_number)
    for _ in range(number_of_time_steps - 1):
        approximated_values = tridiagonal @ approximated_values + boundary_vector
        solution_values.append(approximated_values)
    return solution_values


def exact_solution(initial_condition, boundary_condition, x, t):
    """The calculated exact solution."""
    return np.where(x + t <= 1, initial_condition(x + t), boundary_condition)


def case_1(x: np.ndarray):
    """The given initial condition for case 1."""
    return np.pow(np.e, -400 * np.pow(x - 0.85, 2))


def case_2(x: np.ndarray):
    """The given initial condition for case 2."""
    return np.where((x > 0.85) & (x <= 1), 1.0, 0.0)


def construct_S(matrix_width, cfl_number):
    """Constructs the tridagonal matrix S given a matrix width."""
    diagonal = np.full(
        shape=matrix_width,
        fill_value=1 + cfl_number,
    )
    upper_diagonal = np.full(shape=matrix_width - 1, fill_value=-cfl_number)
    S = diags([diagonal, upper_diagonal], [0, 1], format="csc")
    return S


if __name__ == "__main__":

    final_time = 1.0
    initial_number_of_spatial_points_J = 100
    space_step = 1 / initial_number_of_spatial_points_J
    cfl_number = -0.8
    time_step = int(final_time / (-cfl_number * space_step))
    x_values = np.linspace(start=0, stop=1, num=initial_number_of_spatial_points_J + 1)
    time_steps = np.linspace(start=0, stop=final_time, num=time_step + 1)

    number_of_interior_points = int((1 / space_step) - 1)

    x_values = space_step * np.arange(1, number_of_interior_points + 1)

    cases = [(case_1, 0.0), (case_2, 1.0)]
    for case, boundary_condition in cases:
        approximated_values = approximate_solution(
            case,
            boundary_condition,
            time_steps,
            x_values,
            number_of_interior_points,
            cfl_number,
        )

        exact_values = []
        for time_step in time_steps:
            exact_values.append(
                exact_solution(case, boundary_condition, x_values, time_step)
            )
        plot_function(
            x_values,
            approximated_values[::25],
            exact_values[::25],
            f"{case.__name__}",
        )
