import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import diags
import polars as pl
import math


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


def visualize_matrix(approximated_values, x_values, time_values):
    df = pl.DataFrame(
        approximated_values,
        schema=[f"x={x:.2f}" for x in x_values],
        orient="row",
    ).insert_column(0, pl.Series("t", time_values))

    print(df)


def approximate_transport_equation_solution(
    initial_condition,
    boundary_condition,
    time_values,
    x_values,
    number_of_interior_points,
    cfl_number,
) -> NDArray:
    """Approximates the transport equation through the use of the Lax-Wendroff scheme."""
    approximated_values = np.zeros((len(time_values), len(x_values)))
    approximated_values[0, :] = initial_condition(x_values)
    approximated_values[:, 0] = boundary_condition(time_values)

    tridiagonal = transport_equation_construct_S(number_of_interior_points, cfl_number)
    for n in range(len(time_values) - 1):
        approximated_values[n + 1, 1:-1] = tridiagonal @ approximated_values[n, 1:-1]
        approximated_values[n + 1, -1] = upwind_scheme(
            approximated_values[n], cfl_number
        )
        approximated_values[n + 1, 1] += (
            1 / 2 * cfl_number * (1 + cfl_number)
        ) * approximated_values[n, 0]
        approximated_values[n + 1, -2] += (
            -1 / 2 * cfl_number * (1 - cfl_number)
        ) * approximated_values[n, -1]

    # visualize_matrix(approximated_values, x_values, time_values)
    return approximated_values


def upwind_scheme(approximated_values, cfl_number):
    return approximated_values[-1] - cfl_number * (
        approximated_values[-1] - approximated_values[-2]
    )


def transport_equation_construct_S(matrix_width, cfl_number):
    """Constructs the tridagonal matrix S, for the approximation of the transport equation, given a matrix width."""
    diagonal = np.full(
        shape=matrix_width,
        fill_value=1 - np.pow(cfl_number, 2),
    )
    upper_diagonal = np.full(
        shape=matrix_width - 1, fill_value=-1 / 2 * cfl_number * (1 - cfl_number)
    )
    lower_diagonal = np.full(
        shape=matrix_width - 1, fill_value=1 / 2 * cfl_number * (1 + cfl_number)
    )
    S = diags([diagonal, upper_diagonal, lower_diagonal], [0, 1, -1], format="csc")
    return S


def calculate_error(x_values, approximated_values, exact_solution, final_time):
    """Calculates the maximum error between components of two matrices."""
    difference = abs(exact_solution(x_values, final_time) - approximated_values[-1])
    return np.max(difference)


def convergence_order(space_step_vector, error):
    """Estimates the local order of convergence, using p = log(error1/error2) / log(h1/h2)."""

    order_of_convergence = np.log(error[:-1] / error[1:]) / np.log(
        space_step_vector[:-1] / space_step_vector[1:]
    )
    return order_of_convergence


def plot_error(space_step_vector, error):
    """Plots the given error on a loglog graph."""
    _, ax = plt.subplots()
    ax.loglog(space_step_vector, error)
    ax.set_xlabel("number of spatial points", fontsize=14, fontweight="bold")
    ax.set_ylabel("error", fontsize=14, fontweight="bold")
    plt.grid()
    plt.show()


def evaluate_empirical_convergence(u_0, g0_u, cfl_number, u_exact, final_time, c):
    initial_number_of_spatial_points = [25, 50, 100, 200, 400]
    errors = []
    space_steps = []
    for number_of_spatial_points in initial_number_of_spatial_points:
        space_step = 1 / number_of_spatial_points
        number_of_time_steps = math.ceil(final_time * c / (cfl_number * space_step))
        time_step = final_time / number_of_time_steps
        actual_cfl_number = c * time_step / space_step
        space_steps.append(space_step)
        x_values = np.linspace(start=0, stop=1, num=number_of_spatial_points + 1)
        time_values = np.linspace(
            start=0, stop=final_time, num=number_of_time_steps + 1
        )

        number_of_interior_points = int((1 / space_step) - 1)

        approximated_values = approximate_transport_equation_solution(
            u_0,
            g0_u,
            time_values,
            x_values,
            number_of_interior_points,
            actual_cfl_number,
        )
        errors.append(
            calculate_error(x_values, approximated_values, u_exact, final_time)
        )
    print(
        f"convergence_orders: {convergence_order(np.asarray(space_steps), np.asarray(errors))}"
    )
    plot_error(space_steps, errors)


if __name__ == "__main__":
    c, alpha, kappa, final_time = 1.0, 1.0, 1.0, 1.0
    cfl_number = 0.8

    def u_exact(x, t):
        return np.sin(2 * np.pi * (x - c * t))

    u_0 = lambda x: u_exact(x, 0)
    g0_u = lambda t: u_exact(0, t)

    evaluate_empirical_convergence(u_0, g0_u, cfl_number, u_exact, final_time, c)


# def approximate_heat_equation_solution(
#     initial_condition,
#     boundary_condition,
#     time_step,
#     space_step,
#     x_values,
#     number_of_interior_points,
#     source: NDArray,
# ):
#     """Approximates the diffusion equation through the use of the Crank-Nicolson scheme."""
#     number_of_time_steps = len(time_steps)
#     initial_values = initial_condition(x_values)
#     # boundary_vector = np.zeros(number_of_interior_points)
#     # boundary_vector[-1] = -cfl_number * boundary_condition
#     k = 1  # Change this
#     mesh_fourier_number = k * (time_step / np.pow(space_step, 2))

#     approximated_values = initial_values
#     solution_values = [approximated_values]
#     S = construct_S(number_of_interior_points, mesh_fourier_number, space_step)
#     M = construct_M(number_of_interior_points, mesh_fourier_number)
#     for i in range(number_of_time_steps - 1):
#         approximated_values = crank_nicolson_algorithm(
#             M, S, approximated_values, source[i]
#         )
#         solution_values.append(approximated_values)
#     return np.array(solution_values)

# def construct_M(matrix_width, mesh_fourier_number):
#     """Constructs the tridagonal matrix M, for the approximation of the heat equation, given a matrix width."""
#     diagonal = np.full(shape=matrix_width, fill_value=1 + mesh_fourier_number)
#     lower_diagonal = np.full(
#         shape=matrix_width - 1, fill_value=-0.5 * mesh_fourier_number
#     )
#     M = diags([diagonal, lower_diagonal, lower_diagonal], [0, -1, 1], format="csc")
#     return M


# def construct_S(matrix_width, mesh_fourier_number, space_step):
#     """Constructs the tridagonal matrix S, for the approximation of the heat equation, given a matrix width."""
#     diagonal = np.full(
#         shape=matrix_width,
#         fill_value=1 - mesh_fourier_number,
#     )
#     lower_diagonal = np.full(
#         shape=matrix_width - 1, fill_value=0.5 * mesh_fourier_number
#     )
#     S = diags([diagonal, lower_diagonal, lower_diagonal], [0, -1, 1], format="csc")
#     return S


# def crank_nicolson_algorithm(M, S, approximated_values, source):
#     """The matrix-based Crank Nicolson algorithm."""
#     b = (S @ approximated_values) + source
#     return linalg.spsolve(M, b)


# def approximate_u_time_derivative(
#     time_delta,
#     approximated_values,
#     alpha=1,
# ):
#     """Approximate u_t with the central difference quotient"""
#     return (
#         time_delta * alpha * (np.gradient(approximated_values, time_delta, axis=0) ** 2)
#     )
