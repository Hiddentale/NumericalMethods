import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import polars as pl


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
    """Approximates the transport equation through the use of the FTFS scheme."""
    approximated_values = np.zeros((len(time_values), len(x_values)))
    approximated_values[0, :] = initial_condition(x_values)
    approximated_values[:, 0] = boundary_condition(time_values)

    visualize_matrix(approximated_values, x_values, time_values)

    tridiagonal = transport_equation_construct_S(number_of_interior_points, cfl_number)
    for n in range(len(time_values) - 1):
        approximated_values[n + 1, 1:-1] = tridiagonal @ approximated_values[n, 1:-1]
        # Some kind of issue here still at boundary
        approximated_values[n + 1, -1] = upwind_scheme(
            approximated_values[n], cfl_number
        )

    visualize_matrix(approximated_values, x_values, time_values)


def upwind_scheme(approximated_values, cfl_number):
    return approximated_values[-1] - cfl_number * (
        approximated_values[-1] + approximated_values[-2]
    )


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


# def exact_solution(initial_condition, boundary_condition, x, t):
#     """The calculated exact solution."""
#     return np.where(x + t <= 1, initial_condition(x - t), boundary_condition)


# def chase_1(x: np.ndarray):
#     """The given initial condition for case 1."""
#     return np.pow(np.e, -400 * np.pow(x - 0.85, 2))


# def chase_2(x: np.ndarray):
#     """The given initial condition for case 2."""
#     return np.where((x > 0.85) & (x <= 1), 1.0, 0.0)


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
    S = np.diags([diagonal, upper_diagonal, lower_diagonal], [0, 1, -1], format="csc")
    return S


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


if __name__ == "__main__":
    c, alpha, kappa, T = 1.0, 1.0, 1.0, 1.0

    def u_exact(x, t):
        return np.sin(2 * np.pi * (x - c * t))

    u_0 = lambda x: u_exact(x, 0)
    g0_u = lambda t: u_exact(0, t)

    final_time = 1.0
    initial_number_of_spatial_points_J = 100
    space_step = 1 / initial_number_of_spatial_points_J
    cfl_number = 0.8
    time_step = int(final_time / (cfl_number * space_step))
    x_values = np.linspace(start=0, stop=1, num=initial_number_of_spatial_points_J + 1)
    print(len(x_values))
    time_steps = np.linspace(start=0, stop=final_time, num=time_step + 1)

    number_of_interior_points = int((1 / space_step) - 1)

    approximated_u_values = approximate_transport_equation_solution(
        u_0,
        g0_u,
        time_steps,
        x_values,
        number_of_interior_points,
        cfl_number,
    )
    # exact_values = []
    # for time_step in time_steps:
    #     exact_values.append(exact_solution(chase_1, 0.0, x_values, time_step))
    # plot_function(
    #     x_values,
    #     approximated_u_values[::25],
    #     exact_values[::25],
    #     f"{chase_1.__name__}",
    # )
    # source = approximate_u_time_derivative(time_step, approximated_u_values)

    # approximated_theta_values = approximate_heat_equation_solution()
    # cases = [(chase_1, 0.0), (chase_2, 1.0)]
    # for case, boundary_condition in cases:
    #     approximated_values = approximate_transport_equation_solution(
    #         case,
    #         boundary_condition,
    #         time_steps,
    #         x_values,
    #         number_of_interior_points,
    #         cfl_number,
    #     )

    #     exact_values = []
    #     for time_step in time_steps:
    #         exact_values.append(
    #             exact_solution(case, boundary_condition, x_values, time_step)
    #         )
    #     plot_function(
    #         x_values,
    #         approximated_values[::25],
    #         exact_values[::25],
    #         f"{case.__name__}",
    #     )
