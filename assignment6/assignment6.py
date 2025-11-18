import numpy as np
import matplotlib.pyplot as plt


def Newton(
    given_function,
    given_jacobian,
    initial_guess,
    tolerance: float,
    max_iterations: int,
):
    """Find root of F(x) = 0 using Newton's method.

    Arguments:
        given_function: callabe,
            Vector-valued function F: R^m -> R^m we want to find root of
        given_jabocian: callable,
            Jacobian matrix of F
        initial_guess: float,
            Initial value x_0
        tolerance: float,
            Convergence criterion |x_{n+1} - x_n| < tolerance.
        max_iterations: int,
            Maximum iterations before raising RuntimeError.

    Returns:
        final_approximation: float,
            Final approximation to fixed point.
        total_number_of_iterations: int,
        error_array: list,
            list of of absolute errors of all iterates
    """
    iterates = [initial_guess]
    total_number_of_iterations = 0
    error_array = []
    for i in range(max_iterations):
        total_number_of_iterations = i + 1
        iterates.append(
            iterates[i] - np.linalg.solve(given_jacobian(iterates[i]), given_function(iterates[i]))
        )
        error = np.linalg.norm(iterates[i + 1] - iterates[i])
        error_array.append(error)
        if error < tolerance:
            final_approximation = iterates[-1]
            return final_approximation, total_number_of_iterations, error_array
    raise RuntimeError("Couldn't converge in specified amount of iterations")


def plot_function(x_axis, y_axis, figure_name: str):
    """
    Plots the given function

    Arguments:
        x: NDArray containing values for the x axes
        y: NDArray containing the valuies for the y axes
        figure_name: name to save the figure as
    """
    _, ax = plt.subplots()
    ax.set_title("The relation between the number of iterations of Newton's method and the error")
    ax.set_xlabel("iterations", fontsize=14, fontweight="bold")
    ax.set_ylabel("error", fontsize=14, fontweight="bold")
    x_values = []
    for i in range(x_axis):
        x_values.append(i + 1)

    ax.plot(x_values, y_axis)
    plt.savefig(figure_name, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    given_function = lambda x: np.array(
        [
            -x[0] * np.exp(-(x[0] ** 2 + x[1] ** 2) / 2),
            -x[1] * np.exp(-(x[0] ** 2 + x[1] ** 2) / 2),
        ]
    )

    given_jacobian = lambda x: np.array(
        [
            [
                (-1 + x[0] ** 2) * np.exp(-(x[0] ** 2 + x[1] ** 2) / 2),
                x[0] * x[1] * np.exp(-(x[0] ** 2 + x[1] ** 2) / 2),
            ],
            [
                x[0] * x[1] * np.exp(-(x[0] ** 2 + x[1] ** 2) / 2),
                (-1 + x[1] ** 2) * np.exp(-(x[0] ** 2 + x[1] ** 2) / 2),
            ],
        ]
    )
    # (c)
    initial_guess = np.array([0.25, 0.25])
    result, iterations, errors = Newton(
        given_function, given_jacobian, initial_guess, 1e-8, 1000
    )
    plot_function(iterations, errors, "errors_vs_iterations")

    # (d)
    given_function = lambda x: np.array([-x[0], -x[1]])
    given_jacobian = lambda x: np.array([[-1, 0], [0, -1]])
    result, iterations, errors = Newton(
        given_function, given_jacobian, initial_guess, 1e-8, 1000
    )
    plot_function(iterations, errors, "errors_vs_iterations_logged")
