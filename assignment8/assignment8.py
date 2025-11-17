import numpy as np
import matplotlib.pyplot as plt


def plot_spline(given_function, x):
    """Given a function and a grid of x values, plots the function and its linear interpolant and returns
    an estimate for the error of the linear interpolant.

    Arguments:
        given_function: callable, the function we interpolate linearily.
        x: NDArray, grid values.

    Returns:
        max_error: int, an estimate of the error of the linear interpolant.
    """
    max_error = 0

    plt.figure(figsize=(10, 6))

    x_fine_resolution = np.linspace(x[0], x[-1], 1000)
    plt.plot(
        x_fine_resolution,
        given_function(x_fine_resolution),
        "b-",
        label="Original function",
    )

    for i in range(len(x) - 1):
        # Create fine grid for current interval
        x_i = np.linspace(x[i], x[i + 1], 20)

        linear_interpolant_y_i = given_function(x[i]) + (
            given_function(x[i + 1]) - given_function(x[i])
        ) * (x_i - x[i]) / (x[i + 1] - x[i])

        plt.plot(x_i, linear_interpolant_y_i, "r-")

        actual_function_value_f_i = given_function(x_i)

        # Update maximum error
        interpolant_error = np.max(
            np.abs(actual_function_value_f_i - linear_interpolant_y_i)
        )
        max_error = max(max_error, interpolant_error)

    plt.plot(x, given_function(x), "ro", label="Interpolation points")

    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Function and its Piecewise Linear Interpolant")
    # plt.show()

    return max_error


def given_function(x):
    return 1 / (1 + 25 * x**2)


def test_spline_plotter():
    """Test the plot_spline function with the given_function.

    Returns:
        error: int, an estimate of the error of the linear interpolant.
    """
    x = np.array([-1 + 0.5 * i for i in range(5)])
    error = plot_spline(given_function, x)
    plt.savefig("interpolation_plot.png")
    plt.close()
    return error


def convergence_analysis():
    """Computes the maximum norm error of the piecewise linear interpolant using equispaced points. Then returns a
    loglog plot of the error versus the number of subintervals.

    Returns:
        estimated_convergence_rate: float.
    """
    k_values = np.arange(1, 11)
    m_values = 2**k_values
    errors = []

    for m in m_values:
        x = np.array([-1 + 2 * i / m for i in range(m + 1)])
        error = plot_spline(given_function, x)
        plt.close()
        errors.append(error)

    plt.figure(figsize=(10, 6))
    plt.loglog(m_values, errors, "bo-", label="Computed errors")

    approximate_slope = np.polyfit(np.log(m_values), np.log(errors), 1)
    plt.loglog(
        m_values,
        np.exp(approximate_slope[1]) * m_values ** approximate_slope[0],
        "r--",
        label=f"Fitted line (slope ≈ {approximate_slope[0]:.2f})",
    )

    plt.grid(True)
    plt.xlabel("Number of subintervals")
    plt.ylabel("Maximum error")
    plt.title(
        "LogLog plot of the error against number of subintervals for piecewise linear interpolation"
    )
    plt.legend()
    plt.savefig("loglog_plot.png")
    plt.close()

    estimated_convergence_rate = approximate_slope[0]

    return estimated_convergence_rate



if __name__ == "__main__":
    print("Part (b) error:", test_spline_plotter())

    rate = convergence_analysis()
    print(f"Estimated convergence rate: {rate:.2f}")
