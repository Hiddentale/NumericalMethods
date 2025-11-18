import numpy as np
from matplotlib import pyplot as plt


def composite_trapezoidalquad(
    given_function: callable, a: float, b: float, number_of_subintervals: int
):
    """Calculate the antiderivative of a given integral over a specified domain by approximation.

    Specifically, uses the Composite trapezoidal method to construct a Quadrature rule of the given integral
    and hence approximate the solution.

    Arguments:
        given_function: callable, the integral to approximate
        a: float, the start of the interval for the integrand
        b: float, the end of the interval for the integrand
        number_of_intervals: int, how many intervals to use

    Returns:
        The approximation of the integrand
    """
    subinterval_width = (b - a) / number_of_subintervals

    array_of_x_values = np.linspace(a, b, number_of_subintervals + 1)
    array_of_fx_values = given_function(array_of_x_values)

    integrand = subinterval_width * (
        0.5 * array_of_fx_values[0]
        + np.sum(array_of_fx_values[1:-1])
        + 0.5 * array_of_fx_values[-1]
    )
    return integrand


def given_function():
    return lambda x: x * np.sin(2 * np.pi * x)


def plot_error(number_of_subintervals, errors):
    plt.loglog(number_of_subintervals, errors)
    plt.xlabel("Step size h")
    plt.ylabel("Error")
    plt.title(
        "Composite trapezodial method error analysis"
    )
    plt.savefig("loglog_plot.png")


if __name__ == "__main__":
    function = given_function()
    a, b = 0, 1
    number_of_subintervals = [2**i for i in range(1, 9)]

    true_value = -1 / (2 * np.pi)
    errors = []
    for n in number_of_subintervals:
        result = composite_trapezoidalquad(function, a, b, n)
        errors.append(abs(true_value - result))

    plot_error(errors, [1 / x for x in number_of_subintervals])
