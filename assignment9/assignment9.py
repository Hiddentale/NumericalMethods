import numpy as np


def composite_trapezoidalquad(
    given_function: callable, a: int, b: int, number_of_subintervals: int
):
    """Calculate the antiderivative of a given integral over a specified domain by approximation.

    Specifically, uses the Composite trapezoidal method to construct a Quadrature rule of the given integral
    and hence approximate the solution.

    Arguments:
        given_function: callable, the integral to approximate
        a: int, the start of the interval for the integrand
        b: int, the end of the interval for the integrand
        number_of_intervals: int, how many intervals to use

    Returns:
        The approximation of the integrand
    """
    subinterval_width = (b - a) // number_of_subintervals
    array_of_x_values = np.linspace(a, b, number_of_subintervals + 1)
    array_of_fx_values = given_function(array_of_x_values)
    integrand = (subinterval_width / 2) * array_of_fx_values[0] + (
        subinterval_width / 2
    ) * array_of_fx_values[-1]
    for fx_value in range(array_of_fx_values) - 2:
        integrand += subinterval_width * fx_value

    return integrand

def given_function():
    return lambda x: x * np.sin(2 * np.pi * x)

if __name__ == "__main__":
    function = given_function()
    a, b = 0, 1
    number_of_subintervals = [2**i for i in range(1,9)]
    result = []
    for n in number_of_subintervals:
        result.append(composite_trapezoidalquad(function, a, b, n))
    print(result)