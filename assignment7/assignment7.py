import numpy as np
import matplotlib.pyplot as plt


def pol_interp(x_coordinates, function_values, x_eval):
    """
    Performs Lagrange polynomial interpolation.

    Computes the interpolating polynomial P(x) that passes through all given
    data points (x_coordinates, function_values) and evaluates it at x_eval.

    Args:
        x_coordinates: x-coordinates of the interpolation points.
        function_values: function values at the interpolation points.
        x_eval: x-coordinates where the polynomial should be evaluated.

    Returns:
        list: Values of the interpolating polynomial evaluated at x_eval.
    """
    interpolation_values = []
    for index in range(len(x_coordinates)):
        interpolation_values.append(
            Lagrange_pol(index, x_coordinates, x_eval) * function_values[index]
        )
    result = np.sum(interpolation_values, axis=0).tolist()
    return result


def Lagrange_pol(index: int, x_coordinates: list, x_eval):
    """
    Evaluates the i-th Lagrange basis polynomial at given points.

    Args:
        index: Index i of the basis polynomial to evaluate.
        x_coordinates: List of x-coordinates of all interpolation points.
        x_eval: Array-like of points where the basis polynomial should be evaluated.

    Returns:
        Values of L_i(x) evaluated at x_eval.
    """
    lagrange_polynomial = construct_lagrange_polynomial(index, x_coordinates)
    solutions = lagrange_polynomial(x_eval)
    return solutions


def construct_lagrange_polynomial(index: int, x_coordinates: list):
    """
    Constructs the i-th Lagrange basis polynomial as a callable function.

    Builds the polynomial L_i(x) by computing
    the numerator coefficients and denominator separately.

    Args:
        index: Index i of the basis polynomial to construct.
        x_coordinates: List of x-coordinates of all interpolation points.

    Returns:
        callable: Function that evaluates L_i(x) for any input x.
    """
    coefficients = [1]

    # Construct a list of arrays that contain the coefficients of every (x - x_i) in the lagrange polynomial
    parts = []
    for i, x_point in enumerate(x_coordinates):
        if i != index:
            parts.append([-x_point, 1])

    for part in parts:
        coefficients = multiply_polynomials(coefficients, part)

    B = calculate_denominator(index, x_coordinates)

    # Return a lambda function of our Lagrange polynomial,
    # so we can evaluate it at any given x value without having to recompute it
    return lambda x: sum(
        (coefficient * x**i) / B for i, coefficient in enumerate(coefficients)
    )


def multiply_polynomials(polynomial1: list, polynomial2: list):
    """
    Multiplies two polynomials represented as coefficient lists.

    Polynomials are represented with coefficients in ascending order of powers,
    i.e., [a0, a1, a2, ...] represents a0 + a1*x + a2*x^2 + ...

    Args:
        polynomial1: Coefficient list of the first polynomial.
        polynomial2: Coefficient list of the second polynomial.

    Returns:
        list: Coefficient list of the product polynomial.
    """
    result = [0] * (len(polynomial1) + len(polynomial2) - 1)
    for i, coefficient_1 in enumerate(polynomial1):
        for j, coefficient_2 in enumerate(polynomial2):
            result[i + j] += coefficient_1 * coefficient_2
    return result


def calculate_denominator(index, x_coordinates: list):
    """
    Calculates the denominator of the i-th Lagrange basis polynomial,
    skips if j = i.

    Args:
        index: Index i of the basis polynomial.
        x_pts: List of all x-coordinates of the interpolation points.

    Returns:
        float: The denominator value for the i-th basis polynomial.
    """
    product = 1
    x_i = x_coordinates[index]
    for i in range(len(x_coordinates)):
        if i == index:
            continue
        product *= x_i - x_coordinates[i]
    return product


def given_function(k):
    """
    Returns a function for interpolation.
    """
    return lambda x: np.sin(2 * np.pi * x * k)


def compare_and_plot_equidistant_versus_chebyshev_nodes(number_of_nodes, k, plot_name):
    """
    Compares and plots interpolation of a given function with both equidistant nodes and chebyshev nodes
    at the given number of nodes and value k.
    """
    x_coordinates = np.linspace(-1.0, 1.0, number_of_nodes)
    chebyshev_nodes = np.cos(
        np.pi * (2 * np.arange(number_of_nodes) + 1) / (2 * number_of_nodes)
    )

    # initialize function with specified k value, then give it x_coordinates as input
    function_values = given_function(k)(x_coordinates)
    function_values_chebyshev = given_function(k)(chebyshev_nodes)

    x_eval = np.linspace(-1.0, 1.0, 1001)
    result = pol_interp(x_coordinates, function_values, x_eval)
    result_chebyshev = pol_interp(chebyshev_nodes, function_values_chebyshev, x_eval)

    # Plot functions and nodes
    plt.figure()
    plt.plot(
        x_eval, given_function(k)(x_eval), "b-", label="Original function", linewidth=2
    )
    plt.plot(x_eval, result, "r--", label="Equispaced nodes")
    plt.plot(x_eval, result_chebyshev, "g--", label="Chebyshev nodes")
    plt.plot(x_coordinates, function_values, "ro")
    plt.plot(chebyshev_nodes, function_values_chebyshev, "go")
    plt.xlabel("x", fontsize=14, fontweight="bold")
    plt.ylabel("y", fontsize=14, fontweight="bold")
    plt.legend()
    plt.savefig(plot_name, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    compare_and_plot_equidistant_versus_chebyshev_nodes(
        number_of_nodes=17, k=1, plot_name="interpolation_plot_1"
    )
    compare_and_plot_equidistant_versus_chebyshev_nodes(
        number_of_nodes=33, k=4, plot_name="interpolation_plot_2"
    )
