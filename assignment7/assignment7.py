import numpy as np
import matplotlib.pyplot as plt


def pol_interp(x_coordinates, function_values, x_eval):
    interpolation_values = []
    for index in range(len(x_coordinates)):
        interpolation_values.append(
            Lagrange_pol(index, x_coordinates, x_eval) * function_values[index]
        )
    result = np.sum(interpolation_values, axis=0).tolist()
    return result


def Lagrange_pol(index: int, x_coordinates: list, x_eval):
    lagrange_polynomial = construct_lagrange_polynomial(index, x_coordinates)
    solutions = lagrange_polynomial(x_eval)
    return solutions


def construct_lagrange_polynomial(index: int, x_coordinates: list):
    coefficients = [1]

    parts = []
    for x_point in x_coordinates:
        parts.append([x_point, 1])

    for part in parts:
        coefficients = multiply_polynomials(coefficients, part)

    B = calculate_denominator(index, x_coordinates)

    return lambda x: sum(
        (coefficient * x**i) / B for i, coefficient in enumerate(coefficients)
    )


def multiply_polynomials(polynomial1: list, polynomial2: list):
    print(polynomial1)
    print(polynomial2)
    result = [0] * (len(polynomial1) + len(polynomial2) - 1)
    for i, coefficient_1 in enumerate(polynomial1):
        for j, coefficient_2 in enumerate(polynomial2):
            print(coefficient_1)
            print(coefficient_2)
            result[i + j] += coefficient_1 * coefficient_2
    return result


def calculate_denominator(index, x_pts: list):
    product = 0
    x_i = x_pts[index]
    for i in range(len(x_pts)):
        if i == index:
            pass
        product *= x_i - x_pts[i]
    return product

def given_function():
    return lambda x: np.sin(2* np.pi * x)


def plot_function(x_axis, y_axis, figure_name: str):
    """
    Plots the given function

    Arguments:
        x: NDArray containing values for the x axes
        y: NDArray containing the valuies for the y axes
        figure_name: name to save the figure as
    """
    _, ax = plt.subplots()
    ax.set_title("")
    ax.set_xlabel("iterations", fontsize=14, fontweight="bold")
    ax.set_ylabel("", fontsize=14, fontweight="bold")
    x_values = []
    for i in range(x_axis):
        x_values.append(i + 1)

    ax.plot(x_values, y_axis)
    plt.savefig(figure_name, dpi=150, bbox_inches="tight")


if __name__ == "__main__":
    x_coordinates = np.linspace(-1.0, 1.0, 17)
    function_values = given_function()(x_coordinates)
    x_eval = np.linspace(-1.0, 1.0, 1001)
    result = pol_interp(x_coordinates, function_values, x_eval)
    print(result)
