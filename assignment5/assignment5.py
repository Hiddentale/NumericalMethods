import numpy as np
import math


def fixed_point(
    given_function, initial_guess: float, tolerance: float, max_iterations_allowed: int
):
    """Compute fixed point of a function via successive substitution.

    Iteratively solves x = g(x) using x_{n+1} = g(x_n) until convergence.

    Arguments:
        given_function: callabe,
            The function that we want to solve numerically with fixed point iteration
        initial_guess: float,
            Initial value x_0
        tolerance: float,
            Convergence criterion |x_{n+1} - x_n| < tolerance.
        max_iterations_allowed: int,
            Maximum iterations before raising RuntimeError.

    Returns:
        iterates_of_x: list,
            Complete sequence of iterates [x_0, x_1, ..., x_n]
        final_approximation: float,
            Final approximation to fixed point.
    """
    iterates_of_x = [initial_guess]
    for index in range(0, max_iterations_allowed + 1):
        iterates_of_x.append(given_function(iterates_of_x[index]))
        if np.abs(iterates_of_x[index + 1] - iterates_of_x[index]) <= tolerance:
            final_approximation = iterates_of_x[-1]
            return iterates_of_x, final_approximation
    raise RuntimeError("Couldn't converge in specified amount of iterations")


if __name__ == "__main__":
    given_function = lambda x: (x + 1) * np.exp(-x)
    initial_guess = float(0)
    tolerance = math.pow(10, -9)
    max_iterations_allowed = 1000

    iterates_of_x, final_approximation = fixed_point(
        given_function, initial_guess, tolerance, max_iterations_allowed
    )
    print(f"\n{iterates_of_x=}\n")
    print(f"{final_approximation=}\n")

    # Part (c)
    L_approximation = []
    for index in range(2, len(iterates_of_x)):
        func = np.abs(
            (iterates_of_x[index] - iterates_of_x[index - 1])
            / (iterates_of_x[index - 1] - iterates_of_x[index - 2])
        )
        L_approximation.append(func)
    print(f"{L_approximation=}\n")
