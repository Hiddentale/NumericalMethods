import numpy as np
import math

def fixed_point(
    given_function, initial_guess: float, tolerance: float, max_iterations_allowed: int
):
    """Finds the value of x at the root for a given function given an initial guess
    
    Arguments:
        given_function: The function that we want to solve numerically with fixed point iteration
        initial_guess: float
        tolerance: float, how accurate we need to be (change this)
        max_iterations_allowed: int

    Returns:
        iterates_of_x: list, all points up till now
        final_approximation: float, the approximated root value 
    """
    iterates_of_x = [initial_guess]
    for index in range(0, max_iterations_allowed + 1):
        iterates_of_x.append(given_function(iterates_of_x[index]))
        if np.abs(iterates_of_x[index + 1] - iterates_of_x[index]) <= tolerance:
            final_approximation = iterates_of_x[-1]
            return iterates_of_x, final_approximation 
    raise RuntimeError("Couldn't converge in specified amount of iterations")

if __name__ == "__main__":
    given_function = lambda x: ( x + 1 ) * np.exp(-x)
    initial_guess = float(0)
    tolerance = math.pow(10,-9)
    max_iterations_allowed = 1000

    iterates_of_x, final_approximation = fixed_point(
        given_function, 
        initial_guess, 
        tolerance, 
        max_iterations_allowed
        )
    print(f"\n{iterates_of_x=}\n")
    print(f"{final_approximation=}\n")

    L_approximation = []
    for index in range(2, len(iterates_of_x)):
        func = np.abs((iterates_of_x[index] - iterates_of_x[index - 1])/(iterates_of_x[index - 1] - iterates_of_x[index - 2]))
        L_approximation.append(func)
    print(f"{L_approximation=}\n")