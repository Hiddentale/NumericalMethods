import numpy as np
import matplotlib.pyplot as plt


def initial_condition(x: np.ndarray):
    return np.sin(2 * (np.pi) * x)


def exact_solution(x, t):
    return np.pow(np.e, (-4 * np.pow(np.pi, 2) * t)) * np.sin(2 * np.pi * x)


def ftcs_algorithm(x, t):
    return None


def plot_error(space_step_vector, error_inf):
    plt.loglog(space_step_vector, error_inf)
    plt.grid()
    plt.show()


def tridiagonal_S(matrix_width, mesh_fourier_number):
    diagonal = np.full(shape=matrix_width, fill_value=1 - 2 * mesh_fourier_number)
    lower_diagonal = np.full(shape=matrix_width - 1, fill_value=mesh_fourier_number)
    tridiagonal = (
        np.diag(diagonal, k=0)
        + np.diag(lower_diagonal, k=1)
        + np.diag(lower_diagonal, k=-1)
    )
    return tridiagonal


def initial_values(matrix_width):
    pass


def pde_approximation():
    for space_step in space_step_vector:
        # Initializations in h
        initial_number_of_spatial_points = int(1 / space_step)
        dT = 1 / 6 * (space_step**2)
        r = 1 / 6

        # vector of initial conditions
        x = np.linspace(start=0, stop=1, num=initial_number_of_spatial_points + 1)
        vec = np.vectorize()

        solution = ftcs_algorithm(x)

        error_iter_inf = []
        error_inf.append(error_iter_inf)


# Computation of rate of convergence
rate_inf = None
print(f"The empirical inf rate is: {rate_inf}")

if __name__ == "__main__":
    mesh_fourier_number = 1 / 6
    x_exact = np.linspace(start=0, stop=1, num=201)  # fix this
    final_time = 0.2
    number_of_iterations = 8
    initial_number_of_spatial_points = 5
    initial_space_step = 1 / initial_number_of_spatial_points
    space_steps = [
        initial_space_step * (2 ** (-i)) for i in range(0, number_of_iterations)
    ]
    print(space_steps)
    print(tridiagonal_S(5, mesh_fourier_number))
    x_values = np.linspace(0, 1, 6)
    print(x_values)
    initial_values = initial_condition(x_values)
    print(initial_values)
    U_1 = (
        tridiagonal_S(initial_number_of_spatial_points + 1, mesh_fourier_number)
        * initial_values
    )
    print(U_1)
    # error_inf = []
    # plot_error(space_step_vector, error_inf)

    pass
