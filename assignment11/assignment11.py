import numpy as np
import matplotlib.pyplot as plt

def explicit_Euler(
    given_function: callable,
    initial_time,
    initial_state,
    time_step_size,
    number_of_time_steps: int,
):
    """
    Given an input function(ODE) and several parameters, uses explicit Euler to numerically solve an ODE.
    
    Arguments:
        given_function: callable
        initial_time: float
        initial_state: float
        time_step_size: float
        number_of_time_steps: int

    Returns:
        An array of the solutions of the ODE at every different step in the method.
    """
    solutions = [initial_state]
    previous_time = initial_time
    for i in range(number_of_time_steps):
        previous_solution = solutions[i]
        solution = previous_solution + time_step_size * given_function(
            previous_time, previous_solution
        )
        solutions.append(solution)
        previous_time += time_step_size
    return np.array(solutions)


def modified_euler(
    given_function: callable,
    initial_time,
    initial_state,
    time_step_size,
    number_of_time_steps: int,
):
    """
    Given an input function(ODE) and several parameters, uses the explicit midpoint rule to numerically solve an ODE.
    
    Arguments:
        given_function: callable
        initial_time: float
        initial_state: float
        time_step_size: float
        number_of_time_steps: int

    Returns:
        An array of the solutions of the ODE at every different step in the method.
    """
    solutions = [initial_state]
    previous_time = initial_time
    for i in range(number_of_time_steps):
        previous_solution = solutions[i]
        mid_step_solution = previous_solution + time_step_size * 0.5 * given_function(
            previous_time, previous_solution
        )
        mid_step_time = previous_time + 0.5 * time_step_size
        solution = previous_solution + time_step_size * given_function(
            mid_step_time, mid_step_solution
        )
        solutions.append(solution)
        previous_time += time_step_size
    return np.array(solutions)


def lotka_volterra(t, state, lambda_1=1.0, lambda_2=0.5, x_c=15.0, y_c=10.0):
    """
    Lotka-Volterra predator-prey model.
    state[0] = x (prey), state[1] = y (predator)
    """
    x, y = state[0], state[1]
    dx_dt = lambda_1 * x * (1 - y / y_c)
    dy_dt = lambda_2 * y * (x / x_c - 1)
    return np.array([dx_dt, dy_dt])


if __name__ == "__main__":
    initial_time, total_time = 0.0, 20.0
    initial_state = np.array([10.0, 5.0])

    # Put all plots into one single figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    
    for i in range(4):
        time_step_size = (total_time - initial_time) * 0.0125 * (2 ** (-i))
        number_of_time_steps = int((total_time - initial_time) / time_step_size)
        
        euler_solution = explicit_Euler(lotka_volterra, initial_time, initial_state, 
                                        time_step_size, number_of_time_steps)
        modified_euler_solution = modified_euler(lotka_volterra, initial_time, initial_state, 
                                                 time_step_size, number_of_time_steps)
        
        # Plot specific functions
        ax = axes[i]
        ax.plot(euler_solution[:, 0], euler_solution[:, 1], 
                'b-', linewidth=1.5, label='Explicit Euler', marker='o', 
                markevery=max(1, number_of_time_steps//20), markersize=4)
        ax.plot(modified_euler_solution[:, 0], modified_euler_solution[:, 1], 
                'r-', linewidth=1.5, label='Modified Euler', marker='s', 
                markevery=max(1, number_of_time_steps//20), markersize=4)

        ax.set_xlabel('x (prey)', fontsize=11)
        ax.set_ylabel('y (predator)', fontsize=11)
        ax.set_title(f'τ = {time_step_size:.6f} (N = {number_of_time_steps} steps)', fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig('lotka_volterra_phase_plane.png', dpi=150)
    plt.show()

    # It seems that modified Euler approximates the periodic solution the best.
