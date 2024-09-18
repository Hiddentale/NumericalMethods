import numpy as np
import matplotlib.pyplot as plt

def plot_function(x_axis, y_axis, output_name):
    fig, ax = plt.subplots()
    ax.set_xlabel('x', fontsize=14, fontweight='bold')
    ax.set_ylabel('y', fontsize=14, fontweight='bold')

    ax.plot(x_axis, y_axis)
    plt.savefig(output_name, dpi=150, bbox_inches='tight')

def estimate_alpha(x_axis, y_axis):
    logarithmic_condition_numbers = [np.log(element) for element in y_axis]
    coefficients = np.polyfit(x_axis, logarithmic_condition_numbers, 1)
    scaling_factor = np.exp(coefficients[0])
    return scaling_factor

def calculate_condition_numbers(matrix_sizes):
    condition_numbers = []

    for i in range(4, 13):
        diagonal = [2] * np.power(2, i)
        upper_diagonal = [-1] * (np.power(2, i) - 1)
        tridiagonal_matrix_of_size_i = np.diag(diagonal) + np.diag(upper_diagonal, k=1) + np.diag(upper_diagonal, k=-1)
        condition_number = np.linalg.cond(tridiagonal_matrix_of_size_i)
        condition_numbers.append(condition_number)
    return condition_numbers

# Opdracht (a)
x_axis = [number for number in range(4, 13)]
y_axis = calculate_condition_numbers()
plot_function(x_axis, y_axis, "output.png")

# Opdracht (b)
scaling_factor = estimate_alpha(x_axis, y_axis)
print(f"The estimated scaling factor alpha is: {scaling_factor}")
