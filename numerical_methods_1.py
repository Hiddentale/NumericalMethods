import numpy as np
import matplotlib.pyplot as plt
# Ik neem aan dat de functie en variable namen grotendeels zichzelf documenten als ze goed
# benaamd zijn, als je om een of andere reden overal comments wilt hebben geef dat aub aan.
# Ik weet dat het gebruikelijk is in academische wetenschappen om grotendeels al je code
# te commenten, maar dat is juist omdat de gemiddelde academische wetenschapper echt heel
# slecht is in programmeren en dus wel in taal moet uitleggen wat hun onleesbare code doet.

def plot_function(x_axis, y_axis, output_name):
    fig, ax = plt.subplots()
    ax.set_title("The relation between the size of a matrix and its condition number")
    ax.set_xlabel('size of matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('condition number', fontsize=14, fontweight='bold')

    ax.plot(x_axis, y_axis)
    plt.savefig(output_name, dpi=150, bbox_inches='tight')

def estimate_quadratic_exponent(x_axis, y_axis):
    logarithmic_condition_numbers = [np.log(element) for element in y_axis]
    coefficients = np.polyfit(x_axis, logarithmic_condition_numbers, 1)
    quadratic_exponent = np.exp(coefficients[0])
    return quadratic_exponent

def calculate_condition_numbers():
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
quadratic_exponent = estimate_quadratic_exponent(x_axis, y_axis)
print(f"The estimated scaling factor alpha is: {quadratic_exponent}")
