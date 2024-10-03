import numpy as np
from time import perf_counter
from scipy import linalg
import matplotlib.pyplot as plt

def cholesky_decomposition(input_matrix):
    """Computes the Cholesky decomposition of a symmetric positive-definite matrix.

    Input:
    A 2D symmetric positive-definite matrix, numpy array format.

    Returns:
    The upper triangular matrix of the Cholesky decomposition.
    """
    length_of_given_matrix = len(input_matrix)
    matrix_shape = (length_of_given_matrix, length_of_given_matrix)
    output = np.zeros(matrix_shape)

    for i in range(length_of_given_matrix):
        for j in range(i):
            sum = 0
            for k in range(j):
                sum += output[i, k] * output[j, k]
            output[i, j] = (input_matrix[i, j] - sum) / output[j, j]

        sum = 0
        for k in range(i):
            sum += output[i, k] ** 2
        output[i, i] = np.sqrt(input_matrix[i, i] - sum)
    
    return np.transpose(output)

input = np.array([

    [4, 12, -16],
    [12, 37, -43],
    [-16, -43, 98]]
)
print(cholesky_decomposition(input))

def calculate_cholesky_running_time(tridiagonal_matrix_of_size_i):
    start_time = perf_counter()
    linalg.cholesky(tridiagonal_matrix_of_size_i)
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time

def calculate_lu_running_time(tridiagonal_matrix_of_size_i):
    start_time = perf_counter()
    linalg.lu(tridiagonal_matrix_of_size_i)
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time

def plot_function(matrix_sizes, cholesky_running_times, lu_running_times, output_name):
    """Produces a loglog given specific data."""
    plt.figure(figsize=(10, 6))
    plt.loglog(matrix_sizes, cholesky_running_times, label='Cholesky', marker='o')
    plt.loglog(matrix_sizes, lu_running_times, label='LU', marker='s')
    plt.title("Running Times of Cholesky and LU Decompositions")
    plt.xlabel('Matrix Size', fontsize=12)
    plt.ylabel('Running Time', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("decomposition_times.png", dpi=150, bbox_inches='tight')

def compare_running_times():
    """Compares the running times between Cholesky and LU decomposition for specific matrix sizes,
        and then produces a loglog plot of the acquired data."""
    cholesky_running_times, lu_running_times = [], []
    matrix_sizes = [2 ** i for i in range(10, 15)]

    for m in matrix_sizes:
        diagonal = np.full(m, 2)
        upper_diagonal = np.full(m - 1, -1)
        tridiagonal_matrix_of_size_i = np.diag(diagonal) + np.diag(upper_diagonal, k=1) + np.diag(upper_diagonal, k=-1)

        cholesky_running_times.append(calculate_cholesky_running_time(tridiagonal_matrix_of_size_i))
        lu_running_times.append(calculate_lu_running_time(tridiagonal_matrix_of_size_i))
    
    print(matrix_sizes)
    print(cholesky_running_times)
    print(lu_running_times)
    plot_function(matrix_sizes, cholesky_running_times, lu_running_times, "running_time_comparison")

compare_running_times()