import numpy as np
from numpy.typing import NDArray
from time import perf_counter
from scipy import linalg
import matplotlib.pyplot as plt
from typing import Any


def cholesky_decomposition(input_matrix: NDArray) -> NDArray:
    """Computes the Cholesky decomposition of a symmetric positive-definite matrix.

    Arguments:
        input_matrix: NDArray, A 2D symmetric positive-definite matrix;

    Returns:
        NDArray: The upper triangular matrix of the Cholesky decomposition.
    """
    length_of_given_matrix = len(input_matrix)
    matrix_shape = (length_of_given_matrix, length_of_given_matrix)
    output = np.zeros(matrix_shape)

    # The cholesky algorithm from the lecture notes
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


input = np.array([[4, 12, -16], [12, 37, -43], [-16, -43, 98]])
print(f"Lower triangular matrix L_*:\n {cholesky_decomposition(input)}")


def calculate_cholesky_running_time(tridiagonal_matrix_of_size_i: NDArray) -> float:
    """Calculates the running time for doing the cholesky decomposition on a tridiagonal matrix of size i

    Arguments:
        tridiagonal_matrix_of_size_i: NDArray

    Returns:
        float: The time elapsed for doing cholesky decomposition on the given matrix
    """
    start_time = perf_counter()
    linalg.cholesky(tridiagonal_matrix_of_size_i)
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time


def calculate_lu_running_time(tridiagonal_matrix_of_size_i):
    """Calculates the running time for doing the LU decomposition on a tridiagonal matrix of size i

    Arguments:
        tridiagonal_matrix_of_size_i: NDArray

    Returns:
        float: The time elapsed for doing LU decomposition on the given matrix
    """
    start_time = perf_counter()
    linalg.lu(tridiagonal_matrix_of_size_i)
    end_time = perf_counter()
    elapsed_time = end_time - start_time
    return elapsed_time


def plot_function(
    matrix_sizes: list[Any],
    cholesky_running_times: list[Any],
    lu_running_times: list[Any],
):
    """Produces a loglog plot given specific data.

    Arguments:
        matrix_sizes: list of matrix sizes that we calculated running times at
        cholesky_running_times: list of running times of cholesky decomposition for given matrix sizes
        lu_running_times: list of running times of lu decomposition for given matrix sizes
    """
    plt.figure(figsize=(10, 6))
    plt.loglog(matrix_sizes, cholesky_running_times, label="Cholesky", marker="o")
    plt.loglog(matrix_sizes, lu_running_times, label="LU", marker="s")
    plt.title("Running Times of Cholesky and LU Decompositions")
    plt.xlabel("Matrix Size", fontsize=12)
    plt.ylabel("Running Time", fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.savefig("decomposition_times.png", dpi=150, bbox_inches="tight")


def compare_running_times():
    """Compares the running times between Cholesky and LU decomposition for specific matrix sizes,
    and then produces a loglog plot of the acquired data."""

    cholesky_running_times, lu_running_times = [], []
    matrix_sizes = [2**i for i in range(10, 15)]

    for m in matrix_sizes:
        # Create the tridiagonal matrix of size m
        diagonal = np.full(m, 2)
        upper_diagonal = np.full(m - 1, -1)
        tridiagonal_matrix_of_size_i = (
            np.diag(diagonal)
            + np.diag(upper_diagonal, k=1)
            + np.diag(upper_diagonal, k=-1)
        )
        # Check running times on matrix for both algorithms
        cholesky_running_times.append(
            calculate_cholesky_running_time(tridiagonal_matrix_of_size_i)
        )
        lu_running_times.append(calculate_lu_running_time(tridiagonal_matrix_of_size_i))

    print(f"{matrix_sizes=}")
    print(f"{cholesky_running_times=}")
    print(f"{lu_running_times=}")
    plot_function(
        matrix_sizes,
        cholesky_running_times,
        lu_running_times,
    )


if __name__ == "__main__":
    compare_running_times()
