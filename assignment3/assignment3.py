import numpy as np
from numpy.typing import NDArray


def matrix_is_not_square(matrix: NDArray) -> bool:
    return matrix.shape[0] != matrix.shape[1]


def matrix_is_not_of_type_float(matrix: NDArray) -> bool:
    return matrix.dtype is not float


def matrices_have_mismatched_sizes(matrix_length: int, right_hand_side: int) -> bool:
    return len(right_hand_side) != matrix_length


def LU_decomposition(matrix: NDArray) -> NDArray:
    """
    Performs in-place LU decomposition using Gaussian elimination without pivoting.

    Arguments:
        matrix: NDArray, the square matrix to be decomposed
    """
    if matrix_is_not_square(matrix):
        raise TypeError("Given matrix is not square.")
    if matrix_is_not_of_type_float(matrix):
        matrix = matrix.astype(float)

    dictionary = {}
    # Forward elimination, column by column
    for pivot_index in range(len(matrix)):
        # Eliminate entries below the current pivot
        for matrix_index in range(pivot_index + 1, len(matrix)):
            # Value we multiply the pivot by to eliminate the i-th entry of the row
            multiplier = (
                matrix[matrix_index, pivot_index] / matrix[pivot_index, pivot_index]
            )
            # Eliminate the i-th entry of the row
            matrix[matrix_index] = (
                matrix[matrix_index] - multiplier * matrix[pivot_index]
            )
            # Save current lower triangular part for later usage
            dictionary[matrix_index, pivot_index] = multiplier

    # Add the lower part of the matrix
    for (row, col), multi in dictionary.items():
        matrix[row, col] = multi
    return matrix


def forward_substitute_from_first_to_last_row(
    matrix_length, solution_vector, triangular_matrix, right_hand_side
):
    """Applies the forward substitution algorithm to the given triangular matrix"""
    for i in range(matrix_length):
        solution_vector[i] = right_hand_side[i] - np.dot(
            triangular_matrix[i, :i], solution_vector[:i]
        )


def forward_substitute_from_last_to_first_row(
    matrix_length, solution_vector, triangular_matrix, right_hand_side
):
    """Applies the backward substitution algorithm to the given triangular matrix"""
    for i in range(matrix_length - 1, -1, -1):
        solution_vector[i] = (
            right_hand_side[i]
            - np.dot(triangular_matrix[i, i + 1 :], solution_vector[i + 1 :])
        ) / triangular_matrix[i, i]


def forward_solve(triangular_matrix: NDArray, right_hand_side: NDArray) -> NDArray:
    """Solve a lower triangular system"""

    matrix_length = len(triangular_matrix)

    if matrices_have_mismatched_sizes(matrix_length, right_hand_side):
        raise Exception("Size mismatch between matrix and right-hand side")

    solution_vector = np.zeros(matrix_length)

    forward_substitute_from_first_to_last_row(
        matrix_length, solution_vector, triangular_matrix, right_hand_side
    )

    return solution_vector


def backward_solve(triangular_matrix, right_hand_side):
    """Solve an upper triangular system"""

    matrix_length = len(triangular_matrix)

    if matrices_have_mismatched_sizes(matrix_length, right_hand_side):
        raise Exception("Size mismatch between matrix and right-hand side")

    solution_vector = np.zeros(matrix_length)

    forward_substitute_from_last_to_first_row(
        matrix_length, solution_vector, triangular_matrix, right_hand_side
    )

    return solution_vector


def LU_solve(matrix: NDArray, right_hand_side: NDArray):
    """Solves the given linear system by using LU decomposition.

    Arguments:
        matrix: NDArray, the square matrix to be solved
        right_hand_side: NDArray,
    """
    lu_decomposed_matrix = LU_decomposition(matrix)
    lower_triangular, upper_triangular = np.tril(lu_decomposed_matrix), np.triu(
        lu_decomposed_matrix
    )
    intermediate_vector = forward_solve(lower_triangular, right_hand_side)
    print(intermediate_vector)
    solution = backward_solve(upper_triangular, intermediate_vector)
    return solution, lu_decomposed_matrix


if __name__ == "__main__":

    input = np.array([[6, 18, 3], [2, 12, 1], [4, 15, 3]])
    print(f"input:\n {input}\n")

    right_hand_side = np.array([3, 19, 0])

    solution, lu_decomposed_matrix = LU_solve(input, right_hand_side)
    print(f"solution: \n {solution}\n")
    print(lu_decomposed_matrix)
    print(np.dot(input, solution))
