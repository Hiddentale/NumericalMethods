import numpy as np


def matrices_have_mismatched_sizes(matrix_length, right_hand_side):
    return len(right_hand_side) != matrix_length

def forward_substitute_from_first_to_last_row(matrix_length, solution_vector, triangular_matrix, right_hand_side):
    for i in range(matrix_length):
            solution_vector[i] = (right_hand_side[i] - np.dot(triangular_matrix[i, :i], solution_vector[:i]))

def forward_substitute_from_last_to_first_row(matrix_length, solution_vector, triangular_matrix, right_hand_side):
    for i in range(matrix_length -1, -1, -1):
        solution_vector[i] = (right_hand_side[i] - np.dot(triangular_matrix[i, i+1:], solution_vector[i + 1:])) / triangular_matrix[i, i]

def forward_solve(triangular_matrix, right_hand_side):
    """Solve a lower triangular system"""

    matrix_length = len(triangular_matrix)

    if matrices_have_mismatched_sizes(matrix_length, right_hand_side):
        raise Exception("Size mismatch between matrix and right-hand side")
    
    solution_vector = np.zeros(matrix_length)
    
    forward_substitute_from_first_to_last_row(matrix_length, solution_vector, triangular_matrix, right_hand_side)
        
    return solution_vector

def backward_solve(triangular_matrix, right_hand_side):
    """Solve an upper triangular system"""

    matrix_length = len(triangular_matrix)

    if matrices_have_mismatched_sizes(matrix_length, right_hand_side):
        raise Exception("Size mismatch between matrix and right-hand side")

    solution_vector = np.zeros(matrix_length)

    forward_substitute_from_last_to_first_row(matrix_length, solution_vector, triangular_matrix, right_hand_side)
        
    return solution_vector

def LU_decomposition(matrix):
    """Returns the LU decomposition of a given square matrix without using pivoting."""

    length_of_given_square_matrix = len(matrix)
    matrix_shape = (length_of_given_square_matrix, length_of_given_square_matrix)

    lower_triangular, upper_triangular = np.zeros(matrix_shape), np.zeros(matrix_shape)


    for i in range(length_of_given_square_matrix):
        # All diagonal entries in the lower triangular matrix are 1's
        lower_triangular[i, i] = 1

        # Constructing the upper triangular matrix
        for j in range(i, length_of_given_square_matrix):
            sum_of_previous_calculated_elements = 0
            for k in range(i):
               sum_of_previous_calculated_elements += lower_triangular[i, k] * upper_triangular[k, j]
            upper_triangular[i, j] = matrix[i, j] - sum_of_previous_calculated_elements

        # Constructing the lower triangular matrix
        for m in range(i + 1, length_of_given_square_matrix):
            sum_of_previous_calculated_elements = 0
            for k in range(i):
                sum_of_previous_calculated_elements += lower_triangular[m, k] * upper_triangular[k, i]
            lower_triangular[m, i] = (matrix[m, i] - sum_of_previous_calculated_elements) / upper_triangular[i, i]
    
    return lower_triangular, upper_triangular

def LU_solve(matrix, right_hand_side):
    """Solves the given linear system by using LU decomposition."""
    lower_triangular, upper_triangular = LU_decomposition(matrix)
    intermediate_vector = forward_solve(lower_triangular, right_hand_side)
    solution = backward_solve(upper_triangular, intermediate_vector)
    return solution

matrix = np.array([
    [6, 18, 3],
    [2, 12, 1],
    [4, 15, 3]]
    )

right_hand_side = np.array([3, 19, 0])

solution = LU_solve(matrix, right_hand_side)
print(solution)
print(np.dot(matrix, solution))