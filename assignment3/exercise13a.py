import numpy as np


def LU_decomposition(matrix):
    """Returns the LU decomposition of a given square matrix without using pivoting."""

    length_of_given_square_matrix = len(matrix)
    matrix_shape = (length_of_given_square_matrix, length_of_given_square_matrix)

    lower_triangular, upper_triangular = np.zeros(matrix_shape), np.zeros(matrix_shape)


    for i in range(length_of_given_square_matrix):
        lower_triangular[i][i] = 1

        for j in range(i, length_of_given_square_matrix):
            sum_of_previous_calculated_elements = 0
            for k in range(i):
               sum_of_previous_calculated_elements += lower_triangular[i][k] * upper_triangular[k][j]
            upper_triangular[i][j] = matrix[i][j] - sum_of_previous_calculated_elements
    
        for m in range(i + 1, length_of_given_square_matrix):
            sum_of_previous_calculated_elements = 0
            for k in range(i):
                sum_of_previous_calculated_elements += lower_triangular[m][k] * upper_triangular[k][i]
            lower_triangular[m][i] = (matrix[m][i] - sum_of_previous_calculated_elements) / upper_triangular[i][i]
    
    return lower_triangular, upper_triangular

input = np.array([
    [2, -1, 1],
    [4, 1, -1],
    [1, 1, 2]]
    )

lower_triangular, upper_triangular = LU_decomposition(input)

