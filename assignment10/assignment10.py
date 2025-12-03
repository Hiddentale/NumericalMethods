import numpy as np
import math


def compute_gauss_legrende_quadrature(number_of_nodes: int):
    """Given a number of nodes, calculates the nodes and weights of the Gauss-Legrende quadrature rule and returns
    those nodes and weights.

    Arguments:
        number_of_nodes: int

    Returns:
        nodes: list
        weights: list
    """
    gamma = [calculate_gamma(n) for n in range(2, number_of_nodes + 1)]
    delta = [0] * number_of_nodes
    tridiagonal_matrix = np.diag(gamma, -1) + np.diag(delta, 0) + np.diag(gamma, 1)

    eigenvalues, eigenvectors = np.linalg.eig(tridiagonal_matrix)

    # Sort arrays to be consistent with np.polynomial.legendre.leggauss ouput
    sort_indices = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_indices]
    eigenvectors = eigenvectors[:, sort_indices]

    weights = calculate_weights(eigenvectors, number_of_nodes)
    return eigenvalues, weights


def calculate_gamma(n):
    return (n - 1) / math.sqrt(4 * (n - 1) ** 2 - 1)


def calculate_weights(eigenvectors, number_of_nodes):
    """Calculates the weights of the given eigenvectors

    Arguments:
        eigenvectors: NDArray
        number_of_nodes: int

    Returns:
        weights: NDArray
    """
    sum_of_eigenvectors_first_components = sum(
        [eigenvectors[0][i] ** 2 for i in range(number_of_nodes)]
    )
    weights = []
    for i in range(number_of_nodes):
        weights.append(
            (2 * eigenvectors[0][i] ** 2) / sum_of_eigenvectors_first_components
        )
    return np.array(weights)


def compute_integral_numerically(number_of_nodes, nodes_and_weights):
    """Computes the integral numerically.

    Observe that our original nodes are built on the domain [-1, 1]
    and hence we transform the domain to [pi/2, pi] by use of the linear equation x = pi/4 * t + 3pi /4.

    Arguments:
        number_of_nodes: int
        nodes_and_weights: NDArray

    Returns:
        weights: NDArray
    """
    result = 0
    for i in range(number_of_nodes):
        result += (
            (math.pi / 4)
            * math.sin((math.pi / 4) * nodes_and_weights[0][i] + (3 * math.pi / 4))
            * nodes_and_weights[1][i]
        )
    return result


if __name__ == "__main__":
    NUMBER_OF_NODES = 4
    nodes_and_weights = compute_gauss_legrende_quadrature(NUMBER_OF_NODES)
    print(f"\n{nodes_and_weights=}\n")

    actual_result = np.polynomial.legendre.leggauss(NUMBER_OF_NODES)
    print(f"{actual_result=}\n")

    print(compute_integral_numerically(NUMBER_OF_NODES, nodes_and_weights))
