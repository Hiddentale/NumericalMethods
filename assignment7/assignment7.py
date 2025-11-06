def Lagrange_pol(index: int, x_pts: list):
    None


def multiply_polynomials(polynomial1: list, polynomial2: list):
    result = [0] * (len(polynomial1) + len(polynomial2) - 1)
    for i, coefficient_1 in enumerate(polynomial1):
        for j, coefficient_2 in enumerate(polynomial2):
            result[i, j] += coefficient_1 * coefficient_2
    return result


def calculate_denominator(index, x_pts: list):
    product = 0
    x_i = x_pts[index]
    for i in range(len(x_pts)):
        if i == index:
            pass
        product *= x_i - x_pts[i]
    return product
