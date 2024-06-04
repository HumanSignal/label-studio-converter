import math


def almost_equal_1d(a, b, tol=1e-9):
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))

def almost_equal_2d(a, b, tol=1e-9):
    a = [z for x in a for z in x]
    b = [z for x in b for z in x]
    return all(math.isclose(x, y, abs_tol=tol) for x, y in zip(a, b))