import numpy as np


eye = np.array([[0, 1], [-1, 0]])
core_eps = 1e-3


def f_0(x: np.array) -> float:
    """
    похідна
    """
    # return 1 / (2 * np.pi) * np.log(1 / (0.5 * np.cos(x) + 1.5))
    return 1


def f_1(x: np.array) -> float:
    """
    функція
    """
    return 0


def x_0(t: float) -> np.array:
    return np.array(
        [
            0.5 * np.cos(t),
            0.4 * np.sin(t) - 0.3 * (np.sin(t) ** 2)
        ])


def x_1(t: float) -> np.array:
    return np.array(
        [
            1.3 * np.cos(t),
            np.sin(t)
        ])


def x_der_0(t: float) -> np.array:
    return np.array(
        [
            -0.5 * np.sin(t),
            0.4 * np.cos(t) - 2 * 0.3 * np.sin(t) * np.cos(t)
        ])


def x_der_der_0(t: float) -> np.array:
    return np.array(
        [
            -0.5 * np.cos(t),
            -0.4 * np.sin(t) - 2 * 0.3 * np.cos(t) ** 2 + 0.6 * np.sin(t) ** 2
        ])


def x_der_1(t: float) -> np.array:
    return np.array(
        [
            -1.3 * np.sin(t),
            np.cos(t)
        ])


def x_der_der_1(t: float) -> np.array:
    return np.array(
        [
            -1.3 * np.cos(t),
            -np.sin(t)
        ])


def x_der_der_der_1(t: float) -> np.array:
    return np.array(
        [
            1.3 * np.sin(t),
            -np.cos(t)
        ])


def theta(func, t: float) -> np.array:
    """
    pass x_der_i into this function, not x_i
    """
    res = func(t) / np.linalg.norm(func(t))
    return res


def nu(func, t:float) -> np.array:
    """
    pass x_der_i into this function, not x_i
    """
    res = np.dot(eye, theta(func=func, t=t))
    return res


def T(t: float, tau: float, M: int) -> float:
    res = - 1/M * sum([m * np.cos(m * (t - tau)) for m in range(1, M) - 0.5 * np.cos(M * (t - tau))])

    return res


def R(t: float, tau: float, M: int) -> float:
    res = - 1/M * sum([1 / m * np.cos(m * (t - tau)) for m in range(1, M) - 1 / (2 * M ** 2) * np.cos(M * (t - tau))])

    return res