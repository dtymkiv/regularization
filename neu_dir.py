import numpy as np

from data import (
    f_0,
    x_0,
    x_der_0,
    x_der_der_0,
    x_1,
    x_der_1,
    x_der_der_1,
    x_der_der_der_1,
    core_eps,
    nu,
    theta,
    R,
    T
)


def H_1_1(t: float, tau: float) -> float:
    """
    #TODO update to G_0_0
    """
    if abs(t - tau) > core_eps:
        num = np.dot((x_0(tau) - x_0(t)), theta(x_der_0, t))
        denom = np.linalg.norm(x_0(tau) - x_0(t)) ** 2
        res = num / denom
    else:
        x1 = x_der_1(t)
        x2 = x_der_der_1(t)
        x3 = x_der_der_der_1(t)
        res = (
            1 / (6 * (x1[0] ** 2 + x1[1] ** 2) ** 2 ) *
            (
                -x1[0] ** 4 + 2 * x1[0] ** 3 + 2 * x1[0] * x1[1] * (-6 * x2[0] * x2[1] + x1[1] * x3[0]) -
                x1[1] ** 2 * (x1[1] ** 2 - 3 * x2[0] ** 2 + 3 * x2[1] ** 2 - 2 * x1[1] * x3[1]) +
                x1[0] ** 2 * (-3 * x2[0] ** 2 + 3 * x2[1] ** 2 + 2 * x1[1] * (-x1[1] + x3[1]))
            )
        )
    return res