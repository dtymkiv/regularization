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


def H_0_0_1(t: float, tau: float) -> float:
    res = -0.5 * np.linalg.norm(x_der_0(tau))
    return res


def H_0_0_2(t: float, tau: float) -> float:
    if abs(t - tau) > core_eps:
        num = 4 / np.e * (np.sin((t - tau) / 2) ** 2)
        denom = np.linalg.norm(x_0(t) - x_0(tau)) ** 2
        res = 0.5 * np.linalg.norm(x_der_0(tau)) * np.log(num / denom)
    else:
        res = 0.5 * np.linalg.norm(x_der_0(t)) * np.log(1 / (np.e * (np.linalg.norm(x_der_0(t)) ** 2)))

    return res


def H_0_1(t: float, tau: float) -> float:
    num = np.dot((x_0(t) - x_1(tau)), nu(x_der_1, tau))
    denom = np.linalg.norm(x_0(t) - x_1(tau)) ** 2
    res = np.linalg.norm(x_der_1(tau)) * num / denom

    return res


def H_1_0(t: float, tau: float) -> float:
    num = -np.dot((x_1(t) - x_0(tau)), nu(x_der_1, tau))
    denom = np.linalg.norm(x_1(t) - x_0(tau)) ** 2
    res = np.linalg.norm(x_der_0(tau)) * num / denom

    return res


def H_1_1(t: float, tau: float) -> float:
    if abs(t - tau) > core_eps:
        num = np.dot((x_1(tau) - x_1(t)), theta(x_der_1, t))
        denom = np.linalg.norm(x_1(tau) - x_1(t)) ** 2
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


def Q_0_0(t: float, tau: float) -> float:
    if abs(t - tau) > core_eps:
        num = -np.dot((x_0(t) - x_0(tau)), nu(x_der_0, tau))
        denom = np.linalg.norm(x_0(t) - x_0(tau)) ** 2
        res = np.linalg.norm(x_der_0(tau)) * num / denom
    else:
        x1 = x_der_0(t)
        x2 = x_der_der_0(t)
        res = (x2[0] * x1[1] - x2[1] * x1[0]) / (2 * np.linalg.norm(x_der_0(t)) ** 2)

    return res


def Q_0_1(t: float, tau: float) -> float:
    res = np.linalg.norm(x_der_1(tau)) * (np.dot(nu(x_der_0, t), nu(x_der_1, tau)) / (np.linalg.norm(x_0(t) - x_1(tau)) ** 2) - 
        2 * (np.dot(x_0(t) - x_1(tau), nu(x_der_1, tau)) * np.dot(x_0(t) - x_1(tau), nu(x_der_0, t))) / (np.linalg.norm(x_0(t) - x_1(tau)) ** 4)
    )

    return res


def Q_1_0(t: float, tau: float) -> float:
    res = np.linalg.norm(x_der_0(tau)) * np.log(1 / np.linalg.norm(x_1(t) - x_0(tau)))

    return res


def Q_1_1(t: float, tau: float) -> float:
    if abs(t - tau) > core_eps:
        num = np.dot((x_1(t) - x_1(tau)), nu(x_der_1, tau))
        denom = np.linalg.norm(x_1(t) - x_1(tau)) ** 2
        res = np.linalg.norm(x_der_1(tau)) * num / denom
    else:
        x1 = x_der_1(t)
        x2 = x_der_der_1(t)
        res = (x1[1] * x2[0] - x1[0] * x2[1]) / (2 * np.linalg.norm(x_der_1(t)) ** 2)

    return res


def dirichlet_neymann(M: int, h: np.array) -> list[np.array, np.array]:
    A = np.zeros((4 * M, 4 * M))
    b = np.zeros(4 * M)
    u = np.zeros(2 * M)
    du = np.zeros(2 * M)
    t = [j * np.pi / M for j in range(2 * M)]

    two_m = 1 / (2 * M)

    # перша чверть
    for i in range(2 * M):
        for j in range(2 * M):
            A[i, j] = two_m * H_0_1(t[i], t[j])

    # друга чверть
    for i in range(2 * M):
        for j in range(2 * M, 4 * M):
            A[i, j] = R(t[i], t[j - 2 * M], M) * H_0_0_1(t[i], t[j - 2 * M]) + two_m * H_0_0_2(t[i], t[j - 2 * M])

    # третя чверть
    for i in range(2 * M, 4 * M):
        pre = 0.5 / np.linalg.norm(x_der_1(t[i - 2 * M]))
        for j in range(2 * M):
            A[i, j] = pre * (T(t[i - 2 * M], t[j], M) + two_m * H_1_1(t[i - 2 * M], t[j]))

    # четверта чверть
    for i in range(2 * M, 4 * M):
        for j in range(2 * M, 4 * M):
            A[i, j] = two_m * H_1_0(t[i - 2 * M], t[j - 2 * M])

    for i in range(2 * M):
        b[i] = h[i]

    for i in range(2 * M, 4 * M):
        b[i] = f_0(t[i - 2 * M])

    psi = np.linalg.solve(A, b)
    psi_1 = psi[:2 * M]
    psi_0 = psi[2 * M:]


    for i in range(2 * M):
        u[i] = (
            -0.5 * psi_1[i] + 
            two_m * sum([psi_1[j] * Q_1_1(t[i], t[j]) for j in range(2 * M)]) + 
            two_m * sum([psi_0[j] * Q_1_0(t[i], t[j]) for j in range(2 * M)])
        )

        du[i] = (
            -0.5 * psi_0[i] + 
            two_m * sum([psi_1[j] * Q_0_1(t[i], t[j]) for j in range(2 * M)]) + 
            two_m * sum([psi_0[j] * Q_0_0(t[i], t[j]) for j in range(2 * M)])
        )

    return u, du
