import numpy as np

from dir_neu import dirichlet_neymann
from data import nu, x_der_1, f_0


def neymann_dirichlet():
    pass


if __name__ == '__main__':
    M = 128
    h = np.ones(2 * M)
    # t = [j * np.pi / M for j in range(2 * M)]
    # h = np.array([1 /(2 * np.pi) * - ((1.3 * np.cos(t[i])  + 1.5) * -1.3 * np.sin(t[i])) / (np.linalg.norm(1.3 * np.cos(t[i])  + 1.5) ** 2) for i in range(2 * M)])
    # # print(np.dot((1.3 * np.cos(t[1])  + 0.05), nu(x_der_1, t[1])))
    # # print((1.3 * np.cos(t[1])  + 0.05) * nu(x_der_1, t[1]))
    u, du = dirichlet_neymann(M, h)
    print(du)
    print(u)
    print(max(map(abs, u)))
    # print(np.array([f_0(t[i]) for i in range(2 * M)]))
