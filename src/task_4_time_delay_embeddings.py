import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def lorenz(t, y, sigma, beta, rho):
    return np.array([
        sigma * (y[1] - y[0]),
        rho * y[0] - y[1] - (y[0] * y[2]),
        y[0] * y[1] - beta * y[2]
    ])


# LORENZ ATTRACTOR
def plot_lorenz_attractor(y, _sigma=None, _beta=None, _rho=None, t=None):
    # sigma, beta, rho -> a list of values. the list sizes must be the same.
    # y -> a list of lists containing x, y, z coordinates

    if _sigma is None:
        _sigma = [10]
    if _beta is None:
        _beta = [8 / 3]
    if t is None:
        t = [0, 100]
    if _rho is None:
        _rho = [28]

    for sigma, beta, rho in zip(_sigma, _beta, _rho):
        alpha = 1
        fig = plt.figure()
        ax3 = fig.gca(projection='3d')
        fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
        for y0 in y:
            res = solve_ivp(lorenz, t, y0, args=(sigma, beta, rho))
            x, y, z = res.y
            lab = str("x = " + str(y0[0]) + ", y = " + str(y0[1]) + ", z = " + str(y0[2]))
            dt = 1
            i = 0
            # while i < x.shape[0] - 2 * dt:

            ax3.plot(x[:len(z)-2], x[1:len(z)-1], x[2:len(z)])
                # i += 2 * dt + 1
            alpha *= 0.65

        # ax3.legend()
        plt.show()


if __name__ == '__main__':
    data = np.loadtxt('../datasets/takens_1.txt')

    # Plot for part 1
    i = 0
    dn = 1
    plt.figure()
    while i < data.shape[0]:
        # if data[i, 0] in hist:
        #     break
        plt.scatter(i, data[i, 0], c='blue')
        plt.scatter(i, data[i, 1], c='red')
        i += dn
    plt.show()

    # Plot takens theory
    plt.figure()
    plt.plot(data[:400, 0], data[:400, 1])
    plt.show()

    # Plot lorenz attractor with only x, for verifying takens theory
    y = [[10, 10, 10]]
    plot_lorenz_attractor(y)
