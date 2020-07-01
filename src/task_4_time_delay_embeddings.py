import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf



def lorenz(t, y, sigma, beta, rho):
    return np.array([
        sigma * (y[1] - y[0]),
        rho * y[0] - y[1] - (y[0] * y[2]),
        y[0] * y[1] - beta * y[2]
    ])


# LORENZ ATTRACTOR
def lorenz_attractor(y, sigma=None, beta=None, rho=None, t=None):
    # sigma, beta, rho -> a list of values. the list sizes must be the same.
    # y -> a list of lists containing x, y, z coordinates

    if sigma is None:
        sigma = 10
    if beta is None:
        beta = 8 / 3
    if t is None:
        t = [0, 100]
    if rho is None:
        rho = 28

    res = solve_ivp(lorenz, t, y, args=(sigma, beta, rho))
    x, y, z = res.y

    # reconstructed figure of lorenz attractor along x-axis
    dt = 1
    fig = plt.figure()
    ax3 = fig.gca(projection='3d')
    fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
    ax3.plot(x[:len(z)-2*dt], x[1*dt:len(z)-1*dt], x[2*dt:len(z)])
    plt.show()

    # reconstructed figure of lorenz attractor along y-axis
    dt = 1
    fig = plt.figure()
    ax3 = fig.gca(projection='3d')
    fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
    ax3.plot(y[:len(z)-2*dt], y[1*dt:len(z)-1*dt], y[2*dt:len(z)])
    plt.show()

    # reconstructed figure of lorenz attractor along z-axis
    dt = 1
    fig = plt.figure()
    ax3 = fig.gca(projection='3d')
    fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
    ax3.plot(z[:len(z) - 2 * dt], z[1 * dt:len(z) - 1 * dt], z[2 * dt:len(z)])
    plt.show()

    # original figure
    fig = plt.figure()
    ax3 = fig.gca(projection='3d')
    fig.suptitle(rf'Lorenz system with $\sigma$ = {sigma}, $\rho$ = {rho}, $\beta$ = {beta}')
    ax3.plot(x, y, z)
    plt.show()

    return x


if __name__ == '__main__':
    data = np.loadtxt('../datasets/takens_1.txt')

    # Plot for part 1
    i = 0
    dn = 1
    plt.figure()
    while i < data.shape[0]:
        plt.scatter(i, data[i, 0], c='blue')
        # plt.scatter(i, data[i, 1], c='red')
        i += dn
    plt.xlabel('Time')
    plt.ylabel('x')
    plt.show()

    dt = 50
    no_of_coord = 334

    # Plot x, y of the data-set
    plt.figure()
    plt.plot(data[:no_of_coord, 0], data[:no_of_coord, 1])
    plt.xlabel('y')
    plt.ylabel('x')
    plt.show()

    plt.figure()
    plt.plot(data[range(no_of_coord), 0], data[range(dt, no_of_coord+dt), 0])
    plt.xlabel('dt')
    plt.ylabel('x')
    plt.show()

    # reconstruct lorenz attractor using values along 1-axis
    y = [10, 10, 10]
    x = lorenz_attractor(y)


