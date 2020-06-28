"""
Task 2: Approximating Linear Vector Fields
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import odeint, solve_ivp


def deriv(t, x, A):
    return x.dot(A)


def approximate_linear_vector_fields(dataset_1_path, dataset_2_path):
    """
    This method implements the approximation for the given 2D dataset.
    The approximation is calculated by:
    A^T= (X0^T.X0)^(−1).X0^T.V

    # and the predicted values are:
    # F_hat = X.A^T

    :param dataset_path: It is the path of the datasets needed.
    :return: none, plots the actual f values and approximated-f values on y-axis
             and x-values on x axis.
    """

    names_x0 = ['x0_x', 'x0_y']
    data_x0 = pd.read_csv(dataset_1_path, sep=' ', names=names_x0)
    X0 = data_x0.to_numpy()
    # plt.scatter(X0[:, 0], X0[:, 1], c='blue', alpha=.3)

    names_x1 = ['x1_x', 'x1_y']
    data_x1 = pd.read_csv(dataset_2_path, sep=' ', names=names_x1)
    X1 = data_x1.to_numpy()
    # plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=.3)

    # for i in range(1000):
    #     x, y = X0[i,0], X0[i,1]
    #     u, v = X1[i, 0], X1[i, 1]
    #     plt.quiver(x, y, u, v)
    #     # plt.streamplot(x,y,u,v)
    # plt.show()

    V = (X1-X0)/.1
    A_hat = (np.linalg.inv(X0.T @ X0) @ X0.T @ V).T
    print(A_hat)


    # Ab = np.array([[-0.25, 0, 0],
    #                [0.25, -0.2, 0],
    #                [0, 0.2, -0.1]])

    # time = np.array([0])
    # A0 = np.array([10, 20, 30])

    MA = []
    for i in range(X0.shape[0]):
        sol = solve_ivp(fun=deriv, t_span=[0, 0.2], t_eval=[0.1], y0=X0[i,:] , args=(A_hat,))
        MA.append(sol.y)
    MA = np.array(MA)
    MA = MA.reshape((1000,2))

    #MSE
    MSE = np.square(X1-MA).mean()


    # 3rd part
    t_eval = [i/20 for i in range(100)]
    sol = solve_ivp(fun=deriv, t_span=[0, 100], y0=[10,10], t_eval= t_eval, args=(A_hat,))
    plt.scatter(sol.y[0, :], sol.y[1, :])
    plt.show()

    x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    u, v = np.zeros((20, 20)), np.zeros((20, 20))
    for i in range(0,20):
        for j in range(0, 20):
            u[i,j] = sol.y[0,i*5]
            v[i,j] = sol.y[1,j*5]
    plt.quiver(x,y,u,v)
    plt.streamplot(x,y, u, v)
    plt.show()
    a = ''



if __name__ == '__main__':
    cwd = Path.cwd()
    print(cwd)
    path = cwd
    approximate_linear_vector_fields(path / "datasets" / "linear_vectorfield_data_x0.txt",
                                     path / "datasets" / "linear_vectorfield_data_x1.txt")
    # approximateData_LinearFunction(path / "linear_vectorfield_data_x1.txt")
