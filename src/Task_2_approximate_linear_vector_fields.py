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

    :param dataset_path: It is the path of the datasets needed.
    :return: none, plots the actual f values and approximated-f values on y-axis
             and x-values on x axis.
    """

    """
    Part1: Following code visualizes X0 and X1 and computes the the vector v as per equation
    1.3 in the exercise sheet. We chose ∆t as .1
    """

    names_x0 = ['x0_x', 'x0_y']
    data_x0 = pd.read_csv(dataset_1_path, sep=' ', names=names_x0)
    X0 = data_x0.to_numpy()
    plt.scatter(X0[:, 0], X0[:, 1], c='blue', alpha=.3)

    names_x1 = ['x1_x', 'x1_y']
    data_x1 = pd.read_csv(dataset_2_path, sep=' ', names=names_x1)
    X1 = data_x1.to_numpy()
    plt.scatter(X1[:, 0], X1[:, 1], c='red', alpha=.3)
    plt.show()

    V = (X1 - X0) / .1

    # x, y = np.meshgrid(np.linspace(-2, 2, 20), np.linspace(-2, 2, 20))
    # u, v = np.zeros((20, 20)), np.zeros((20, 20))
    # for i in range(0, 20):
    #     for j in range(0, 20):
    #         u[i, j] = V.T[0, i * 50]
    #         v[i, j] = V.T[1, j * 50]
    # plt.quiver(x, y, u, v)
    # plt.streamplot(x, y, u, v)
    plt.show()

    """
    Part2: Following code approximates and prints the A matrix (A_hat) using the following equation.
    A^T= (X0^T.X0)^(−1).X0^T.V
    We also compute X1 predictions using the approximated A_hat matrix 
    and solve_ivp as differential equation solver to find X1 approximations at T=0.1
    Using these values we calculateour Mean Squared Error (MSE)
    """

    A_hat = (np.linalg.inv(X0.T @ X0) @ X0.T @ V).T
    print("A_hat: " + str(A_hat))

    X1_approx = []
    for i in range(X0.shape[0]):
        sol = solve_ivp(fun=deriv, t_span=[0, 0.2], t_eval=[0.1], y0=X0[i, :], args=(A_hat,))
        X1_approx.append(sol.y)
    X1_approx = np.array(X1_approx)
    X1_approx = X1_approx.reshape((1000, 2))

    # MSE
    MSE = np.square(X1 - X1_approx).mean()
    print("MSE: " + str(MSE))

    """
    Part3: Following code deals with part 3 of of the task.
    We compute first 100 points with with earlier chosen ∆t of 0.1
    and plot them to show trajectory
    We also compute points for first 100 seconds (Time step = 1)
    and plot them to show trajectory
    Then we plot the phase portrait of the trajectory.
    """
    t_eval = [i/10 for i in range(100)]  # For first 100 points (Evaluated at at .1,.2,.3,...)
    sol = solve_ivp(fun=deriv, t_span=[0, 100], y0=[10, 10], t_eval=t_eval, args=(A_hat,))
    plt.scatter(sol.y[0, :], sol.y[1, :])
    plt.xlabel("x_coordinate")
    plt.ylabel("y_coordinate")
    plt.show()

    x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    u, v = np.zeros((20, 20)), np.zeros((20, 20))
    for i in range(0, 20):
        for j in range(0, 20):
            u[i, j] = sol.y[0, i * 5]
            v[i, j] = sol.y[1, j * 5]
    plt.quiver(x, y, u, v)
    plt.streamplot(x, y, u, v)
    plt.show()

    t_eval = [i for i in range(100)] # For first 100 seconds (Evaluated at at 1,2,3,...)
    sol = solve_ivp(fun=deriv, t_span=[0, 100], y0=[10, 10], t_eval=t_eval, args=(A_hat,))
    plt.plot(sol.y[0, :], sol.y[1, :])
    plt.xlabel("x_coordinate")
    plt.ylabel("y_coordinate")
    plt.show()

    x, y = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    u, v = np.zeros((20, 20)), np.zeros((20, 20))
    for i in range(0, 20):
        for j in range(0, 20):
            u[i, j] = sol.y[0, i * 5]
            v[i, j] = sol.y[1, j * 5]
    plt.quiver(x, y, u, v)
    plt.streamplot(x, y, u, v)
    plt.show()


if __name__ == '__main__':
    cwd = Path.cwd()
    path = cwd
    approximate_linear_vector_fields(path / "datasets" / "linear_vectorfield_data_x0.txt",
                                     path / "datasets" / "linear_vectorfield_data_x1.txt")
