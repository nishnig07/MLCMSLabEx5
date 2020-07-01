"""
Task 3: Approximating non linear vector fields
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def derivative_func(t, x, Approx_func):
    """
    This function is used by the solve_ivp to calculate the solution for the
    differential equation.

    :param t: time
    :param x: the initial x values
    :param Approx_func: the linear operator A
    :return: dot product of x and Approx_func
    """
    return x.dot(Approx_func)


def approximate_nonlinear_vector_field(dataset_path):
    """
    This method approximates the values of X1 in data sets, plots
    the initial points, and then calculates the approximated values
    using linear function and then calculates the mean squared error.

    :param dataset_path: This is the path of the data set
    :return: None, plots and calculates the MSE.
    """

    file_X0 = "nonlinear_vectorfield_data_x0.txt"
    names_X0 = ['X0_x', 'X0_y']
    data_X0 = pd.read_csv(dataset_path / file_X0, sep=' ', names=names_X0).to_numpy()
    plt.scatter(data_X0[:, 0], data_X0[:, 1])

    names_X1 = ['X1_x', 'X1_y']
    file_X1 = "nonlinear_vectorfield_data_x1.txt"
    data_X1 = pd.read_csv(dataset_path / file_X1, sep=' ', names=names_X1).to_numpy()
    plt.scatter(data_X1[:, 0], data_X1[:, 1])
    plt.title("Given data set X0 and X1")
    plt.show()

    """
    Following block calculates the approximate values using differential
    solver solve_ivp
    """
    V = (data_X1 - data_X0) / 0.1
    approx_func_At = np.linalg.inv(data_X0.T @ data_X0) @ data_X0.T @ V
    approx_values = []
    for i in range(data_X0.shape[0]):
        sol = solve_ivp(fun=derivative_func, t_span=[0, 10], t_eval=[0.1],
                        y0=data_X0[i, :], args=(approx_func_At,))
        approx_values.append(sol.y)
    approx_values = np.array(approx_values)
    approx_values = approx_values.reshape((2000, 2))

    """
    We now plot the original data of X1 and the newly approximated data.
    """
    plt.scatter(data_X1[:, 0], data_X1[:, 1])
    plt.scatter(approx_values[:, 0], approx_values[:, 1], c='green')
    plt.title("Given X1 and approximated values")
    plt.title("Approximated vector field")
    plt.show()

    """
    We now plot the vector filed and the phase portrait.
    """
    x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    u, v = np.zeros((10, 10)), np.zeros((10, 10))
    for i in range(0, 10):
        for j in range(0, 10):
            u[i, j] = approx_values.T[0, i]
            v[i, j] = approx_values.T[1, j]
    plt.quiver(x, y, u, v)
    plt.streamplot(x, y, u, v)
    plt.title("Approximated Vector field")
    plt.show()

    """
    Following block calculates the mean squared error of the X1 and calculate
    approximated values.
    """
    MSE = np.square(data_X1 - approx_values).mean()
    print(MSE)


def approximate_nonlinear_vector_field_radial(dataset_path, L, epsilon):
    """
    This method approximates the values of X1 in data sets, plots
    the initial points, and then calculates the approximated values
    using linear function and then calculates the mean squared error.

    :param dataset_path: This is the path of the data set
    :param L: This is the number of coefficients/centres we want to find.
    :param epsilon: It is the bandwidth.

    :return: None, calculates the MSE.
    """

    file_X0 = "nonlinear_vectorfield_data_x0.txt"
    names_X0 = ['X0_x', 'X0_y']
    data_X0 = pd.read_csv(dataset_path / file_X0, sep=' ', names=names_X0).to_numpy()

    names_X1 = ['X1_x', 'X1_y']
    file_X1 = "nonlinear_vectorfield_data_x1.txt"
    data_X1 = pd.read_csv(dataset_path / file_X1, sep=' ', names=names_X1).to_numpy()

    """
    Following block calculates the values of phi_l's for each point in dataset of X0
    and form the corresponding phi_X matrix with the given value of L.
    """
    phi = np.empty([2000, L])
    for l in range(L):
        phi_l = np.exp(-np.square(np.linalg.norm(data_X0 - data_X0[l],
                                                 axis=1)) / epsilon ** 2)
        phi[:, l] = phi_l

    """
    The following block performs the approximation of  the vector field.
    """
    V = (data_X1 - data_X0) / 0.1
    approx_func_Ct = np.linalg.inv(phi.T @ phi) @ phi.T @ V
    final = phi @ approx_func_Ct
    plt.scatter(final[:, 0], final[:, 1], c='green',
                label='approximated f(x)_hat values')
    plt.show()

    """
    The following code plots the approximated vector field and the phase portrait.
    """
    x, y = np.meshgrid(np.linspace(-5, 5, 10), np.linspace(-5, 5, 10))
    u, v = np.zeros((10, 10)), np.zeros((10, 10))
    for i in range(0, 10):
        for j in range(0, 10):
            u[i, j] = final.T[0, i]
            v[i, j] = final.T[1, j]
    plt.quiver(x, y, u, v)
    plt.streamplot(x, y, u, v)
    plt.title("Approximated Vector field")
    plt.show()

    """
    The following code calculates the MSE for the dataset X1 and the final values.
    """
    MSE = np.square(data_X1 - final).mean()
    print(MSE)


if __name__ == '__main__':
    cwd = Path.cwd()
    path = path = cwd / "datasets"
    approximate_nonlinear_vector_field(path)
    # approximate_nonlinear_vector_field_radial(path,
    #                                           L=500,
    #                                           epsilon=0.01)
