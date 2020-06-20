"""
Task 1: Approximating functions
"""

"""
Part 1 and 2: To approximate the function in dataset 
'linear_function_data.txt' and 'nonlinear_function_data.txt'
with a linear function.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import minimize, rosen


def approximateData_LinearFunction(dataset_path):
    """
    This method implements the approximation for the given dataset.
    The approximation is calculated by:
    A^T= (X^T.X)^(−1).X^T.F

    and the predicted values are:
    F_hat = X.A^T

    :param dataset_path: It is the path of the datasets needed.
    :return: none, plots the actual f values and approximated-f values on y-axis
             and x-values on x axis.
    """

    """
    Following block loads the data into columns 'x' and 'f(x)'
    """
    names = ['x', 'f(x)']
    data = pd.read_csv(dataset_path, sep=' ', names=names).to_numpy()
    X_array = data[:, 0]
    X = data[:, 0].reshape((1000, 1))
    f = data[:, 1].reshape((1000, 1))
    """
    Following block approximates using least-squares minimization formula mentioned above
    """
    approx_func_A = np.linalg.inv(X.T @ X) @ X.T @ f
    Approx_func_XAt = X * approx_func_A.T

    """
    Following block plots the x-values to actual f-values and approximated f-values  
    """
    plt.scatter(data[:, 0], data[:, 1],
                c='blue',
                label='actual f(x) values')
    plt.plot(data[:, 0], Approx_func_XAt, c='green',
             label='approximated f(x)_hat values')
    plt.xlabel('x values')
    plt.ylabel('f(x) and f(x)_hat values')
    plt.legend()
    plt.show()

    """
    Following code uses in-built library for least squares
    np.linalg.lstsq
    """
    X_stacked = np.vstack([X_array, np.ones(len(X_array))]).T
    m, _ = np.linalg.lstsq(X_stacked, data[:, 1], rcond=None)[0]
    plt.plot(X_array, data[:, 1], 'o', label='actual f(x) values')
    plt.plot(X_array, m * X_array, 'r', label='approximated f(x)_hat values')
    plt.xlabel('x values')
    plt.ylabel('f(x) and f(x)_hat values')
    plt.legend()
    plt.show()



def approximateData_RadialBasisFunction(dataset_path, L, epsilon):
    """
    This method implements the approximation for the given dataset.
    The approximation is done using radial basis functions, where
    f(x)_hat is the approximated function and is given by:

    f(x) = c_1*phi_1 + c_2*phi_2 . . . c_l*phi_l
    :param dataset_path: It is the path of the dataset
    :param L: number of phi functions needed
    :return:
    """

    """
    Following block loads the data into columns 'x' and 'f(x)'
    """
    names = ['x', 'f(x)']
    data = pd.read_csv(dataset_path, sep=' ', names=names).to_numpy()
    X_array = data[:, 0]
    X = data[:, 0].reshape((1000, 1))
    f = data[:, 1].reshape((1000, 1))
    """
    for any specific datapoint x, we calculate corresponding phi_l's.
    phi_l = 1, when x_l = datapoint.
    """
    phi = np.empty([1000, L])
    for eachpoint in range(L):
        phi_l = np.exp(-np.square((X_array - X_array[eachpoint]) / epsilon))
        plt.scatter(X_array, phi_l)
        phi[:, eachpoint] = phi_l
    plt.show()
    """
    Now we calculate the Coefficient atrix which will decide the peak
    of the phi functions to give the f(x)_hat approximated values.
    """
    approx_func_Ct = np.linalg.inv(phi.T @ phi) @ phi.T @ f
    plt.scatter(data[:, 0], data[:, 1],
                c='blue',
                label='actual f(x) values')
    plt.scatter(data[:, 0], phi@approx_func_Ct, c='green',
             label='approximated f(x)_hat values')
    plt.show()


if __name__ == '__main__':
    cwd = Path.cwd()
    print(cwd)

    path = cwd / "datasets"
    approximateData_LinearFunction(path / "linear_function_data.txt")
    approximateData_LinearFunction(path / "nonlinear_function_data.txt")
    approximateData_RadialBasisFunction((path / "nonlinear_function_data.txt"),
                                        7,
                                        epsilon=0.8)

