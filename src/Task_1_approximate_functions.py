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
    plt.scatter(data[:, 0], data[:, 1], c='blue',
                label='actual f(x) values')
    plt.plot(data[:, 0], Approx_func_XAt, c='green',
             label='approximated f(x)_hat values')
    plt.xlabel('x values')
    plt.ylabel('f(x) and f(x)_hat values')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    cwd = Path.cwd()
    print(cwd)
    path = path = cwd / "datasets"
    approximateData_LinearFunction(path / "linear_function_data.txt")
    approximateData_LinearFunction(path / "nonlinear_function_data.txt")
