"""
Task 3: Approximating non linear vector fields
"""
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def deriv(t, x, A):
    return x.dot(A)


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

    names_X1 = ['X1_x', 'X1_y']
    file_X1 = "nonlinear_vectorfield_data_x1.txt"
    data_X1 = pd.read_csv(dataset_path / file_X1, sep=' ', names=names_X1).to_numpy()

    """
    Following block calculates the approximate values using differential
    solver solve_ivp
    """
    V = (data_X1 - data_X0) / 0.1
    approx_func_At = np.linalg.inv(data_X0.T @ data_X0) @ data_X0.T @ V
    approx_values = []
    for i in range(data_X0.shape[0]):
        sol = solve_ivp(fun=deriv, t_span=[0, 0.2], t_eval=[0.1], y0=data_X0[i, :], args=(approx_func_At,))
        approx_values.append(sol.y)
    approx_values = np.array(approx_values)
    approx_values = approx_values.reshape((2000, 2))
    plt.scatter(approx_values[:, 0], approx_values[:, 1])
    plt.show()

    """
    Following block calculates the mean squared error of the X1 and calculate
    approximated values.
    """
    # mean_squared_error = np.sum(np.square(data_X1 - approx_values)) / 2000
    mean_squared_error = np.square(data_X1 - approx_values).mean()
    print(mean_squared_error)


def approximate_nonlinear_vector_field_radial(dataset_path, L, L_eval, epsilon):
    """
    This method approximates the values of X1 in data sets, plots
    the initial points, and then calculates the approximated values
    using linear function and then calculates the mean squared error.

    :param dataset_path: This is the path of the data set
    :param L: This is the number of coefficients/centres we want to find.
    :param L_eval: This is the index of centre where we want to approximate the field.
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
    phi = []
    for eachpoint in range(L):
        phi_l = np.exp(-np.square((data_X0 - data_X0[eachpoint]) / epsilon))
        phi.append(phi_l)
    phi = np.array(phi)
    phi = phi.reshape((2000, 2, L))

    """
    The following block performs the approximation of  the vector field.
    
    If finds the values for each of the value of L, and we can evaluate it at the
    required value, L_eval.
    """
    V = (data_X1 - data_X0) / 0.1
    Ct = []
    for each_L in range(L):
        approx_func_Ct = np.linalg.inv(phi[:, :, each_L].T @ phi[:, :, each_L]) @ phi[:, :, each_L].T @ V
        Ct.append(approx_func_Ct)
    Ct = np.array(Ct)
    approx_values = phi[:, :, L_eval] @ Ct[L_eval, :, :]

    """
    The following code finds the MSE for the dataset X1 and the approx values at L_eval.
    Best value of L_eval will be the value at which the MSE will be minimum.
    """
    MSE = np.square(data_X1 - approx_values).mean()
    print(MSE)


if __name__ == '__main__':
    cwd = Path.cwd()
    path = path = cwd / "datasets"
    # approximate_nonlinear_vector_field(path)
    approximate_nonlinear_vector_field_radial(path,
                                              L=1000,
                                              L_eval=500,
                                              epsilon=0.5)