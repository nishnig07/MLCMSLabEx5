import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.interpolate import Rbf
from scipy.integrate import cumtrapz


def pca(windows):
    """
    returns data after performing PCA
    :param windows: window of shape 1053 dims.
    :return: np.ndarray of size 3 having the principle components
    """

    # Self-implemented PCA
    # centered_dataset = (windows - windows.mean())
    # U, s, Vt = np.linalg.svd(centered_dataset, full_matrices=True)
    # principalDf = centered_dataset @ Vt.T[:, :3]

    # in-built PCA
    centered_dataset = (windows - windows.mean())
    pca = PCA(n_components=3)
    principalDf = pca.fit_transform(centered_dataset - centered_dataset.mean())
    print(principalDf.shape)

    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(principalDf[:, 0],
                     principalDf[:, 1],
                     principalDf[:, 2],
                     )
    threedee.set_xlabel('principal component 1')
    threedee.set_ylabel('principal component 2')
    threedee.set_zlabel('principal component 3')

    plt.title('PCA')
    plt.show()

    return principalDf


def plots_9(principalDf, data):
    """
    plot the 9 require plots for the second part
    :param principalDf: the data of the principal components
    :param data: loaded data-set
    :return: none
    """
    for i in range(1, 10):
        threedee = plt.figure().gca(projection='3d')
        threedee.scatter(principalDf[:, 0],
                         principalDf[:, 1],
                         principalDf[:, 2],
                         # n=1,
                         c=data[:13651, i],
                         cmap='jet'
                         )
        threedee.set_xlabel('principal component 1')
        threedee.set_ylabel('principal component 2')
        threedee.set_zlabel('principal component 3')

        plt.title('Data plotted against 3 PCs for column ' + str(i))
        plt.show()


def plot_arclength(principalDf, data):
    """
    plot approximately one loop of the figure
    :param principalDf: principal component data
    :param data: data
    :return: none
    """
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(principalDf[:2000, 0],
                     principalDf[:2000, 1],
                     principalDf[:2000, 2],
                     # n=1,
                     c=data[:2000, 1],
                     cmap='jet'
                     )
    threedee.set_xlabel('principal component 1')
    threedee.set_ylabel('principal component 2')
    threedee.set_zlabel('principal component 3')

    plt.title('Data plotted against 3 PCs with 2000 coordinates for column 1')
    plt.show()


if __name__ == '__main__':
    # Load the data
    data = np.loadtxt('../datasets/MI_timesteps.txt')
    data = data[1000:, :]
    print(data.shape)

    # Make the PCA windows
    windows = []
    i = 0
    while i < data.shape[0]:
        window = [data[i:i + 351, 1], data[i:i + 351, 2], data[i:i + 351, 3]]
        windows.append(np.array(window).T.flatten())
        if i + 351 < data.shape[0]:
            i += 1
        else:
            break

    windows = np.array(windows)
    print(windows.shape, windows[-1].shape)

    # Calculate the PCA
    principalDf = pca(windows)

    # Part 1, 2, 3 plots
    plots_9(principalDf, data)
    plot_arclength(principalDf, data)

    # Print arc length
    arc_length = []
    for i in range(1, 2001):
        arc_length.append(np.linalg.norm(principalDf[i - 1, :] - principalDf[i, :]))
    arc_length = np.array(arc_length)
    print(np.sum(arc_length))

    # Vector field
    vec_field = principalDf[1:2001] - principalDf[0:2000]

    # Plot vector fields
    plt.figure()
    plt.plot(range(2000), vec_field[:, 0], label="Change in x")
    plt.plot(range(2000), vec_field[:, 1], label="Change in y")
    plt.plot(range(2000), vec_field[:, 2], label="Change in Z")
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Difference')
    plt.show()

    # RBF approximation
    rbfx = Rbf(range(2000), vec_field[:, 0], function='gaussian')
    rbfy = Rbf(range(2000), vec_field[:, 1], function='gaussian')
    rbfz = Rbf(range(2000), vec_field[:, 2], function='gaussian')

    # Interpolate new values
    dx, dy, dz = rbfx(range(2000)), rbfy(range(2000)), rbfz(range(2000))

    MSE = np.square(vec_field[:, 0] - dx).mean()
    print('MSE dx = ', MSE)

    MSE = np.square(vec_field[:, 1] - dy).mean()
    print('MSE dy = ', MSE)

    MSE = np.square(vec_field[:, 2] - dz).mean()
    print('MSE dz = ', MSE)

    x = []
    y = []
    z = []
    # Actually integrate them
    for day in range(14):
        x.append(cumtrapz(dx, range(14000 + 2000 * day, 16000 + 2000 * day)))
        y.append(cumtrapz(dy, range(14000 + 2000 * day, 16000 + 2000 * day)))
        z.append(cumtrapz(dz, range(14000 + 2000 * day, 16000 + 2000 * day)))

    # Plot the integrated values
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(x,
                     y,
                     z,
                     # n=1,
                     c=x,
                     cmap='jet'
                     )
    threedee.set_xlabel('principal component 1')
    threedee.set_ylabel('principal component 2')
    threedee.set_zlabel('principal component 3')
    plt.title('Predicted colors for 14 days')
    plt.show()
