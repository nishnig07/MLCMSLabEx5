import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



data = np.loadtxt('../datasets/MI_timesteps.txt')
data = data[1000:, :]
print(data.shape)

# plt.figure()
# plt.plot(data[1000:, 0], data[1000:, 1])
# plt.plot(data[1000:, 0], data[1000:, 2])
# plt.plot(data[1000:, 0], data[1000:, 3])
# plt.show()
#
#
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.plot(data[10000:data.shape[0]-700:350, 1], data[10350:data.shape[0]-350:350, 1], data[10700:data.shape[0]:350, 1])
# ax.plot(data[10000:data.shape[0]-700:350, 3], data[10350:data.shape[0]-350:350, 3], data[10700:data.shape[0]:350, 3])
# plt.show()

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

centered_dataset = (windows - windows.mean())
U, s, Vt = np.linalg.svd(centered_dataset, full_matrices=True)
principalDf = centered_dataset @ Vt.T[:, :3]

# centered_dataset = (windows - windows.mean())
# pca = PCA(n_components=3)
# principalDf = pca.fit_transform(centered_dataset)
# print(principalDf.shape)

for i in range(1, 2):
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

    plt.title('Data plotted against 3 PCs')
    plt.show()


for i in range(1, 2):
    threedee = plt.figure().gca(projection='3d')
    threedee.scatter(principalDf[:2000, 0],
                     principalDf[:2000, 1],
                     principalDf[:2000, 2],
                     # n=1,
                     c=data[:2000, i],
                     cmap='jet'
                     )
    threedee.set_xlabel('principal component 1')
    threedee.set_ylabel('principal component 2')
    threedee.set_zlabel('principal component 3')

    plt.title('Data plotted against 3 PCs')
    plt.show()

arc_length = []
vec_field = principalDf[1:2001] - principalDf[0:2000]
for i in range(1, 2001):
    arc_length.append(np.linalg.norm(principalDf[i - 1, :] - principalDf[i, :]))

arc_length = np.array(arc_length)
print(np.sum(arc_length))

L = 7
phi = np.empty([2000, L])

for eachpoint in range(L):
    phi_l = np.exp(-np.square((arc_length - arc_length[eachpoint]) / 5.7))
    # plt.scatter(range(arc_length.shape[0]), phi_l)
    # print(phi_l.shape)
    phi[:, eachpoint] = phi_l
plt.show()

approx_func_Ct = np.linalg.inv(phi.T @ phi) @ phi.T @ arc_length
MSE = np.square(arc_length - phi@approx_func_Ct).mean()
print(MSE)

plt.scatter(range(arc_length.shape[0]), arc_length)
plt.scatter(range(arc_length.shape[0]), phi@approx_func_Ct, c='green',
         label='approximated f(x)_hat values')
plt.show()



# print(vec_field.shape)
# threedee = plt.figure().gca(projection='3d')
# threedee.scatter(vec_field[:2000, 0],
#                  vec_field[:2000, 1],
#                  vec_field[:2000, 2],
#                  # n=1,
#                  c=data[:2000, 1],
#                  cmap='jet'
#                  )
# threedee.set_xlabel('principal component 1')
# threedee.set_ylabel('principal component 2')
# threedee.set_zlabel('principal component 3')
# plt.title('Data plotted against 3 PCs')
# plt.show()

# fig = plt.figure()
# ax = fig.gca(projection='3d')
# x, y, z = np.meshgrid(np.linspace(-10, 15, 10), np.linspace(-10, 15, 10), np.linspace(-7.5, 7.5, 10))
# u, v, w = vec_field[::200, 0], vec_field[::200, 1], vec_field[::200, 2]
# ax.quiver(x, y, z, u, v, w)
# plt.show()
