import numpy as np
import matplotlib.pyplot as plt

fig, ax = plt.subplots(ncols=2, nrows=1)

nx, ny = (30, 20)
x = np.linspace(0, 1, nx)
y = np.linspace(0, 2, ny)
xv, yv = np.meshgrid(x, y)

# map1 = ax[0].pcolormesh(xv)
# map2 = ax[1].pcolormesh(yv)

map1 = ax[0].contourf(xv)
map2 = ax[1].contourf(yv)

cbar1 = fig.colorbar(map1, ax=ax[0])
cbar2 = fig.colorbar(map2, ax=ax[1])

plt.show()