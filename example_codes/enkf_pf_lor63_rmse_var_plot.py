import numpy as np
import matplotlib.pyplot as plt

nes = np.load('nes.npy')
rmses = np.load('rmses.npy')
vars = np.load('vars.npy')

fig, ax = plt.subplots(2, 1, sharex=True)
plt.subplots_adjust(hspace=0.1)

labels = ['EnKF', 'Bootstrap PF', 'VMPF']
for i in range(3):
    ax[0].plot(nes, rmses[i, :], marker='s', label=labels[i])
    ax[1].plot(nes, vars[i, :], marker='s')

ax[0].tick_params(axis='x', bottom=False)
ax[0].set_ylabel('RMSE')
ax[1].set_ylabel('Varianza')

fig.legend(framealpha=1)

plt.show()