import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

xt = np.load('xt_osc.npy')
y = np.load('y.npy')
xa_enkf = np.load('xa_enkf_osc.npy')
sQest_tray = np.load('sQest_tray_osc.npy')
sRest_tray = np.load('sRest_tray_osc.npy')
rmses_tray = np.load('rmses_tray_osc.npy')
coverages_tray = np.load('coverages_tray_osc.npy')

nx, ne, ncy, nq, nr = xa_enkf.shape

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
plt.subplots_adjust(hspace=0.1, wspace=0.1)

for i in range(nq):
    for j in range(nr):
        
        if (i == 0) and (j == 0):
            legends = ['Valor real', 'Observaciones', 'Media del ensamble', 'Partículas']
        else:
            legends = [None]*4
        ax[i][j].plot(y[0, :], 'r.', label=legends[1], zorder=1000)
        ax[i][j].plot(xt[0, :], 'k--', label=legends[0], zorder=1000)
        ax[i][j].plot(xa_enkf[0, :, :, i, j].mean(0), 'b', label=legends[2], zorder=1000)
        for k in range(ne):
            ax[i][j].plot(xa_enkf[0, k, :, i, j], '0.5', label=legends[3] if k == 0 else None)
            
    if j == 0:
        ax[i][j].set_ylabel('x')

fig.legend(framealpha=1, ncol=2)


plt.show()
