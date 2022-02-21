import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

xt = np.load('xt_osc.npy')
y = np.load('y.npy')
xa_enkf = np.load('xa_enkf_osc.npy')
sQest = np.load('sQest_tray_osc.npy')
sRest = np.load('sRest_tray_osc.npy')
rmses = np.load('rmses_tray_osc.npy')
coverages = np.load('coverages_tray_osc.npy')

start = 35
stop = 70
xrange = np.arange(start, stop)

xt = xt[:, start:stop]
y = y[:, start:stop]
xa_enkf = xa_enkf[:, :, start:stop, ...]

nx, ne, ncy, nq, nr = xa_enkf.shape

fig, ax = plt.subplots(3, 3, sharex=True, sharey=True)
plt.subplots_adjust(
    top=0.85, bottom=0.095, left=0.11, right=0.9, hspace=0.0, wspace=0.0)

signs = ['<', '=', '>']
props = dict(facecolor='white', alpha=1)

for i in range(nq):
    for j in range(nr):
        
        if (i == 0) and (j == 0):
            legends = ['Valor real', 'Observaciones', 'Media del ensamble', 'Partículas']
        else:
            legends = [None]*4
        ax[i][j].plot(xrange, y[0, :], 'r.', label=legends[1], zorder=1000)
        ax[i][j].plot(xrange, xt[0, :], 'k--', label=legends[0], zorder=1000)
        ax[i][j].plot(xrange, xa_enkf[0, :, :, i, j].mean(0), 'b', label=legends[2], zorder=1000)

        textstr = '\n'.join((
            # fr'Q {signs[i]} Q_t',
            # fr'R {signs[j]} R_t',
            fr'RMSE: {rmses[i, j].round(2)}',
            fr'Cobertura: {coverages[i, j].round(2)}'))
        ax[i][j].text(0.1, 0.9, textstr, 
            transform=ax[i][j].transAxes,
            fontsize=8,
            verticalalignment='top',
            bbox=props,
            zorder=10001)
        for k in range(ne):
            ax[i][j].plot(xrange, xa_enkf[0, k, :, i, j], '0.5', label=legends[3] if k == 0 else None)

        if j == 0:
            ax[i][j].set_ylabel('x')
        else:
            ax[i][j].tick_params(axis='y', left=False)

        if i == 2:
            ax[i][j].set_xlabel('t')
        else:
            ax[i][j].tick_params(axis='x', bottom=False)

        if j == 2:
            ax[i][j].text(1.08, 0.5, fr'$Q {signs[i]} Q_t$',
                    horizontalalignment='center',
                    verticalalignment='center',
                    rotation='vertical',
                    transform=ax[i][j].transAxes)

        if i == 0:
            ax[i][j].text(0.5, 1.02, fr'$R {signs[j]} R_t$',
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    transform=ax[i][j].transAxes)

fig.legend(loc='upper center', framealpha=1, ncol=2, bbox_to_anchor=(0.5, 1.01))

plt.show()
