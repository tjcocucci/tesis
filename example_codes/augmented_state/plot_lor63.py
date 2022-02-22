import matplotlib.pyplot as plt
import numpy as np

xa = np.load('xa.npy')
y = np.load('y.npy')
xt = np.load('xt.npy')
true_rhos = np.load('true_rhos.npy')

rho_true = 28
nx, ncy = xt.shape
ne = xa.shape[1]

idx0 = 0
idx_aug = -1
legends = ['Valor real', 'Observaciones', 'Media del ensamble', 'Partículas']

fig, ax = plt.subplots(1, 1, sharex=True)
plt.subplots_adjust(top=0.88,
bottom=0.11,
left=0.11,
right=0.9,
hspace=0.1,
wspace=0.2)

ax.plot(true_rhos, 'k--', label=legends[0], zorder=1000)
ax.set_ylabel(r'$\rho$')
ax.plot(xa[-1, :, :].mean(0), 'b', label=legends[2], zorder=1000)
for i in range(ne):
    ax.plot(xa[-1, i, :], '0.5', label=legends[3] if i == 0 else None)
ax.set_xlabel('t')


fig.legend(framealpha=1, ncol=2)

plt.show()
