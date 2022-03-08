from turtle import color
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FuncFormatter
import pickle

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1fk' % (x * 1e-3)

def frob_norm(A):
    return np.sqrt(np.trace(A @ A.T))

def frob_dist(A, B):
    return frob_norm(A - B)

def subdiagonal_mean(A, idx=1):
    return np.mean([ A[i, i+idx] for i in range(A.shape[0]-idx) ])

def lower_triangle_mean(A, idx=1):
    elements = [list(A[i, i+idx:]) for i in range(A.shape[0]-idx)]
    return np.mean(np.concatenate(elements))

with open('Q_hat.pkl', 'rb') as file:
    Q_hat = pickle.load(file)
with open('Q_hat_EM_EnKS.pkl', 'rb') as file:
    Q_hat_EM = pickle.load(file)

with open('rmse_llik_EM_EnKS.pkl', 'rb') as file:
    rmse_llik_EM_EnKS = pickle.load(file)

rmse_EM_EnKS = rmse_llik_EM_EnKS[0]
llik_EM_EnKS = rmse_llik_EM_EnKS[1]

with open('enkf_llik_total.pkl', 'rb') as file:
    enkf_llik_total = pickle.load(file)
with open('enkfs_llik_total.pkl', 'rb') as file:
    enkfs_llik_total = pickle.load(file)
with open('vmpf_llik_total.pkl', 'rb') as file:
    vmpf_llik_total = pickle.load(file)

with open('enkf_RMSE_total.pkl', 'rb') as file:
    enkf_RMSE_total = pickle.load(file)
with open('enkfs_RMSE_total.pkl', 'rb') as file:
    enkfs_RMSE_total = pickle.load(file)
with open('vmpf_RMSE_total.pkl', 'rb') as file:
    vmpf_RMSE_total = pickle.load(file)



Q_hat_EnKF, Q_hat_EnKFS, Q_hat_VMPF = (Q_hat[..., i] for i in range(3))

nx = Q_hat_EnKF.shape[0]

Q_true = scipy.linalg.toeplitz([1, 0.3] + [0]*(nx-3) + [0.3]) * 0.3


cmap = cm.get_cmap('Blues', 5)
fig, axs = plt.subplots(3, 1, sharex=True)
plt.subplots_adjust(hspace=0.1)
colors = [cmap(1), cmap(2), cmap(3), cmap(4)]
c = colors[2]

axs[0].plot([frob_dist(Q_hat_EnKFS[..., i], Q_true)
                   for i in range(20, Q_hat_EnKFS.shape[-1])],
       color=colors[0])
axs[0].plot([frob_dist(Q_hat_EnKF[..., i], Q_true)
                   for i in range(20, Q_hat_EnKF.shape[-1])],
       color=colors[1])
axs[0].plot([frob_dist(Q_hat_VMPF[..., i], Q_true)
                   for i in range(20, Q_hat_VMPF.shape[-1])],
       color=colors[2])
axs[0].axhline(y=frob_dist(Q_hat_EM[..., -1], Q_true),
          color=colors[3], ls='--')

xrange = range(0, Q_hat_VMPF.shape[-1]-20)[::20]

formatter = FuncFormatter(thousands)

axs[1].plot(xrange, enkfs_llik_total[1:], ls='-', color=colors[0])
axs[1].plot(xrange, enkf_llik_total[1:], ls='-', color=colors[1])
axs[1].plot(xrange, vmpf_llik_total[1:], ls='-', color=colors[2])
axs[1].axhline(y=llik_EM_EnKS, ls='--', color=colors[3])

axs[2].plot(xrange, enkfs_RMSE_total[1:], ls='-', label='OSS-EnKF', color=colors[0])
axs[2].plot(xrange, enkf_RMSE_total[1:], ls='-', label='IS-EnKF', color=colors[1])
axs[2].plot(xrange, vmpf_RMSE_total[1:], ls='-', label='IS-VMPF', color=colors[2])
axs[2].axhline(y=rmse_EM_EnKS, ls='--', label='Batch EM', color=colors[3])

axs[1].yaxis.set_major_formatter(formatter)

axs[0].set_ylabel('Distancia Frobenious\n' + r'al valor real de $\mathbf{Q}$')
axs[1].set_ylabel('Log-verosimilitud')
axs[2].set_ylabel('RMSE de variables\nde estado')

axs[2].set_xlabel('Ciclo de asimilación')

fig.legend(framealpha=1, ncol=2)
plt.show()




fig, ax = plt.subplots()

for i in range(nx):
    ax.plot(Q_hat_EnKFS[i, i, 20:], color=colors[1])
ax.axhline(y=subdiagonal_mean(Q_true, idx=0), c=colors[1], ls='--')

for i in range(2, nx):
    for j in range(i-2):
        ax.plot(Q_hat_EnKFS[i, j, 20:], color=colors[3])
ax.axhline(y=lower_triangle_mean(Q_true, idx=2), c=colors[3], ls='--')

for i in range(nx-1):
    ax.plot(Q_hat_EnKFS[i, i+1, 20:], color=colors[2])
ax.axhline(y=subdiagonal_mean(Q_true, idx=1), c=colors[2], ls='--')

ax.plot([], color=colors[1], label='Diagonal')
ax.plot([], color=colors[2], label='Subdiagonal')
ax.plot([], color=colors[3], label='Triángulo inferior')

ax.set_ylabel('Estimaciones ' + r'por entrada de $\mathbf{Q}$')
ax.set_xlabel('Ciclo de asimilación')


fig.legend(ncol=3, framealpha=1)

# axs[0].text(-0.11, 1.05, '(a)', transform=axs[0].transAxes, style='italic')
# axs[1].text(-0.11, 0.98, '(b)', transform=axs[1].transAxes, style='italic')

plt.show()