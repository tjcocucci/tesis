import numpy as np
import matplotlib.pyplot as plt

last = 200
xs = np.load('xs.npy')[..., :last]
xa = np.load('xa.npy')[..., :last]
xf = np.load('xf.npy')[..., :last]
Q_hat = np.load('Q_hat.npy')[..., :last]
R_hat = np.load('R_hat.npy')[..., :last]
loglikelihood = np.load('loglikelihood.npy')[..., :last]
RMSE_f = np.load('RMSE_f.npy')[..., :last]
RMSE_a = np.load('RMSE_a.npy')[..., :last]
RMSE_s = np.load('RMSE_s.npy')[..., :last]
Q_true = np.load('Q_true.npy')
R_true = np.load('R_true.npy')

nx, ne, ncy = xa.shape
nit = loglikelihood.shape[0]

def matrix_rmse(x, xt):
    return np.sqrt(np.mean((x - xt[..., np.newaxis])**2, axis=(0, 1)))

print(np.array([np.diag(Q_hat[:, :, t]) for t in range(nit)]))
Q_hat_diags = np.array([np.mean(np.diag(Q_hat[:, :, t])) for t in range(nit)])
R_hat_diags = np.array([np.mean(np.diag(R_hat[:, :, t])) for t in range(nit)])

print(loglikelihood)

fig, ax = plt.subplots(2, 1, sharex=True)
plt.subplots_adjust(hspace=0.1)
# ax[0].tick_params(axis='x', bottom=False)

ax[0].plot(Q_hat_diags, color='blue', label='Q')
ax[0].plot(R_hat_diags, color='red', label='R')
ax[1].plot(matrix_rmse(Q_hat, Q_true), color='blue')
ax[1].plot(matrix_rmse(R_hat, R_true), color='red')
ax[0].axhline(y=0.5, color='black', ls='--')
ax[0].axhline(y=0.05, color='black', ls='--')
ax[1].axhline(y=0.0, color='black', ls='--')

ax[0].set_ylabel('Media de la diagonal')
ax[1].set_ylabel('RMSE parámetros')
ax[1].set_xlabel('Iteración de EM')

ax[0].set_ylim(0, 0.75)
ax[1].set_ylim(-0.045, 0.22)

fig.legend(ncol=2, framealpha=1, loc='upper center')
plt.show()

# fig, ax = plt.subplots()
# ax.plot(matrix_rmse(Q_hat, Q_true), color='blue')
# ax.plot(matrix_rmse(R_hat, R_true), color='blue')
# ax.axhline(y=0, color='black', ls='--')
# plt.show()

fig, ax = plt.subplots()
ax_twin = ax.twinx()
ax.plot(loglikelihood, color='blue', label='Log-verosimilitud')
ax_twin.plot(RMSE_s, color='red', label='RMSE')
ax.set_xlabel('Iteración de EM')
ax.set_ylabel('Log-verosimilitud')
ax_twin.set_ylabel('RMSE variables de estado')
fig.legend(ncol=2, framealpha=1, loc='upper center')
plt.show()
