import numpy as np
import pickle
import matplotlib.pyplot as plt

nx = 3
sig_q = 0.3
Q_true = sig_q * np.eye(nx)

with open('Q_hat.pkl', 'rb') as file:
    Q_hat = pickle.load(file)

Q_hat_EnKF, Q_hat_EnKFS, Q_hat_VMPF = (Q_hat[..., i] for i in range(3))

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True)

tab10 = plt.get_cmap('tab10')

reps = 20
for j in range(reps):
    diag_mean_EnKF = np.array([np.mean(np.diag(Q_hat_EnKF[..., i, j])) for i in range(20, Q_hat_EnKF.shape[2])])
    diag_mean_EnKFS = np.array([np.mean(np.diag(Q_hat_EnKFS[..., i, j])) for i in range(20, Q_hat_EnKFS.shape[2])])
    diag_mean_VMPF = np.array([np.mean(np.diag(Q_hat_VMPF[..., i, j])) for i in range(20, Q_hat_VMPF.shape[2])])
    

    l1 = ax1.plot(diag_mean_EnKF, c='0.5', label='Repeticiones' if j==0 else None)
    l2 = ax2.plot(diag_mean_EnKFS, c='0.5')
    l3 = ax3.plot(diag_mean_VMPF, c='0.5')


a = np.array([
        np.array([
            np.mean(np.diag(Q_hat_EnKF[..., i, j])) for i in range(20, Q_hat_EnKFS.shape[2])
]) for j in range(reps)])
ax1.plot(np.mean(a, axis=0), 'blue', label='Media de las repeticiones')

a = np.array([
        np.array([
            np.mean(np.diag(Q_hat_EnKFS[..., i, j])) for i in range(20, Q_hat_EnKFS.shape[2])
]) for j in range(reps)])
ax2.plot(np.mean(a, axis=0), 'blue')

a = np.array([
        np.array([
            np.mean(np.diag(Q_hat_VMPF[..., i, j])) for i in range(20, Q_hat_EnKFS.shape[2])
]) for j in range(reps)])
ax3.plot(np.mean(a, axis=0), 'blue')

l4 = ax1.axhline(y=np.mean(np.diag(Q_true)), label='Valor real', c='k', ls='--')
ax2.axhline(y=np.mean(np.diag(Q_true)), c='k', ls='--')
ax3.axhline(y=np.mean(np.diag(Q_true)), c='k', ls='--')

fig.legend(framealpha=1)
ax3.set_xlabel('Ciclo de asimilación')
ax2.set_ylabel('Media de la diagonal de ' + r'$\mathbf{Q}$')
# ax1.set_title('Convergence results')
# plt.tight_layout()
fig.subplots_adjust(hspace=0.1)

plt.show()