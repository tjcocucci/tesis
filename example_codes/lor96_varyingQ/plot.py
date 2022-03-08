import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats
import scipy.linalg
import pickle

with open('Q_hat.pkl', 'rb') as file:
    Q_hat = pickle.load(file)

Q_hat_EnKF, Q_hat_EnKFS, Q_hat_VMPF = (Q_hat[..., i] for i in range(3))

ncy = Q_hat_EnKF.shape[-1]
nx = Q_hat_EnKF.shape[0]

def logistic_growth(t, k, r, p0, t0):
    return k / (1 + ((k-p0)/p0)* np.exp(-r*(t-t0))) + p0
k = 2
p0 = 1
t0 = (ncy-1)/2.5
r = 1/600
lg = lambda t: logistic_growth(t, k, r, p0, t0)

def Q_true(k: int):
    return scipy.linalg.toeplitz([1, 0.3] + [0]*(nx-3) + [0.3])*lg(k) * 0.3

def extended_diagonal_mean(A, idx=0):
    n = A.shape[0]
    return np.mean([A[i%n, (i+idx)%n] for i in range(n)])

def truncated_lower_triangle_mean(A, idx=1, height=1):
    elements = [list(A[i+idx:i+idx+height, i]) for i in range(A.shape[0]-idx)]
    return np.mean(np.concatenate(elements))

diag_mean_EnKF = np.array([extended_diagonal_mean(Q_hat_EnKF[..., i]) 
                            for i in range(ncy)])
first_subdiag_mean_EnKF = np.array([ extended_diagonal_mean(Q_hat_EnKF[..., i], idx=1) 
                                     for i in range(ncy) ])
lower_triangle_mean_EnKF = np.array([ truncated_lower_triangle_mean(Q_hat_EnKF[..., i], idx=2, height=nx-3) 
                                      for i in range(ncy) ])

diag_mean_EnKFS = np.array([extended_diagonal_mean(Q_hat_EnKFS[..., i]) 
                            for i in range(ncy)])
first_subdiag_mean_EnKFS = np.array([ extended_diagonal_mean(Q_hat_EnKFS[..., i], idx=1) 
                                     for i in range(ncy) ])
lower_triangle_mean_EnKFS = np.array([ truncated_lower_triangle_mean(Q_hat_EnKFS[..., i], idx=2, height=nx-3) 
                                      for i in range(ncy) ])

diag_mean_VMPF = np.array([extended_diagonal_mean(Q_hat_VMPF[..., i]) 
                            for i in range(ncy)])
first_subdiag_mean_VMPF = np.array([ extended_diagonal_mean(Q_hat_VMPF[..., i], idx=1) 
                                     for i in range(ncy) ])
lower_triangle_mean_VMPF = np.array([ truncated_lower_triangle_mean(Q_hat_VMPF[..., i], idx=2, height=nx-3) 
                                      for i in range(ncy) ])


cmap = cm.get_cmap('Blues', 5)
fig, axs = plt.subplots(3, 1, sharex=True)
plt.subplots_adjust(hspace=0.1)
colors = [cmap(1), cmap(2), cmap(3), cmap(4)]

# colors = ['red', 'orange', 'blue', 'green']

ax = axs[0]
l1 = ax.plot(diag_mean_EnKFS, c=colors[0], label='OSS-EnKF')
ax.plot(diag_mean_EnKF, c=colors[1], ls='-', label='IS-EnKF')
ax.plot(diag_mean_VMPF, c=colors[2], ls='-', label='IS-VMPF')
ax.plot([extended_diagonal_mean(Q_true(k)) for k in range(ncy-1)], c=colors[3], ls='--', label='Valor real')


ax = axs[1]
l2 = ax.plot(first_subdiag_mean_EnKFS, c=colors[0])
ax.plot(first_subdiag_mean_VMPF, c=colors[1], ls='-')
ax.plot(first_subdiag_mean_EnKF, c=colors[2], ls='-')
ax.plot([extended_diagonal_mean(Q_true(k), idx=1) for k in range(ncy-1)], c=colors[3], ls='--')

ax = axs[2]
l4 = ax.plot(lower_triangle_mean_EnKFS, c=colors[0])
ax.plot(lower_triangle_mean_VMPF, c=colors[1], ls='-')
ax.plot(lower_triangle_mean_EnKF, c=colors[2], ls='-')
ax.plot([truncated_lower_triangle_mean(Q_true(k), idx=2, height=nx-3) for k in range(ncy-1)], c=colors[3], ls='--')


axs[2].set_xlabel('Ciclo de asimilación')
axs[1].set_ylabel('Covarianza media de\n variables vecinas')
axs[0].set_ylabel('Varianza media')
axs[2].set_ylabel('Covarianza media de\n variables no-vecinas')
fig.legend(framealpha=1, ncol=2)

plt.show()