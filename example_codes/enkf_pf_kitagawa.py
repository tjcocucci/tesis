from builtins import range
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from enkf import enkf, simulate_truth_obs
from bootstrap_pf import bootstrap_PF
from kitagawa import kitagawa_model, nx

np.random.seed(200)

ne = 20
ncy = 100
m = 1
nt = 1

Q = np.eye(nx) * 0.1
Qt = lambda t: Q
H = np.eye(nx)
# H = np.array([[1, 0, 0]])
Ht = lambda t: H
R = np.eye(m) * 2
Rt = lambda t: R

def likelihood(x, y):
    return st.multivariate_normal(mean=y, cov=R).pdf(H @ x)

def f(t, x):
    xtemp = np.copy(x)
    for i in range(nt):
        xtemp = kitagawa_model(t+i, xtemp) 
    return xtemp

x0 = np.ones(nx) * 0.05
for i in range(100):
    x0 = f(i, x0)

x0_ens = st.multivariate_normal(mean=x0, cov=Q*10).rvs(ne)[..., np.newaxis].T
print(x0_ens.shape)
print(x0.shape)
xt, y = simulate_truth_obs(ncy, x0, f, H, R)

xf_enkf, xa_enkf = enkf(y, x0_ens, f, Ht, Rt)

xf_pf, xa_pf, w = bootstrap_PF(x0_ens, f, likelihood, y)
# print(w)

fig, ax = plt.subplots(1, 2)
for i in range(ne):
    ax[0].plot(xa_enkf[0, i, :], '0.5')
ax[0].plot(xt[0, :], 'k--')
ax[0].plot(y[0, :], 'r.')
ax[0].plot(xa_enkf[0, :, :].mean(0), 'b')

for i in range(ne):
    ax[1].plot(xa_pf[0, i, :], '0.5')
ax[1].plot(xt[0, :], 'k--')
ax[1].plot(y[0, :], 'r.')
ax[1].plot(xa_pf[0, :, :].mean(0), 'b')

plt.show()
