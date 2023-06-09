import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from enkf import enkf, simulate_truth_obs
from bootstrap_pf import bootstrap_PF
from vmpf import vmpf
from lor63 import lor63_step, nx
import pandas as pd

np.random.seed(200)

ncy = 200
dt = 0.01
nt = 5
m = 3

Q = np.eye(nx) * 0.5
Qt = lambda t: Q
H = np.eye(nx)
# H = np.array([[1, 0, 0]])
Ht = lambda t: H
R = np.eye(m) * 0.5
Rt = lambda t: R

def likelihood(x, y):
    return st.multivariate_normal(mean=y, cov=R).pdf(H @ x)

def f(t, x):
    x_temp = np.copy(x)
    for i in range(nt):
        x_temp = lor63_step(t, x_temp, dt)
    return x_temp + st.multivariate_normal(cov=Q).rvs()

def f_no_noise(t, x):
    x_temp = np.copy(x)
    for i in range(nt):
        x_temp = lor63_step(t, x_temp, dt)
    return x_temp


x0 = np.ones(nx) * 0.05
for i in range(100):
    x0 = f(i, x0)

xt, y = simulate_truth_obs(ncy, x0, f, H, R)

def rmse(x, xt):
    return np.sqrt(np.mean((x - xt[:, np.newaxis, :])**2))

def var(x):
    return np.mean((x - x.mean(1)[:, np.newaxis, :])**2)

nes = np.array([2, 5, 10, 20, 50])

rmses = np.zeros((3, len(nes)))
vars = np.zeros((3, len(nes)))

for i, ne in enumerate(nes):

    x0_ens = st.multivariate_normal(mean=x0, cov=Q*10).rvs(ne).T

    xf_enkf, xa_enkf = enkf(y, x0_ens, f, Ht, Rt)

    xf_pf, xa_pf, w = bootstrap_PF(x0_ens, f, likelihood, y)

    nit = 50
    xf_vmpf, xa_vmpf = vmpf(nit, y, x0_ens, f, f_no_noise, Q, R, H)

    rmses[0, i] = rmse(xa_enkf, xt)
    rmses[1, i] = rmse(xa_pf, xt)
    rmses[2, i] = rmse(xa_vmpf, xt)

    vars[0, i] = var(xa_enkf)
    vars[1, i] = var(xa_pf)
    vars[2, i] = var(xa_vmpf)

fig, ax = plt.subplots(3, 1, sharex=True)
for i in range(ne):
    ax[0].plot(xa_enkf[1, i, :], '0.5')
ax[0].plot(xt[1, :], 'k--')
ax[0].plot(y[1, :], 'r.')
ax[0].plot(xa_enkf[1, :, :].mean(0), 'b')

for i in range(ne):
    ax[1].plot(xa_pf[1, i, :], '0.5')
ax[1].plot(xt[1, :], 'k--')
ax[1].plot(y[1, :], 'r.')
ax[1].plot(xa_pf[1, :, :].mean(0), 'b')

for i in range(ne):
    ax[2].plot(xa_vmpf[1, i, :], '0.5')
ax[2].plot(xt[1, :], 'k--')
ax[2].plot(y[1, :], 'r.')
ax[2].plot(xa_vmpf[1, :, :].mean(0), 'b')

plt.show()

np.save('nes.npy', nes)
np.save('rmses.npy', rmses)
np.save('vars.npy', vars)
