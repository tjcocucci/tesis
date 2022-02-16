import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from enkf import enkf, simulate_truth_obs
from bootstrap_pf import bootstrap_PF
from vmpf import vmpf
from lor63 import lor63_step, nx

np.random.seed(200)

ne = 2
ncy = 200
dt = 0.01
nt = 5
m = 3

Q = np.eye(nx) * 1
Qt = lambda t: Q
H = np.eye(nx)
# H = np.array([[1, 0, 0]])
Ht = lambda t: H
R = np.eye(m) * 1
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

x0_ens = st.multivariate_normal(mean=x0, cov=Q*10).rvs(ne).T

xt, y = simulate_truth_obs(ncy, x0, f, H, R)

xf_enkf, xa_enkf = enkf(y, x0_ens, f, Ht, Rt)

xf_pf, xa_pf, w = bootstrap_PF(x0_ens, f, likelihood, y)

nit = 50
xf_vmpf, xa_vmpf = vmpf(nit, y, x0_ens, f, f_no_noise, Q, R, H)

fig, ax = plt.subplots(3, 1, sharex=True)
plt.subplots_adjust(hspace=0.1)

for j, ens in enumerate([xa_enkf, xa_pf, xa_vmpf]):
    if j == 2:
        legends = ['Valor real', 'Observaciones', 'Media del ensamble', 'Partículas']
        ax[j].set_xlabel('t')
    else:
        ax[j].tick_params(axis='x', bottom=False)
        legends = [None]*4
    ax[j].plot(xt[1, :], 'k--', label=legends[0], zorder=1000)
    ax[j].plot(y[1, :], 'r.', label=legends[1], zorder=1000)
    ax[j].plot(ens[1, :, :].mean(0), 'b', label=legends[2], zorder=1000)
    for i in range(ne):
        ax[j].plot(ens[1, i, :], '0.5', label=legends[3] if i == 0 else None)
        
    ax[j].set_ylabel('x')

fig.legend(framealpha=1, ncol=2)



plt.show()
