import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from enkf import enkf, simulate_truth_obs
from bootstrap_pf import bootstrap_PF
from vmpf import vmpf
from lor63 import lor63_step, nx

np.random.seed(200)

ne = 100
ncy = 200
dt = 0.01
nt = 1
m = 3

sQ = 1
sR = 1
Q = np.eye(nx) * sQ
Qt = lambda t: Q
H = np.eye(nx)
Ht = lambda t: H
R = np.eye(m) * sR
Rt = lambda t: R

def covered_test(x, P, xt, alpha=0.95):
    nx = x.shape
    s = (xt - x).T @ np.linalg.inv(P) @ (xt - x)
    return s < st.chi2(df=nx).ppf(alpha)

def time_series_coverage(x, P, xt, alpha=0.95):
    nx, ncy = x.shape
    covered = np.array([covered_test(x[:, t], P[..., t], xt[:, t], alpha=alpha)
                        for t in range(ncy)])
    return np.sum(covered) / float(ncy)

def ensemble_time_series_coverage(x, xt, alpha=0.95):
    nx, ne, ncy = x.shape
    means = np.array([np.mean(x[..., t], axis=1) for t in range(ncy)]).T
    covariances = np.array([np.cov(x[..., t]) for t in range(ncy)]).T
    return time_series_coverage(means, covariances, xt, alpha=alpha)

def rmse_ens(x, xt):
    return np.sqrt(np.mean((x - xt[:, np.newaxis, :])**2))

def f(t, x, Q=Q):
    x_temp = np.copy(x)
    for i in range(nt):
        x_temp = lor63_step(t, x_temp, dt)
    return x_temp + st.multivariate_normal(cov=Q).rvs()

x0 = np.ones(nx) * 0.05
for i in range(100):
    x0 = f(i, x0)

x0_ens = st.multivariate_normal(mean=x0, cov=Q*10).rvs(ne).T

xt, y = simulate_truth_obs(ncy, x0, f, H, R)


sQest = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.9]) * sQ
sRest = np.array([0.1, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75, 1.9]) * sR

sQest = np.linspace(0.1, 2.5, 20) * sQ
sRest = np.linspace(0.1, 2.5, 20) * sR

rmses = np.empty((sQest.shape[0], sRest.shape[0]))
coverages = np.empty((sQest.shape[0], sRest.shape[0]))

for i, q in enumerate(sQest):
    for j, r in enumerate(sRest):
        np.random.seed(400)
        fQ = lambda t, x: f(t, x, Q=Q*q)
        Rtest = lambda t: R*r
        xf_enkf, xa_enkf = enkf(y, x0_ens, fQ, Ht, Rtest)

        rmse = rmses[i, j] = rmse_ens(xa_enkf[..., 50:], xt[..., 50:])
        cov = coverages[i, j] = ensemble_time_series_coverage(xa_enkf[..., 50:], xt[..., 50:])
        print(q, r)
        print(rmse, cov)

np.save('sQest.npy', sQest)
np.save('sRest.npy', sRest)
np.save('rmses.npy', rmses)
np.save('coverages.npy', coverages)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(rmses)
ax[1].imshow(coverages)
plt.show()
