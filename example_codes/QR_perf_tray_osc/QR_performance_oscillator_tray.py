import sys
import os
path = os.getcwd()
path = os.path.abspath(path+'/..')
if not path in sys.path:
    sys.path.insert(0, path)
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from enkf import enkf, simulate_truth_obs

np.random.seed(200)

omega_sq = 2.0
dt = 0.01

M = np.array([
    [ -(dt**2)*omega_sq + 1.0, dt ],
    [ -dt*omega_sq, 1.0 ]
])

nx = 2
ne = 50
ncy = 100
nt = 10
m = 2

sQ = 0.1
sR = 5
Q_base = np.eye(nx)
Q = Q_base * sQ
Qt = lambda t: Q
H = np.eye(nx)
Ht = lambda t: H
R_base = np.eye(nx)
R = R_base * sR
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
        x_temp = M @ x_temp
    return x_temp + st.multivariate_normal(cov=Q).rvs()

x0 = np.ones(nx) * 0.05
for i in range(100):
    x0 = f(i, x0)

x0_ens = st.multivariate_normal(mean=x0, cov=Q*10).rvs(ne).T

xt, y = simulate_truth_obs(ncy, x0, f, H, R)


sQest = np.array([0.1, 1, 2]) * sQ
sRest = np.array([0.1, 1, 2]) * sR

# sQest = np.linspace(0.1, 2.5, 20) * sQ
# sRest = np.linspace(0.1, 2.5, 20) * sR

rmses = np.empty((sQest.shape[0], sRest.shape[0]))
coverages = np.empty((sQest.shape[0], sRest.shape[0]))

xa_enkf = np.zeros((nx, ne, ncy, len(sQest), len(sRest)))

for i, q in enumerate(sQest):
    for j, r in enumerate(sRest):
        np.random.seed(400)
        fQ = lambda t, x: f(t, x, Q=Q_base*q)
        Rtest = lambda t: R_base*r
        xf_enkf, xa_enkf[..., i, j] = enkf(y, x0_ens, fQ, Ht, Rtest)

        rmse = rmses[i, j] = rmse_ens(xa_enkf[..., 10:, i, j], xt[..., 10:])
        cov = coverages[i, j] = ensemble_time_series_coverage(xa_enkf[..., 10:, i, j], xt[..., 10:])
        print(q, r)
        print(rmse, cov)

np.save('xt_osc.npy', xt)
np.save('y.npy', y)
np.save('xa_enkf_osc.npy', xa_enkf)
np.save('sQest_tray_osc.npy', sQest)
np.save('sRest_tray_osc.npy', sRest)
np.save('rmses_tray_osc.npy', rmses)
np.save('coverages_tray_osc.npy', coverages)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(rmses)
ax[1].imshow(coverages)
plt.show()
