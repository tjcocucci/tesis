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
from lor63 import lor63_step, nx

np.random.seed(200)

ne = 20
ncy = 500
dt = 0.01
nt = 5
m = 3

Q = np.eye(nx) * 1
Q_est = np.eye(nx+1) * 1
Q_est[-1, -1] = 0.2
Qt = lambda t: Q
Qt_est = lambda t: Q_est
H = np.eye(nx)
# H = np.array([[1, 0, 0]])
Ht = lambda t: H
H_est = np.hstack([H, np.zeros(m)[:, np.newaxis]])
Ht_est = lambda t: H_est
R = np.eye(m) * 1
Rt = lambda t: R

true_rhos = np.sin(2*np.pi * np.arange(ncy)/(ncy))*5 + 28

def likelihood(x, y):
    return st.multivariate_normal(mean=y, cov=R).pdf(H @ x)

def f(t, x):
    x_temp = np.copy(x)
    for i in range(nt):
        x_temp = lor63_step(t, x_temp, dt, rho=true_rhos[t])
    return x_temp + st.multivariate_normal(cov=Q).rvs()

def f_est(t, x):
    x_temp = np.copy(x)
    for i in range(nt):
        x_temp[:nx] = lor63_step(t, x_temp[:nx], dt, rho=x[-1])
    return x_temp + st.multivariate_normal(cov=Q_est).rvs()

x0 = np.ones(nx) * 0.05
for i in range(100):
    x0 = f(i, x0)

x0_aug = np.zeros(nx+1)
x0_aug[:nx] = x0
x0_aug[-1] = 20
x0_ens = st.multivariate_normal(mean=x0_aug, cov=Q_est*100).rvs(ne).T

xt, y = simulate_truth_obs(ncy, x0, f, H, R)

xf_enkf, xa_enkf = enkf(y, x0_ens, f_est, Ht_est, Rt)

np.save('xa.npy', xa_enkf)
np.save('true_rhos.npy', true_rhos)
np.save('y.npy', y)
np.save('xt.npy', xt)
