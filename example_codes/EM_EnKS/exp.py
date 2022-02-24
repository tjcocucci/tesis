import sys
import os
path = os.getcwd()
path = os.path.abspath(path+'/..')
if not path in sys.path:
    sys.path.insert(0, path)
import numpy as np
import scipy.stats as st
from enkf import enkf, simulate_truth_obs
from em_enkf import em_enks
from lor63 import lor63_step, nx

np.random.seed(200)

dt = 0.1
omega_sq = 2.0
nx = 2
m = 2

M = np.array([
    [ -(dt**2)*omega_sq + 1.0, dt ],
    [ -dt*omega_sq, 1.0 ]
    ])

ne = 20
ncy = 1000
# dt = 0.01
nt = 1
# m = 3

sQ = 0.05
sR = 0.5
Q_true = np.eye(nx) * sQ
H = np.eye(nx)
Ht = lambda t: H
# H = np.array([[1, 0, 0]])
R_true = np.eye(m) * sR
Rt = lambda t: R_true

def f_no_noise(t, x):
    x_temp = np.copy(x)
    for i in range(nt):
        x_temp = M @ x_temp
        # x_temp = lor63_step(t, x_temp, dt)
    return x_temp

def f_from_Q(Q):
    def f_est(t, x):
        x_temp = f_no_noise(t, x)
        return x_temp + st.multivariate_normal(cov=Q).rvs()
    return f_est

f_true = f_from_Q(Q_true)

x0 = np.ones(nx) * 0.05
for i in range(100):
    x0 = f_true(i, x0)

x0_ens = st.multivariate_normal(mean=x0, cov=Q_true*10).rvs(ne).T
Qinit = np.random.normal(0, sQ, size=(nx, nx))
Qinit = Qinit.T @ Qinit * sQ * 2 + 5 * Q_true
Rinit = np.random.normal(0, sR, size=(m, m))
Rinit = Rinit.T @ Rinit * sR * 2 + 5 * R_true
nit = 500

xt, y = simulate_truth_obs(ncy, x0, f_true, H, R_true)

res = em_enks(x0_ens, Qinit, Rinit, Ht, f_no_noise, f_from_Q, y, nit, xt,
            estQ=True, estR=True, structQ='full', Q0=None)

np.save('xs.npy', res['xs'])
np.save('xa.npy', res['xa'])
np.save('xf.npy', res['xf'])
np.save('Q_hat.npy', res['Q_hat'])
np.save('R_hat.npy', res['R_hat'])
np.save('loglikelihood.npy', res['loglikelihood'])
np.save('RMSE_f.npy', res['RMSE_f'])
np.save('RMSE_a.npy', res['RMSE_a'])
np.save('RMSE_s.npy', res['RMSE_s'])
np.save('Q_true.npy', Q_true)
np.save('R_true.npy', R_true)
