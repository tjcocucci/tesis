import numpy as np
import scipy.stats as st
from tqdm import tqdm

def crossed_cov(X,Y):
    
    n, m = X.shape
    centeredX = X - np.vstack(np.mean(X, axis=1))
    centeredY = Y - np.vstack(np.mean(Y, axis=1))
    covXY = centeredX @ centeredY.T / (m - 1.)

    return covXY

def enkf_analysis(xf, y, H, R, infl = 1.0):
    nx, ne = xf.shape

    S = np.cov(xf).reshape((nx, nx)) * infl

    SHT = S @ H.T
    K = SHT @ np.linalg.pinv(H @ SHT + R)
    a = st.multivariate_normal(cov=R).rvs(ne)

    D = np.vstack(y) + a.T
    innov = D - H @ xf

    xa = xf + K @ innov

    return xa

def enkf(y, x0, f, H, R, infl=1.0):

    ncy = y.shape[-1]
    nx, ne = x0.shape

    xf = np.zeros((nx, ne, ncy))
    xa = np.zeros((nx, ne, ncy))

    # Initialization
    xf[:, :, 0] = x0
    xa[:, :, 0] = xf[:, :, 0]

    for k in tqdm(np.arange(1, ncy), leave=False, desc=f'EnKF{ne}'):
        # Evolve particles forward
        for j in np.arange(ne):
            xf[:, j, k] = f(k-1, xa[:, j, k-1])

        # Compute analysis ensemble
        xa[:, :, k] = enkf_analysis(xf[:, :, k], y[:, k], H(k), R(k), infl=infl)

    return xf, xa

def enks(xf, xa):

    nx, nens, ncy = xa.shape

    xs = np.zeros((nx, nens, ncy))
    xs[:, :, -1]=xa[:, :, -1]

    for k in np.arange(ncy-2, -1, -1):

        Paf = crossed_cov(xa[:, :, k], xf[:, :, k+1])
        Pff = np.cov(xf[:, :, k+1])

        if nx > 1:
          invPf = np.linalg.pinv(Pff)
        else:
          invPf = 1. / Pff

        K = np.dot(Paf, invPf)

        xs[:,:,k] = xa[:, :, k] + K.dot(xs[:, :, k+1] - xf[:, :, k+1])

    return xs

def simulate_truth_obs(ncy, x0, f, H, R):

    m, nx = H.shape
    xt = np.zeros((nx, ncy))
    y = np.zeros((m, ncy))
    xt[:, 0] = x0[:]
    y[:, 0] = np.nan

    for k in range(1, ncy):
        xt[:, k] = f(k-1, xt[:, k-1])
        y[:, k] = H @ xt[:, k] + st.multivariate_normal(cov=R).rvs()
    
    return xt, y
