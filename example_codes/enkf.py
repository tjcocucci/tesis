import numpy as np
import scipy.stats as st
from tqdm import tqdm

def enkf_analysis(xf, y, H, R, infl = 1.0):
    nx, ne = xf.shape

    S = np.cov(xf) * infl

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

    for k in tqdm(np.arange(1, ncy)):
        # Evolve particles forward
        for j in np.arange(ne):
            xf[:, j, k] = f(k-1)(xa[:, j, k-1])

        # Compute analysis ensemble
        xa[:, :, k] = enkf_analysis(xf[:, :, k], y[:, k], H(k), R(k), infl=infl)

    return xf, xa
