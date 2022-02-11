import numpy as np
import scipy.stats as st

def KF_analysis(xf, Pf, y, H, R):
    S = H.dot(Pf).dot(H.T) + R
    K = Pf.dot(H.T).dot(np.linalg.inv(S))
    v = y - H.dot(xf)
    xa = xf + K.dot(v)
    Pa = Pf - K.dot(H).dot(Pf)
    return xa, Pa

def KF(x0, P0, y, M, Q, H, R):
    nx = x0.shape[0]
    m, ntimes = y.shape
    
    xf = np.zeros((nx, ntimes))
    xa = np.zeros((nx, ntimes))
    Pf = np.zeros((nx, nx, ntimes))
    Pa = np.zeros((nx, nx, ntimes))    
    
    xf[:, 0] = x0
    xa[:, 0] = x0
    Pf[..., 0] = P0
    Pa[..., 0] = P0    
    
    for i in range(1, ntimes):
        
        # Forecast
        xf[:, i] = M.dot(xa[:, i-1])
        Pf[..., i] = M.dot(Pa[..., i-1]).dot(M.T) + Q
        
        # Analysis
        if np.isnan(y[:, i]).any():
            xa[:, i], Pa[..., i] = xf[:, i], Pf[..., i]
        else:
            xa[:, i], Pa[..., i] = KF_analysis(xf[:, i], Pf[..., i], y[:, i], H, R)

    return xf, Pf, xa, Pa

def simulate_truth_obs(ntimes, x0, M, H, Q, R):

    m, nx = H.shape
    xt = np.zeros((nx, ntimes))
    y = np.zeros((m, ntimes))
    xt[:, 0] = x0
    y[:, 0] = np.nan

    for i in range(1, ntimes):
        xt[:, i] = M.dot(xt[:, i-1])
        xt[:, i] += st.multivariate_normal(cov=Q, allow_singular=True).rvs()  
        y[:, i] = H.dot(xt[:, i])
        y[:, i] += st.multivariate_normal(cov=R).rvs()
    
    return xt, y
