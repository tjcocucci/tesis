from statistics import mean
import numpy as np
import scipy.stats as st
from tqdm import tqdm
from enkf import enkf, enks

def rmse_ens(x, xt):
    return np.sqrt(np.mean((x - xt[:, np.newaxis, :])**2))

def loglikelihood(xf, y, H, R):
    nx, Nens, ncy = xf.shape
    
    llik = 0
    for k in np.arange(ncy):
        A = H(k) @ np.cov(xf[..., k]) @ H(k).T + R
        Ainv = np.linalg.inv(A)
        diff = y[:, k] - (H(k) @ np.mean(xf[..., k], axis=1))
        llik += diff.T @ Ainv @ diff + np.log(np.linalg.det(A))
    llik *= -0.5
    return llik

def montecarlo_loglikelihood(xf, y, H, R):
    nx, ne, ncy = xf.shape
    m = y.shape[0]
    R_true = np.eye(m) * 0.5
    llik_values = np.zeros(ne)
    for j in range(ne):
        values = [st.multivariate_normal(
            mean=(H(t) @ xf[:, j, t]), cov=R).logpdf(y[:, t])
        for t in range(ncy)]
        llik_values[j] = np.nansum(values)
    return np.mean(llik_values)

def _maximize(f_no_noise, H, y, xs, structQ, Q0):

    nx, Nens, ncy = xs.shape
    m = y.shape[0]
    
    # update Q
    sumSig = np.zeros((nx, nx))
    for t in np.arange(ncy-1):
        sigma = np.zeros((nx, nx))
        for k in np.arange(Nens):
            dif = xs[:, k, t+1] - f_no_noise(t, xs[:, k, t])
            dif = np.reshape(dif, (nx,1))
            sigma += dif @ dif.T
        sigma /= float(Nens)
        sumSig += sigma

    if structQ == 'full':
        Q = sumSig / (ncy-1)
    elif structQ == 'diag':
        Q = np.diag(np.diag(sumSig)) / ncy
    elif structQ == 'const':
        alpha = np.trace(np.linalg.inv(Q0).dot(sumSig)) / float(ncy*nx)
        Q = alpha * Q0 

    # update R
    sumaomega = np.zeros((m, m))
    for k in np.arange(1, ncy):
        omega = np.zeros((m, m))
        for i in np.arange(Nens):
            dif = y[:, k] - H(k) @ xs[:,i,k]
            dif = np.reshape(dif, (m,1))
            omega += dif @ dif.T
        sumaomega += omega/float(Nens)
    
    R = sumaomega/float(ncy)

    return Q, R

def em_enks(x0, Qinit, Rinit, H, f_no_noise, f_from_Q, y, nit, xt,
            estQ=True, estR=True, structQ='full', Q0=None):
    
    if structQ == 'const' and Q0 == None:
        raise TypeError('Must specify Q0 template')

    loglik = np.zeros(nit)
    rmse_f = np.zeros(nit)
    rmse_a = np.zeros(nit)
    rmse_s = np.zeros(nit)

    Q_hat  = np.zeros(np.r_[Qinit.shape, nit+1])
    R_hat  = np.zeros(np.r_[Rinit.shape, nit+1])
    
    Q_hat[:, :, 0] = Qinit
    R_hat[:, :, 0] = Rinit
    print(Q_hat[:, :, 0])
    print(R_hat[:, :, 0])
    for i in tqdm(np.arange(nit), desc='EM'):

        f = f_from_Q(Q_hat[:,:,i])
        Rt = lambda t: R_hat[:,:,i]
        xf, xa = enkf(y, x0, f, H, Rt, infl=1.0)
        xs = enks(xf, xa)
        x0 = np.copy(xs[..., 0])
        loglik[i] = montecarlo_loglikelihood(xf, y, H, R_hat[..., i])
        
        rmse_f[i] = rmse_ens(xf, xt)
        rmse_a[i] = rmse_ens(xa, xt)
        rmse_s[i] = rmse_ens(xs, xt)

        # M-step
        Q_new, R_new = _maximize(f_no_noise, H, y, xs, structQ, Q0)
        
        if estQ:
            Q = Q_new
        else:
            Q = Q_hat[:, :, i]
        if estR:
            R = R_new
        else:
            R = R_hat[:, :, i]

        Q_hat[:, :, i+1] = Q
        R_hat[:, :, i+1] = R
     
    res = {
            'xs': xs,
            'xa': xa,
            'xf': xf,
            'Q_hat': Q_hat,
            'R_hat': R_hat,
            'loglikelihood': loglik,
            'RMSE_f': rmse_f,
            'RMSE_a': rmse_a,
            'RMSE_s': rmse_s
    }

    return res
