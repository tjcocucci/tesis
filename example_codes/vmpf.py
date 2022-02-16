import numpy as np
from tqdm import tqdm

def vmpf_step(xf, y, nit, grad_KL,
    learning_rate=0.03, KL_grad_mod_thresh=0.05, rho=0.9, eps=1e-6):

    nx, ne = xf.shape
    x = np.copy(xf)
    KL_grad_mod = np.inf
    it = 0
    cum_grad = np.zeros((nx, ne))
    while it <= nit and KL_grad_mod >= KL_grad_mod_thresh:

        KL_grad_mod = 0
        grad_KL_values = grad_KL(x)
        for j in range(ne):
            grad = grad_KL_values[:, j]
            cum_grad[:, j] = rho * cum_grad[:, j] + (1 - rho) * grad**2
            KL_grad_mod = KL_grad_mod + np.sqrt(grad @ grad)
            x[:, j] = x[:, j] + learning_rate * grad / (eps + np.sqrt(cum_grad[:, j]))

        KL_grad_mod = KL_grad_mod / ne
        it += 1

    return x

def vmpf(nit, y, X0_sample, transition_sample, f, Q, R, H, bandwidth=2, **kwargs):
    ntimes = y.shape[-1]
    nx, ne = X0_sample.shape
    
    xf = np.zeros((nx, ne, ntimes))
    xa = np.zeros((nx, ne, ntimes))
    xf[..., 0] = X0_sample
    xa[..., 0] = X0_sample


    Qinv = np.linalg.pinv(Q)
    Rinv = np.linalg.pinv(R)
    Ainv = (1/bandwidth) * Qinv


    for t in tqdm(range(1, ntimes)):

        Mx = np.zeros((nx, ne))

        # Forecast
        for j in range(ne):
            xf[:, j, t] = transition_sample(t-1, xa[:, j, t-1])
            Mx[:, j] = f(t, xa[:, j, t-1])

        def grad_KL(x):
            gradlogp_values = gradlogp(x, y[:, t], Mx, Qinv, H, Rinv)
            kernel_values = np.array([
                [kernel(x[:, i], x[:, j], Ainv) for i in range(ne)]
            for j in range(ne)])            
            kernel_grad = np.array([
                [-0.5 * Ainv @ (x[:, i] - x[:, j]) for i in range(ne)]
            for j in range(ne)])
            kernel_grad = kernel_values[..., np.newaxis] * kernel_grad

            return (gradlogp_values @ kernel_values + kernel_grad.sum(1).T) / ne

        # Analysis
        xa[..., t] = vmpf_step(xf[..., t], y, nit, grad_KL, **kwargs)
    
    return xf, xa

def kernel(x, y, Ainv):
    return np.exp(-0.5 * (x - y).T @ Ainv @ (x - y))

def gradlogp(x, y, Mx, Qinv, H, Rinv):
    psi_values = psi(x, Mx, Qinv)
    return H.T @ Rinv @ (y[:, np.newaxis] - H @ x) - Qinv @ (x - (Mx @ psi_values)/psi_values.sum(1))

def psi(x, Mx, Qinv):
    return np.exp(-0.5 * (x - Mx).T @ Qinv @ (x - Mx))
