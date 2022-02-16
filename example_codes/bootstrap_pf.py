import numpy as np
from tqdm import tqdm

def bootstrap_PF_step(xf, w, y, observational_pdf, Neff_thresh=np.inf):
    nx, ne = xf.shape
    
    # Compute weights
    w_new = np.array([observational_pdf(xf[:, i], y)
                  for i in range(ne)])
    w_new = np.multiply(w/np.max(w), w_new)

    try:
        w_new /= np.sum(w_new)
    except:
        w_new = np.repeat(1/ne, ne)

    Neff = 1/np.sum(w_new**2)
    
    # Resample
    if Neff/ne < Neff_thresh:
        resample_indexes = np.random.choice(ne, replace=True, p=w_new, size=ne)
        xa = xf[:, resample_indexes]
        w_new = np.repeat(1./ne, ne)
    else:
        xa = np.copy(xf)
    return xa, w_new
    
def bootstrap_PF(X0_sample, transition_sample, observational_pdf, y,
                 Neff_thresh=np.inf):
    ntimes = y.shape[-1]
    nx, ne = X0_sample.shape
    
    w = np.zeros((ne, ntimes))
    xf = np.zeros((nx, ne, ntimes))
    xa = np.zeros((nx, ne, ntimes))
    w[:, 0] = np.repeat(1./ne, ne)
    xf[..., 0] = X0_sample
    xa[..., 0] = X0_sample
    
    for i in tqdm(range(1, ntimes)):
        
        # Forecast
        for j in range(ne):
            xf[:, j, i] = transition_sample(i-1, xa[:, j, i-1])
        
        # Analysis
        xa[..., i], w[:, i] = bootstrap_PF_step(
            xf[..., i], w[:, i-1], y[:, i],
            observational_pdf, Neff_thresh=Neff_thresh)
    
    return xf, xa, w
