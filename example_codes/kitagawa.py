import numpy as np

nx = 1

def kitagawa_model(k, x):
    x_return = np.copy(x)
    x_return[0] = 0.5*x[0] + 25*x[0]/(1+x[0]**2) + 8*np.cos(1.2*k) + np.random.randn()
    return x_return
