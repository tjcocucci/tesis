import numpy as np

nx = 3

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def dxdt(t, state, rho=rho, sigma=sigma, beta=beta):
    x, y, z = state
    dx = np.zeros(state.shape[0])
    dx[0] = sigma * (y - x)
    dx[1] = x * (rho - z) - y
    dx[2] = x * y - beta * z
    return dx

def rk4(f, x, t, dt, **kwargs):
    k1 = dt * f(t, x, **kwargs)
    k2 = dt * f(t+dt/2, x+k1/2, **kwargs)
    k3 = dt * f(t+dt/2, x+k2/2, **kwargs)
    k4 = dt * f(t+dt  , x+k3, **kwargs)
    return x + (k1 + 2*(k2 + k3) + k4)/6

def lor63_step(t, x, dt, **kwargs):
    return rk4(dxdt, x, t, dt, **kwargs)
