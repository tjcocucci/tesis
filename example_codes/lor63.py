import numpy as np

nx = 3

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def dxdt(t, state):
    x, y, z = state
    dx = np.zeros(state.shape[0])
    dx[0] = sigma * (y - x)
    dx[1] = x * (rho - z) - y
    dx[2] = x * y - beta * z
    return dx

def rk4(f, x, t, dt):
    k1 = dt * f(t, x)
    k2 = dt * f(t+dt/2, x+k1/2)
    k3 = dt * f(t+dt/2, x+k2/2)
    k4 = dt * f(t+dt  , x+k3)
    return x + (k1 + 2*(k2 + k3) + k4)/6

def lor63_step(t, x, dt):
    return rk4(dxdt, x, t, dt)
