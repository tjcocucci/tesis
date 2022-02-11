import numpy as np

rho = 28.0
sigma = 10.0
beta = 8.0 / 3.0

def dxdt(state, t):
    x, y, z = state
    dx = np.zeros(state.shape[0])
    dx[0] = sigma * (y - x)
    dx[1] = x * (rho - z) - y
    dx[2] = x * y - beta * z
    return dx