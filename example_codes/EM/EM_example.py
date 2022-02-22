import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from gaussian_mixture import GaussianMixture

np.random.seed(200)

sigma_r = 1
H = lambda x: (x-0.1)**2

xgrid = np.linspace(-2, 2, 100)
dx = xgrid[1] - xgrid[0]

# means = [np.array([-1]), np.array([0])]
# covariances = [np.eye(1), np.eye(1)*0.5]
# weights = [0.3, 0.7]
# gm = GaussianMixture(weights, means, covariances)

# def p_x(x, theta):
#     means = [np.array([theta]), np.array([1])]
#     gm = GaussianMixture(weights, means, covariances)
#     return gm.pdf(x)

def p_x(x, theta):
    return st.norm(loc=theta).pdf(x)

def p_y_x(y, Hx, theta):
    return st.norm(loc=Hx, scale=sigma_r).pdf(y)

def p_x_y(x, y, theta):
    return p_xy(x, y, theta) / p_y(y, theta)

def p_x_y_precomp(x, y, theta, p_y_value):
    return p_xy(x, y, theta) / p_y_value

def p_xy(x, y, theta):
    return p_y_x(y, H(x), theta) * p_x(x, theta)

def p_y(y, theta):
    values = np.array([p_xy(xp, y, theta) for xp in xgrid])
    return values.sum()*dx 

def logp_y(y, theta):
    values = np.array([p_xy(xp, y, theta) for xp in xgrid])
    return np.log(values.sum()*dx) 


theta_true = -1
np.random.seed(2000)
xt = st.norm(loc=theta_true).rvs()
# xt = GaussianMixture(weights, [np.array([theta_true]), np.array([0])], covariances).sample(1)
obs = st.norm(loc=H(xt), scale=sigma_r).rvs()
theta0 = 0.7

def elbo(theta, theta_t):
    p_y_value = p_y(obs, theta_t)

    values = np.array(
        [p_x_y_precomp(xp, obs, theta_t, p_y_value) * np.log(p_xy(xp, obs, theta)) for xp in xgrid]
    )
    return values.sum()*dx

p_y_value = p_y(obs, theta0)
constant = np.array(
        [p_x_y_precomp(xp, obs, theta0, p_y_value) * 
         np.log(p_x_y_precomp(xp, obs, theta0, p_y_value)) for xp in xgrid]
).sum()*dx

elbo_values0 = np.array([elbo(t, theta0) for t in xgrid]) - constant

theta1 = xgrid[np.argmax(elbo_values0)]

p_y_value = p_y(obs, theta1)
constant = np.array(
        [p_x_y_precomp(xp, obs, theta1, p_y_value) * 
         np.log(p_x_y_precomp(xp, obs, theta1, p_y_value)) for xp in xgrid]
).sum()*dx

elbo_values1 = np.array([elbo(t, theta1) for t in xgrid]) - constant
theta2 = xgrid[np.argmax(elbo_values1)]

logpy_values = np.array([logp_y(obs, t) for t in xgrid])

np.save('xgrid.npy', xgrid)
np.save('elbo0.npy', elbo_values0)
np.save('elbo1.npy', elbo_values1)
np.save('logpy_values.npy', logpy_values)
np.save('theta_obs.npy', np.array([obs, theta0, theta1, theta2]))
