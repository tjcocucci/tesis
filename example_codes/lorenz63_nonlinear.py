import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.integrate import odeint
from lor63 import dxdt

Np = 100

np.random.seed(200)

state0_mean = np.array([1.0, 1.0, 1.0])*0.05
state0_var = np.eye(3)*1

state0 = scipy.stats.multivariate_normal(state0_mean, state0_var).rvs(size=Np).T
print(state0.shape)

t = np.arange(0.0, 1, 0.01)

states = np.zeros((len(t), 3, Np))
for i in range(Np):
    states[:, :, i] = odeint(dxdt, state0[:, i], t)

cm = plt.get_cmap('RdBu')
norm  = plt.Normalize(0, len(t))

fig, ax = plt.subplots()
for i in range(len(t)):
    s = 5 if ((i != 0) and (i != len(t)-1)) else 15
    alpha = 0.5 if ((i != 0) and (i != len(t)-1)) else 1
    ax.scatter(states[i, 0, :], states[i, 1, :],
        c=cm(norm(i)),
        edgecolor='none', s=s,
        alpha=alpha
    )

plt.tick_params(axis='both',
    which='both',
    bottom=False,
    top=False,
    labelbottom=False,
    right=False,
    left=False,
    labelleft=False
)

plt.show()

