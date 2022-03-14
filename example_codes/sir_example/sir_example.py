import numpy as np
import matplotlib.pyplot as plt

nx = 3
beta = 0.4
incubation_rate = 0.2
recovery_rate = 0.2
death_rate = 0.1

def dxdt(x, beta=beta, recovery_rate=recovery_rate):

    s, i, r = x

    dx = np.zeros(nx)

    n = x.sum()
    dx[0] = -(beta * i * s) / n
    dx[1] = ((beta * i * s) / n) - incubation_rate * i
    dx[2] = recovery_rate * i

    return dx

def sir_step(x, dt, **kwargs):
    return x + dt*dxdt(x)

dt = 0.1
ncy = 700

x = np.zeros((nx, ncy))
x[:, 0] = np.array([99, 1, 0]) * 100

for i in range(1, ncy):
    x[:, i] = sir_step(x[:, i-1], dt)

fig, ax = plt.subplots()

colors = ['blue', 'red', 'green']
labels = ['Susceptibles', 'Infectados', 'Recuperados']

for i in range(nx):
    ax.plot(x[i, :], color=colors[i], label=labels[i])

ax.set_xlabel('Tiempo')
ax.set_ylabel('Población')
fig.legend(framealpha=1, ncol=nx)
plt.show()
