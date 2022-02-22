import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

xgrid = np.load('xgrid.npy')
elbo0 = np.load('elbo0.npy')
elbo1 = np.load('elbo1.npy')
logpy_values = np.load('logpy_values.npy') 
obs, theta0, theta1, theta2 = np.load('theta_obs.npy')
thetas = np.array([theta0, theta1, theta2])

logp_theta = logpy_values[np.nonzero(np.in1d(xgrid, [theta0, theta1, theta2]))[0]]


logp_theta = logpy_values[np.isclose(xgrid, thetas[:, np.newaxis], xgrid[1]-xgrid[0]).sum(0) == 1]
print(logp_theta)


fig, ax = plt.subplots()
plt.subplots_adjust(
    top=0.88,
    bottom=0.11,
    left=0.11,
    right=0.9,
    hspace=0.2,
    wspace=0.2
)



cmap = cm.get_cmap('Blues', 4)
ax.plot(xgrid, elbo0, color=cmap(1), label=r'$\mathcal{L}(p(\mathbf{X} | \mathbf{Y} ; \mathbf{\theta}_1), \mathbf{\theta})$')
ax.plot(xgrid, elbo1, color=cmap(2), label=r'$\mathcal{L}(p(\mathbf{X} | \mathbf{Y} ; \mathbf{\theta}_2), \mathbf{\theta})$')
ax.plot(xgrid, logpy_values, color='red', label=r'$\log p(\mathbf{Y} ; \mathbf{\theta})$')
# ax.axvline(x=obs, color='r', ls='--', label='observation')
ax.vlines(x=thetas[0], color=cmap(1), ls='--', ymin=-6, ymax=logp_theta[0])
ax.vlines(x=thetas[1], color=cmap(2), ls='--', ymin=-6, ymax=logp_theta[1])
ax.vlines(x=thetas[2], color=cmap(3), ls='--', ymin=-6, ymax=logp_theta[2])
# ax.vlines(x=thetas[0], color=cmap(1), ls='--', label=r'$\mathbf{\theta}_0$', ymin=-6, ymax=logp_theta[0])
# ax.vlines(x=thetas[1], color=cmap(2), ls='--', label=r'$\mathbf{\theta}_1$', ymin=-6, ymax=np.max(elbo0))
# ax.vlines(x=thetas[2], color=cmap(3), ls='--', label=r'$\mathbf{\theta}_2$', ymin=-6, ymax=np.max(elbo1))
ax.set_ylim(-2.2, -1.9)
ax.set_xlim(-1.8, 1.8)

ax.set_xticks(thetas)
ax.set_xticklabels([r'$\mathbf{\theta}_0$', r'$\mathbf{\theta}_1$', r'$\mathbf{\theta}_2$'])

# ax.set_xlabel(r'$\theta$')
ax.set_ylabel('Log-verosimilitud')

ax.legend(framealpha=1, loc='lower left')
plt.show()
