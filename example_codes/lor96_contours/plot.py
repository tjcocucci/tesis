import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib as mpl
import pickle
from matplotlib.ticker import FuncFormatter

def thousands(x, pos):
    'The two args are the value and tick position'
    return '%1.1f' % (x * 1e-3)

formatter = FuncFormatter(thousands)

def subdiagonal_mean(A, idx=0):
    n = A.shape[0]
    return np.mean([A[i, i+idx] for i in range(n-idx)])

with open('q_means_good_cond.pkl', 'rb') as file:
    q_means_good_cond = pickle.load(file)
with open('r_means_good_cond.pkl', 'rb') as file:
    r_means_good_cond = pickle.load(file)
with open('loglik_enkf1.pkl', 'rb') as file:
    loglik_enkf_good = pickle.load(file)
with open('q_means_bad_cond.pkl', 'rb') as file:
    q_means_bad_cond = pickle.load(file)
with open('r_means_bad_cond.pkl', 'rb') as file:
    r_means_bad_cond = pickle.load(file)
with open('loglik_enkf.pkl', 'rb') as file:
    loglik_enkf_bad = pickle.load(file)

nx = 8
m = 8

sig_q_bad = 0.3
Q_true_bad_cond = sig_q_bad * np.eye(nx)
sig_r_bad = 1.5
R_true_bad_cond = sig_r_bad * np.eye(m)

sig_q_good = 0.3
Q_true_good_cond = sig_q_good * np.eye(nx)
sig_r_good = 0.5
R_true_good_cond = sig_r_good * np.eye(m)

rs = np.arange(0.1, 3, 0.125)
Nr = len(rs)
qs = np.arange(0.1, 3, 0.125)
Nq = len(qs)

cmap = mpl.cm.get_cmap('jet_r')
# indeces = (np.linspace(0.02, 0.8, 256))**2
indeces = np.linspace(0.1, 0.8, 256)
newcmp = ListedColormap(cmap(indeces))

lines_color = '1.0'


X, Y = np.meshgrid(rs, qs)

fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

vmin = np.min([loglik_enkf_good, loglik_enkf_bad])
vmax = np.max([loglik_enkf_good, loglik_enkf_bad])
levels = np.linspace(vmin, vmax, 50)
levels = 500 * np.round(levels/500)
# levels = np.linspace(100*np.floor(vmin/100), 100*np.ceil(vmax/100), 35)

contour = ax1.contourf(X, Y, loglik_enkf_good.T, levels, vmin=vmin, vmax=vmax, cmap=newcmp)
for j in range(q_means_good_cond.shape[0]):
    ax1.scatter(r_means_good_cond[j, :7], q_means_good_cond[j, :7], color=str(j/q_means_good_cond.shape[0]), s=0.5)
ax1.plot([subdiagonal_mean(R_true_good_cond, idx=0)],
         [subdiagonal_mean(Q_true_good_cond, idx=0)], 'ro')
ax1.set_xlabel('$\sigma_R^2$')
ax1.set_ylabel('$\sigma_Q^2$')

contour2 = ax2.contourf(X, Y, loglik_enkf_bad.T, levels, vmin=vmin, vmax=vmax, cmap=newcmp)
for j in range(q_means_bad_cond.shape[0]):
    ax2.scatter(r_means_bad_cond[j, :7], q_means_bad_cond[j, :7], color=str(j/q_means_bad_cond.shape[0]), s=0.5)


ax2.plot([subdiagonal_mean(R_true_bad_cond, idx=0)],
         [subdiagonal_mean(Q_true_bad_cond, idx=0)], 'ro')
ax2.set_xlabel('$\sigma_R^2$')

cbaxes = fig.add_axes([0.1, 0.2, 0.8, 0.03]) 

ax1.set_ylim(0.1, 2.9)
ax1.set_xlim(0.1, 2.9)
ax2.set_ylim(0.1, 2.9)
ax2.set_xlim(0.1, 2.9)

fig.subplots_adjust(bottom=0.4)
cbar = plt.colorbar(contour2, orientation='horizontal', shrink=0.8, cax=cbaxes, format=formatter)
cbar.set_label(r'Log-verosimilitud $[10^3]$')
labels = cbar.ax.get_xticklabels()
cbar.ax.set_xticklabels(labels, rotation=30, ha='right')

plt.show()