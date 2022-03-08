
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.cm as cm
import matplotlib as mpl
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import pickle

nx = 3
sig_q = 0.3
Q_true = sig_q * np.eye(nx)

alphas = np.arange(0.55, 1, 0.05)

def subdiagonal_mean(A, idx=0):
    n = A.shape[0]
    return np.mean([A[i, i+idx] for i in range(n-idx)])

with open('Q_hat_EnKFS.pkl', 'rb') as file:
    Q_hat_EnKFS = pickle.load(file)  

nx, nx, ncy, Nalphas = Q_hat_EnKFS.shape

fig, ax1 = plt.subplots()
colormap = plt.get_cmap('Greys', Nalphas)
colormap = plt.cm.Greys(np.linspace(0.3, 1, Nalphas)) 
colormap = ListedColormap(colormap)
colormap = plt.get_cmap('viridis_r', Nalphas)


for j in range(Nalphas):
    diag_mean_EnKFS = np.array([subdiagonal_mean(Q_hat_EnKFS[..., i, j], idx=0)
                               for i in range(ncy)])
    c = colormap(j)
    l2 = ax1.plot(diag_mean_EnKFS, color=colormap(j), ls='-')
    
l4 = ax1.axhline(y=np.mean(np.diag(Q_true)), label='True value', c='k', ls='--')

cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8]) 
fig.subplots_adjust(right=0.8)

norm = mpl.colors.BoundaryNorm(np.arange(1, Nalphas+1) , Nalphas+1)
sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, cax=cbaxes)
tick_locs = (np.arange(Nalphas+1) - 0.5)
cbar.set_ticks(tick_locs)
cbar.ax.set_yticklabels(alphas.round(2))
cbar.ax.set_ylabel(r'   $\alpha$', rotation=0)

ax1.set_xlabel('Ciclo de asimilación')
ax1.set_ylabel(r'Media de la diagonal de $\mathbf{Q}$')

plt.show()