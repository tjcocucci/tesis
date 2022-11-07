import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import seaborn as sns

sQest = np.load('sQest_osc.npy')
sRest = np.load('sRest_osc.npy')
rmses = np.load('rmses_osc.npy')
coverages = np.load('coverages_osc.npy')

# first = 2
# last = len(sQest) - 2
# sQest = sQest[first:last]
# sRest = sRest[first:last]
# rmses = rmses[first:last, first:last]
# coverages = coverages[first:last, first:last]

fig, ax = plt.subplots(1, 2, sharey=True)
fig.subplots_adjust(top=0.725, bottom=0.155, left=0.125, right=0.9, hspace=0.2, wspace=0.2)

divnorm=colors.TwoSlopeNorm(0.95)

itrue = np.where(abs(sQest-0.5) < 1e-6)[0][0]
jtrue = np.where(abs(sRest-0.5) < 1e-6)[0][0]

sns.heatmap(rmses, ax=ax[0], cmap='RdBu_r', cbar_kws=dict(use_gridspec=False, location="top"))
sns.heatmap(coverages, ax=ax[1], cmap='RdBu', norm=divnorm, cbar_kws=dict(use_gridspec=False, location="top"))
ax[0].add_patch(Rectangle((itrue, jtrue), 1, 1, fill=False, edgecolor='red', lw=3))
ax[1].add_patch(Rectangle((itrue, jtrue), 1, 1, fill=False, edgecolor='red', lw=3))

ax[0].set_yticks(np.arange(1.5, len(sQest), 2))
ax[1].set_yticks(np.arange(1.5, len(sQest), 2))
ax[0].set_yticklabels(sQest.round(2)[1::2])
ax[1].set_yticklabels(sQest.round(2)[1::2])

ax[0].set_xticks(np.arange(1.5, len(sRest), 2))
ax[1].set_xticks(np.arange(1.5, len(sRest), 2))
ax[0].set_xticklabels(sRest.round(2)[1::2])
ax[1].set_xticklabels(sRest.round(2)[1::2])

ax[0].tick_params(axis='x', rotation=90)
ax[1].tick_params(axis='x', rotation=90)
ax[0].tick_params(axis='y', rotation=0)

ax[0].set_xlabel(r'$\sigma_R^2$')
ax[0].set_ylabel(r'$\sigma_Q^2$')
ax[1].set_xlabel(r'$\sigma_R^2$')

ax[0].set_title('RMSE', y=50, pad=50)
ax[1].set_title('Cobertura', y=50, pad=50)

plt.savefig("../figs/QR_heatmap.eps",bbox_inches='tight')
plt.show()
