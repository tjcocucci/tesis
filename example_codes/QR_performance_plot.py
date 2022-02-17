import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

sQest = np.load('sQest_osc.npy')
sRest = np.load('sRest_osc.npy')
rmses = np.load('rmses_osc.npy')
coverages = np.load('coverages_osc.npy')

sQest = sQest[1:]
sRest = sRest[1:]
rmses = rmses[1:, 1:]
coverages = coverages[1:, 1:]

fig, ax = plt.subplots(1, 2)

divnorm=colors.TwoSlopeNorm(0.95)
ax[0].imshow(rmses, cmap='Reds')
ax[1].imshow(coverages, cmap='RdBu', norm=divnorm)


print(sQest,  sRest)

ax[0].set_xticks(range(len(sQest)))
ax[0].set_yticks(range(len(sRest)))
ax[1].set_xticks(range(len(sQest)))
ax[1].set_yticks(range(len(sRest)))

ax[0].set_xticklabels(sQest.round(2))
ax[0].set_yticklabels(sRest.round(2))
ax[1].set_xticklabels(sQest.round(2))
ax[1].set_yticklabels(sRest.round(2))

# fig.colorbar(r, ax=ax[0])
# fig.colorbar(c, ax=ax[1])
plt.show()
