import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

np.random.seed(123456789)

def KF_analysis(xf, Pf, y, H, R):
    S = H.dot(Pf).dot(H.T) + R
    K = Pf.dot(H.T).dot(np.linalg.inv(S))
    v = y - H.dot(xf)
    xa = xf + K.dot(v)
    Pa = Pf - K.dot(H).dot(Pf)
    return xa, Pa

def KF(x0, P0, y, M, Q, H, R):
    nx = x0.shape[0]
    m, ntimes = y.shape
    
    xf = np.zeros((nx, ntimes))
    xa = np.zeros((nx, ntimes))
    Pf = np.zeros((nx, nx, ntimes))
    Pa = np.zeros((nx, nx, ntimes))    
    
    xf[:, 0] = x0
    xa[:, 0] = x0
    Pf[..., 0] = P0
    Pa[..., 0] = P0    
    
    for i in range(1, ntimes):
        
        # Forecast
        xf[:, i] = M.dot(xa[:, i-1])
        Pf[..., i] = M.dot(Pa[..., i-1]).dot(M.T) + Q
        
        # Analysis
        if np.isnan(y[:, i]).any():
            xa[:, i], Pa[..., i] = xf[:, i], Pf[..., i]
        else:
            xa[:, i], Pa[..., i] = KF_analysis(xf[:, i], Pf[..., i], y[:, i], H, R)

    return xf, Pf, xa, Pa


 # Setup experiment

dt = 0.1
omega_sq = 2.0
ntimes = 200
nx = 2
m = 1

M = np.array([
    [ -(dt**2)*omega_sq + 1.0, dt ],
    [ -dt*omega_sq, 1.0 ]
    ])

Q = np.eye(nx) * 0.005

# H = np.eye(m)
H = np.array([
    [1, 0],
])
assert(H.shape == (m, nx))

R = np.eye(m)*0.1

 # Generate truth and obs

xt = np.zeros((nx, ntimes))
y = np.zeros((m, ntimes))
xt[:, 0] = np.ones(nx)
y[:, 0] = np.nan

for i in range(1, ntimes):
    xt[:, i] = M.dot(xt[:, i-1])
    xt[:, i] += st.multivariate_normal(cov=Q, allow_singular=True).rvs()  
    y[:, i] = H.dot(xt[:, i])
    y[:, i] += st.multivariate_normal(cov=R).rvs()

# y[:, np.arange(ntimes) % 2 != 0] = np.nan

x0 = np.array([2, 2])
P0 = np.array([[2, 0], [0, 1]])
xf, Pf, xa, Pa = KF(x0, P0, y, M, Q, H, R)

cut = 53

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
plt.subplots_adjust(hspace=0.1)
ax1.plot(y[0, :], color='tab:red', marker='.', ls='None', label='Observaciones', zorder=100)
ax1.plot(xa[0, :], color='tab:blue', label='Análisis (media)', zorder=100)
ax1.plot(xt[0, :], 'k--', label='Valor real', zorder=100)
ax1.axvline(x=cut, color='0.2')
ax1.tick_params(axis='x', bottom=False)
ax1.set_ylabel('Posición')

# ax2.plot(y[1, :], 'r.')
ax2.plot(xa[1, :], color='tab:blue')
ax2.plot(xt[1, :], 'k--')
ax1.axvline(x=cut, color='0.5')

ax2.set_xlabel('Tiempo')
ax2.set_ylabel('Velocidad')

fig.legend(framealpha=1)
plt.show()

# xgrid = np.linspace(np.min(xt[0, :]), np.max(xt[0, :]), 500)
xgrid = np.linspace(0.7, 1.5, 500)

fig, ax = plt.subplots()
xa_pdf = st.norm(xa[0, cut], Pa[0, 0, cut]).pdf(xgrid)
xf_pdf = st.norm(xf[0, cut], Pf[0, 0, cut]).pdf(xgrid)
likelihood = st.norm(y[0, cut], R[0, 0]).pdf(xgrid)

xa_pdf[xa_pdf < 0.001] = np.nan
xf_pdf[xf_pdf < 0.001] = np.nan
likelihood[likelihood < 0.001] = np.nan
print(st.norm(xa[0, cut], Pa[0, 0, cut]).isf(0.999))
print(st.norm(xa[0, cut], Pa[0, 0, cut]).pdf(st.norm(xa[0, cut], Pa[0, 0, cut]).ppf(0.999)))

ax.plot(xgrid, xa_pdf, color='tab:blue', label=r'pdf filtrante: $p(x_t | y_{1:t})$')
ax.plot(xa[0, cut], 0, color='tab:blue', marker='o', ms=5, ls='None', label='Media análisis', zorder=100)
ax.plot(xgrid, xf_pdf, color='tab:olive', label=r'pdf del pronóstico: $p(x_t | y_{1:t-1})$')
ax.plot(xf[0, cut], 0, color='tab:olive', marker='o', ms=5, ls='None', label='Media pronóstico', zorder=100)
ax.plot(xgrid, likelihood, color='tab:red', label=r'Verosimilitud observacional: $p(y_t | x_t)$')
ax.plot(y[0, cut], 0, color='tab:red', marker='o', ms=5, ls='None', label='Observación', zorder=100)
ax.plot(xt[0, cut], 0, color='black', marker='o', ms=5, ls='None', label='Valor real', zorder=100)

# ax.vlines(ymin=0, ymax=np.nanmax(xa_pdf), x=xa[0, cut], color='b')
# ax.vlines(ymin=0, ymax=np.nanmax(xf_pdf), x=xf[0, cut], color='g')
# ax.vlines(ymin=0, ymax=np.nanmax(likelihood), x=y[0, cut], color='r')

ax.set_ylabel(r'$p(\cdot)$')
ax.set_xlabel('Posición')
plt.legend(framealpha=1)
plt.show()
