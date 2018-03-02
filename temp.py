import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

kern = 'gaussian'
minx, maxx = -4, 7

for band in [2, 1, 0.5]:
    X = np.array([[2.60],[2.08],[2.36],[3.05]])
    
    X_plot = np.linspace(minx, maxx, 10000)[:, np.newaxis]
    fig, ax = plt.subplots()
    kde = KernelDensity(kernel=kern, bandwidth=band).fit(X)
    log_dens = kde.score_samples(X_plot)
    Xdns = np.exp(log_dens)
    ax.plot(X_plot[:, 0], Xdns, color = 'b')
    ax.plot(X[:, 0], -0.05 + 0 * X[:, 0], '+k', color = 'b')
    
    X = np.array([[-0.4],[3.60]])
    
    kde = KernelDensity(kernel=kern, bandwidth=band).fit(X)
    log_dens = kde.score_samples(X_plot)
    Ydns = np.exp(log_dens)
    ax.plot(X_plot[:, 0], Ydns, color = 'r')
    ax.plot(X[:, 0], -0.05 + 0 * X[:, 0], '+k', color = 'r')
    
    
    def dr(x):
         if x:
             return 'r'
         else:
             return 'b'
    drf = np.vectorize(dr)
    drc = drf(Xdns <= Ydns)
    
    ax.scatter(X_plot[:, 0], -0.075 + 0 * X_plot[:, 0], c = drc, s = 1)
    
    ax.set_xlim(minx, maxx)
    ax.set_ylim(-0.1,.7)
    plt.show()
    
    dreg = []
    for i in range(1, len(drc)):
        if drc[i] != drc[i-1]:
            dreg += [X_plot[:, 0][i]]
    
    print('Window width: ',band)
    print('Decision regions', dreg)