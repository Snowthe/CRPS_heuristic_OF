# -*- coding: utf-8 -*-
"""
Comparing the CRPS as a heuristic norm compared to other norms, the RMSE, 
MAE, and MRE (i.e. the L2, L1 and L1/2 norm).
Displaying the CRPS' relative insensitivity to (a small group of) outliers, 
compared to its high sensitivity to an overall bias.

Used in connection with the manuscript 
"A robust objective function for calibration of groundwater models in light of 
deficiencies of model structure and observations"
by Raphael Schneider, Hans Jørgen Henriksen, Simon Stisen
(this script displays Figure 2 in the manuscript)

@author: Raphael Schneider, GEUS Hydro, rs@geus.dk, May 2022
"""

import numpy as np
import properscoring as ps
import statsmodels.distributions as smd
import matplotlib.pyplot as plt

obs = 0 # true value
nsim = 100000 # number of "simulated values"
nscen = 4 # number of "scenarios"

# define "scenarios"
sim = np.empty((nscen,nsim))
sim[0,:] = np.random.normal(loc=0.0, scale=1.0, size=nsim) #ndist, mean 0, std 1 - BENCHMARK
sim[1,:] = np.random.normal(loc=0.0, scale=2.0, size=nsim) #ndist, mean 0, std 2.0 (different spread)
sim[2,:] = sim[0,:] + 0.5 #ndist, mean 0.2, std 1 (BENCHMARK plus bias)
temp = np.random.normal(loc=0.0, scale=5.0, size=int(nsim/10)) #add 10% positive outliers to BENCHMARK
temp = abs(temp)
sim[3,:] = sim[0,:]; sim[3,0:len(temp)] = temp

# calculate metrics
crps = np.empty((nscen,2)) # column 1: value; column 2: ratio to BENCHMARK
rmse = crps.copy()
mse = crps.copy()
mae = crps.copy()
mre = crps.copy()
for i in range(sim.shape[0]):
    crps[i,0] = ps.crps_ensemble(obs,sim[i,:])
    crps[i,1] = crps[i,0] / crps[0,0]
    rmse[i,0] = np.sqrt(np.mean((sim[i,:] - obs)**2))
    rmse[i,1] = rmse[i,0] / rmse[0,0]
    mse[i,0] = np.mean((sim[i,:] - obs)**2)
    mse[i,1] = mse[i,0] / mse[0,0]
    mae[i,0] = np.mean(np.abs(sim[i,:] - obs))
    mae[i,1] = mae[i,0] / mae[0,0]
    mre[i,0] = np.mean(np.sqrt(np.abs((sim[i,:] - obs))))
    mre[i,1] = mre[i,0] / mre[0,0]



# plot for paper
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(9,6))
axes = np.ravel(axes)
axes[0].axis([-5, 5, 0, 1])
for i in range(nscen):
    ecdf = smd.ECDF(sim[i,:])
    x = np.linspace(min(sim[i,:]), max(sim[i,:]), num=nsim)
    y = ecdf(x)
    axes[i].step(x, y, color='b')
    axes[i].axvline(x=obs, color='r', linestyle=':')
    if i == 0:
        axes[i].set_title('$\mathcal{N}(0,1)$ - reference (r.)'+'\n'+' ')
    elif i == 1: 
        axes[i].set_title('$\mathcal{N}(0,2)$'+'\n'+
            'CRPS: '+str(round(crps[i,1], 1))+', MSE: '+str(round(mse[i,1], 1))+', MAE: '+str(round(mae[i,1], 1))+', MRE: '+str(round(mre[i,1], 1))+' of r.')
    elif i == 2:
        axes[i].set_title('$\mathcal{N}(0.5,1)$'+'\n'+
          'CRPS: '+str(round(crps[i,1], 1))+', MSE: '+str(round(mse[i,1], 1))+', MAE: '+str(round(mae[i,1], 1))+', MRE: '+str(round(mre[i,1], 1))+' of r.')
        axes[i].set_xlabel('error')
    elif i == 3:
        axes[i].set_title('$\mathcal{N}(0,1)$ with 10% outliers from $|\mathcal{N}(0,5)|$'+'\n'+
                  'CRPS: '+str(round(crps[i,1], 1))+', MSE: '+str(round(mse[i,1], 1))+', MAE: '+str(round(mae[i,1], 1))+', MRE: '+str(round(mre[i,1], 1))+' of r.')
        axes[i].set_xlabel('error')
fig.tight_layout()
