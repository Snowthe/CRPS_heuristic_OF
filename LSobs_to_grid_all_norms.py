# -*- coding: utf-8 -*-
"""
Script to obtain MIKE SHE model grid aggregated error metrics based on a text 
file of groundwater head residuals. See the example file 
"Storaa_GWhead_test_observations.txt" for the expected format.

Naming convention in output: j<col>k<row>z<layer>

Used in PEST/OSTRICH calibration.

Input: Text file with groundwater head observations and simulations, in 
    format as in "Storaa_GWhead_test_observations.txt"
Output: Grid aggregated error metrics, in format as in 
    "Storaa_GWhead_test_grids.txt". The columns "MAE_sqrt" and "MRE_sqrt" are 
    the square root of the MAE and MRE grid value, respectively, and provided
    to allow for a calculation of the aggregated MAE and MRE in PEST/OSTRICH
    (which typically simply take a weighted SSE, resulting in the MSE in case 
     one uses ME as resodials)

Usage:
LSobs_to_grid_all_norms.py <LS_xxx_observations.txt> [-c]
'-c': optional, will also output grid-based, disaggregated CRPS values (by 
calling CRPS_disagg.py)
(for the optional call, see/double check last lines in this script!)

@author: Raphael Schneider, GEUS Hydro, rs@geus.dk, May 2022
"""

import sys
import subprocess as sp
import os
import pandas as pd
import numpy as np
import re

def rmse_from_err(X, err):
     return(np.sqrt(np.mean(X[err]**2)))

def mae_from_err(X, err):
    return(np.mean(np.abs(X[err])))
    
def mae_rt_from_err(X, err):
    return(np.sqrt(np.mean(np.abs(X[err]))))

def mre_from_err(X, err):
    return(np.mean(np.sqrt(np.abs(X[err]))))

def mre_rt_from_err(X, err):
    return(np.sqrt(np.mean(np.sqrt(np.abs(X[err])))))


if len(sys.argv)==1: #if ran from IDE/console
    lsfile = 'Storaa_GWhead_observations_test.txt'
else:
    lsfile = sys.argv[1]

# read observations output from LayerStatistics
lsobs = pd.read_csv(lsfile, index_col=False, sep='\t')
# add column for gridID
lsobs.insert(1,'gridID','missing')
lsobs['gridID'] = 'j'+lsobs['COLUMN'].map(str)+'k'+lsobs['ROW'].map(str)+'z'+lsobs['LAYER'].map(str)

# aggregate values per grid (in new dataframe)
g_rmse = lsobs.groupby('gridID').apply(rmse_from_err, 'ME')
g_rmse.name = 'RMSE'
g_mae = lsobs.groupby('gridID').apply(mae_from_err, 'ME')
g_mae.name = 'MAE'
g_mae_rt = lsobs.groupby('gridID').apply(mae_rt_from_err, 'ME')
g_mae_rt.name = 'MAE_sqrt'
g_mre = lsobs.groupby('gridID').apply(mre_from_err, 'ME')
g_mre.name = 'MRE'
g_mre_rt = lsobs.groupby('gridID').apply(mre_rt_from_err, 'ME')
g_mre_rt.name = 'MRE_sqrt'
g_mean = lsobs.groupby('gridID').ME.mean()
g_nobs = lsobs.groupby('gridID').size()
g_nobs.name = 'nobs'
g_nwells = lsobs.groupby('gridID').OBS_ID.nunique()
g_nwells.name = 'nwells'
g_depth = lsobs.groupby('gridID').Depth.mean()
g_depth.name = 'depth_mean'
g_obsh = lsobs.groupby('gridID').OBS_VALUE.mean()
g_obsh.name = 'obs_mean'
g_simh = lsobs.groupby('gridID').SIM_VALUE_INTP.mean()
g_simh.name = 'sim_mean'
g_col = lsobs.groupby('gridID').COLUMN.mean()
g_row = lsobs.groupby('gridID').ROW.mean()
g_lay = lsobs.groupby('gridID').LAYER.mean()


lsgrid = pd.concat([g_mean, g_rmse, g_mae, g_mae_rt, g_mre, g_mre_rt, g_nobs, g_nwells, g_depth, g_obsh, g_simh, 
                    g_col, g_row, g_lay], axis=1)


lsgridfile = lsfile.split('observations.txt')[0]+'grids.txt'
lsgrid.to_csv(lsgridfile, sep='\t')

if len(sys.argv)==3 and sys.argv[2]=='-c':
    sp.call(sys.executable+' ./CRPS_disagg.py -i '+lsgridfile.replace('\\','/'), shell=True)
