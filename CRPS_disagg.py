# -*- coding: utf-8 -*-
"""
Script to calculate a "disaggregated CRPS", i.e. output a value for each 
single observation/grid ("ensemble member" instead of only the actual CRPS value)

Used in combination with PEST/OSTRICH, to make the individual observations 
visible to PEST/OSTRICH (and, hence, give correct estimates of parameter 
confidence intervals etc. if the used optimization algorithm supports that)
I.e. after PEST/OSTRICH squares and sums all residuals, the resulting value is 
equal to the CRPS squared

Input: _grids.txt file, which is output from LSobs_to_grid_all_norms.py
Output: _crpsDis.txt file, which displays all crps values (weighted and 
    unweighted), aggregated as well as disaggregated per grid

Uses code (adapted) from the properscoring python package: 
https://github.com/TheClimateCorporation/properscoring
(LICENSE see below)

Usage:
    CRPS_disagg.py -i <LS_grids_file.txt>

@author: Raphael Schneider, GEUS Hydro, rs@geus.dk, May 2022
Reference: Schneider, R., Henriksen, H. J., and Stisen, S.: A robust objective 
function for calibration of groundwater models in light of deficiencies of 
model structure and observations, 613, 128339, 
https://doi.org/10.1016/j.jhydrol.2022.128339, 2022.
"""

import sys
import getopt
import pandas as pd
import numpy as np

#%% CRPS Functions
"""
The following functions are taken and adapted from the properscoring python package: 
https://github.com/TheClimateCorporation/properscoring
(for ease of use, to avoid having to install the package)

----
LICENSE (properscoring)
----
Copyright 2015 The Climate Corporation
Licensed under the Apache License, Version 2.0 (the "License"); you may not 
use this file except in compliance with the License. You may obtain a copy 
of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT 
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the 
License for the specific language governing permissions and limitations 
under the License.
"""
def _move_axis_to_end(array, axis):
    """
    Function taken directly from properscoring package (_utils.py): 
    https://github.com/TheClimateCorporation/properscoring
    """
    array = np.asarray(array)
    return np.rollaxis(array, axis, start=array.ndim)

def _argsort_indices(a, axis=-1):
    """
    Function taken directly from properscoring package (_utils.py): 
    https://github.com/TheClimateCorporation/properscoring
    
    Like argsort, but returns an index suitable for sorting the
    the original array even if that array is multidimensional
    """
    a = np.asarray(a)
    ind = list(np.ix_(*[np.arange(d) for d in a.shape]))
    ind[axis] = a.argsort(axis)
    return tuple(ind)

def crps_disaggregated(observations, forecasts, weights=None, gridID=None, issorted=False, axis=-1):
    """
    Code taken and adapted from properscoring package: 
    https://github.com/TheClimateCorporation/properscoring
    crps_ensemble() in _crps.py and _crps_ensemble_gufunc() in _gufuncs.py
    
    PLEASE NOTE: This function works only for a single "observation", e.g. the 
    expected value of 0 for ME on wells!
    
    Parameters:
    ----
    observations : float
        The optimal value, typically 0 (ME = 0).
        (In the standard formulation of CRPS, this would be the observed/true 
        value, and could be a vector. Here, single "observation" is expected)
    forecasts: array of len N
        The actual forecasted/simulated values (i.e. actual MEs)
    weights : array_like, optional (default=None)
        If provided, the CRPS is calculated exactly with the assigned
        probability weights to each forecast. Weights should be positive, but
        do not need to be normalized. By default, each forecast is weighted
        equally.
    gridID : array of len N
        (optional) Identifier of each well/grid as additional input; returns it 
        in same order as the individual CRPS values
    issorted : bool, optional (default=False)
        Optimization flag to indicate that the elements of `ensemble` are
        already sorted along `axis`.
    axis : int, optional (default=-1)
        Axis in forecasts and weights which corresponds to different ensemble
        members, along which to calculate CRPS.

    Returns
    -------
    integral : float
        CRPS
    increment : np.ndarray
        CRPS, individual contribution of each forecast
    gridID : np.ndarray
        gridID, sorted as increments
    """
    observations = np.asarray(observations)
    forecasts = np.asarray(forecasts)
    if gridID is None:  # gridID added by Raphael Schneider
        gridID = np.arange(len(forecasts))
    else:
        gridID = np.asarray(gridID)
    if axis != -1:
        forecasts = _move_axis_to_end(forecasts, axis)
        gridID = _move_axis_to_end(gridID, axis) #added by Raphael Schneider
    if weights is not None:
        weights = _move_axis_to_end(weights, axis)
        if weights.shape != forecasts.shape:
            raise ValueError('forecasts and weights must have the same shape')
    if observations.shape not in [forecasts.shape, forecasts.shape[:-1]]:
        raise ValueError('observations and forecasts must have matching '
                         'shapes or matching shapes except along `axis=%s`'
                         % axis)
    if observations.shape == forecasts.shape:
        if weights is not None:
            raise ValueError('cannot supply weights unless you also supply '
                             'an ensemble forecast')
        return abs(observations - forecasts)
    if not issorted:
        if weights is None: #adapted by Raphael Schneider
            #forecasts = np.sort(forecasts, axis=-1)
            idx = _argsort_indices(forecasts, axis=-1)
            forecasts = forecasts[idx]
            gridID = gridID[idx]
        else:
            idx = _argsort_indices(forecasts, axis=-1)
            forecasts = forecasts[idx]
            gridID = gridID[idx] #added by Raphael Schneider
            weights = weights[idx]
    if weights is None:
        weights = np.ones_like(forecasts)
    # BEWARE! forecasts are assumed sorted in NumPy's sort order
    # we index the 0th element to get the scalar value from this 0d array:
    # http://numba.pydata.org/numba-doc/0.18.2/user/vectorize.html#the-guvectorize-decorator
    #obs = observations[0] # works only if function definded in "numba" style as in _gufuncs
    obs = observations
    if np.isnan(obs):
        #result = np.nan
        #return
        return np.nan #adapted by Raphael Schneider
    total_weight = 0.0
    for n, weight in enumerate(weights):
        if np.isnan(forecasts[n]):
            # NumPy sorts NaN to the end
            break
        if not weight >= 0: #adapted by Raphael Schneider
            # this catches NaN weights
            return np.nan
        total_weight += weight
    obs_cdf = 0
    forecast_cdf = 0
    prev_forecast = 0
    integral = 0
    increment = np.empty_like(forecasts) #introduced by Raphael Schneider to be able to return the single values assigned to each single observation
    for n, forecast in enumerate(forecasts):
        if np.isnan(forecast):
            # NumPy sorts NaN to the end
            if n == 0:
                integral = np.nan
            # reset for the sake of the conditional below
            forecast = prev_forecast
            break

        if obs_cdf == 0 and obs < forecast:
            increment[n] = (obs - prev_forecast) * forecast_cdf ** 2 + (forecast - obs) * (forecast_cdf - 1) ** 2
            integral += increment[n]
            obs_cdf = 1
        else:
            increment[n] = ((forecast - prev_forecast) * (forecast_cdf - obs_cdf) ** 2)
            integral += increment[n]

        forecast_cdf += weights[n] / total_weight
        prev_forecast = forecast
    if obs_cdf == 0:
        # forecast can be undefined here if the loop body is never executed
        # (because forecasts have size 0), but don't worry about that because
        # we want to raise an error in that case, anyways
        integral += obs - forecast
    return integral, increment, gridID


#%% main code - called from command line
def main():
    # command line option handling
    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:')
    except getopt.GetoptError as err:
        print(str(err))
        print(r'Usage: CRPS_disagg.py -i <LS_wells or grids file>')
        sys.exit(2)
    infile = ''
    for o, a in opts:
        if o == '-i':
            infile = a
        else:
            assert False, 'unhandled option'
    
    # use next line to test from IDE
    # infile = 'Storaa_all_QS_2000-2008_grids.txt'
    
    # actual stuff
    lsin = pd.read_csv(infile, index_col=False, sep='\t') #read in ls file as DataFrame
    lsin.rename(columns = {'depth_mean':'Depth'}, inplace=True)
    # assign weights based on depth and number of observations
    lsin['weight'] = np.nan
    lsin.loc[lsin['nobs']==1, ['weight']] = 1
    lsin.loc[(lsin['nobs']>=2) & (lsin['nobs']<10), ['weight']] = 2
    lsin.loc[(lsin['nobs']>=10) & (lsin['nobs']<100), ['weight']] = 3
    lsin.loc[lsin['nobs']>=100, ['weight']] = 5
    if sum(np.isnan(lsin.weight)) > 0:
        sys.exit(r'ERROR: did not manage to assign weights to all observations')
    
    # initialize Dataframe to hold results (only aggregated values; rest added later)
    cols = ['unweighted','weighted','ME','nobs']
    lsout = pd.DataFrame(index=['CRPS'], columns=cols, dtype=float)
    # get CRPS from properscoring (i.e. the aggregated value)
    temp = np.asarray(lsin.ME).copy()
    lsout.loc['CRPS','unweighted'] = crps_disaggregated(0, temp)[0] #unweighted
    tempw = np.asarray(lsin.weight)
    lsout.loc['CRPS','weighted'] = crps_disaggregated(0, temp, weights=tempw)[0] #weighted
    lsout.loc['CRPS','ME'] = np.nan
    lsout.loc['CRPS','nobs'] = lsin['nobs'].sum()
  
    # disaggregated CRPS
    temp = None; temp = np.asarray(lsin.ME).copy()
    tempg = None; tempg = np.asarray(lsin.gridID).copy()
    crps_d = crps_disaggregated(0, temp, tempg) #unweighted
    tempw = None; tempw = np.asarray(lsin.weight)
    crpsw_d = crps_disaggregated(0, temp, tempg, tempw) #weighted
    """
    NOTE: The single CRPS values do not come in the same order as input
    i.e. the gridID is carried through the CRPS function and returned in 
    respective order (3rd element of return tuple)
    Here, the observations still will be written out in original order, which
    also is fixed between runs, to allow handling with PEST/OSTRICH
    """
    markerdf = pd.DataFrame(index=['start of disaggregated CRPS'], columns=cols, data='') #add to provide marker for PEST/OSTRICH
    tempdf = pd.DataFrame(index=lsin.gridID, columns=cols, dtype=float)
    for i, row in tempdf.iterrows():
        """
        OBIVOUS IMPLEMENTATION - DOES NOT WORK: pass to PEST as sqrt(CRPS_i), results as CRPS^2 as final phi, does not work as the individual contributions to CRPS from each observation differ wildly from run to run
           tempdf.loc[i, 'unweighted'] = np.sqrt(crps_d[1][crps_d[2] == row.name]) #pass to PEST as sqrt(), as PEST itself squares and sums - results in CRPS^2 as final Phi
           tempdf.loc[i, 'weighted'] = np.sqrt(crpsw_d[1][crpsw_d[2] == row.name]) #pass to PEST as sqrt(), as PEST itself squares and sums - results in CRPS^2 as final Phi
        """
        #ALTERNATIVE: pass to PEST as ME scaled with CRPS/RSSE, results in CRPS^2 as final Phi
        tempdf.loc[i, 'unweighted'] = float(lsin[lsin['gridID'] == row.name]['ME']) * (crps_d[0] / np.sqrt(np.sum(lsin['ME']**2)))
        tempdf.loc[i, 'weighted'] = float(lsin[lsin['gridID'] == row.name]['ME']) * (crpsw_d[0] / np.sqrt(np.sum(lsin['ME']**2)))
        tempdf.loc[i, 'ME'] = float(lsin[lsin['gridID'] == row.name]['ME'])
        tempdf.loc[i, 'nobs'] = int(lsin[lsin['gridID'] == row.name]['nobs'])
    
    lsout = lsout.append(markerdf)
    # remove all boreholes with negative z (indication of boreholes below bottom of model or above topography)
    tempdf.drop(tempdf[np.asarray(list(zip(*tempdf.index.str.split('z')))[1], dtype=int) < 0].index, inplace=True)
    lsout = lsout.append(tempdf)

    # write back to .txt file that can be read by PEST
    lsout.to_csv(infile.split('.txt')[0]+'_crpsDis.txt', sep='\t', na_rep='NaN')

if __name__ == "__main__":
    main()
