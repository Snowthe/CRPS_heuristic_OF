# CRPS_heuristic_OF
Using the Continuous Ranked Probability Score (CRPS) as an heuristic objective 
function in hydrological model calibration

This repository contains scripts used to calculate grid-aggregated error 
metrics for residuals of groundwater head observations from wells. The error 
metrics are 
* MSE
* MAE
* MRE
* CRPS

It is originally used in context with the calibration of a MIKE SHE 
hydrological model with PEST or OSTRICH optimization tools.

"Storaa_GWhead_test_observations.txt" is a test input file, and 
"Storaa_GWhead_test_grids.txt" and "Storaa_GWhead_test_grids_crpsDis.txt" are 
the respective test output files.

## Reference
A robust objective function for calibration of groundwater models in light of 
deficiencies of model structure and observations.
Raphael Schneider, Hans Jørgen Henriksen, Simon Stisen.
Submitted to Journal of Hydrology.

## LICENSE
Copyright 2022 Raphael Schneider

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.