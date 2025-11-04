# AD99py 

## Library
This repository includes the core implementation of AD99 independently of the processing code used for the manuscript. The package can be installed in editable mode by cloning this repository and running:
```
pip install -e . 
```
To use the package simply import:
```python
from ad99py import AlexanderDunkerton1999

ad99 = AlexanderDunkerton1999()
gwd_u = ad99.gwd(u,N,z,rho)
```
This package also has optional support for dask for running an implementation of AD99 that is more dask optimized (`ad99pydask.AlexanderDunkerton1999Dask`). Your mileage may vary on performance between both variants depending on hardware.

This package also contains helper methods to calculate buoynacy frequency and density from standard model output fields (`ad99py.variables`) in addition to calculating directly resolved GW flux contributions (`ad99py.resolved_flux`). If Loon data is available, `ad99py.loon` can build a dictionary of estimated various Loon fluxes.

## Notebooks
This notebooks contain the code used to generate figures + supplemental materials for the manuscript "Steady source gravity wave parameterizations and the observed momentum flux intermittency". Notebook include:
- `era5_compute_fluxes_2014.ipynb`, downloads 2014 ERA5 data from GCP bucket, coarsifies and calculates GWMF using AD99
- `gcm_compute_fluxes.ipynb`, uses MiMA data and calculates GWMF using AD99
- `flux_distrbutions.ipynb`, caculates and visualizes the PDFs of momentum fluxes at a level and over a certain ocean basin (or all basins)
- `calibrate.ipynb`, perform tuning experiments by varying the value of $c_w$ and $F_{S0}$ and visualizing the impact on the PDFs and the moment
- `remaining_flux.ipynb`, shows the fraction of total momentum flux still present in each profile (or averaged over many profiles).
