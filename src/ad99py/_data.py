import requests
import numpy as np
import pathlib
import os
from importlib import resources
from platformdirs import user_cache_dir
from ._masks import get_xarray_mask,get_numpy_mask
from ._loon import process_flights

PACKAGE_NAME = "ad99py"
LOON_DATA_URL = 'https://stacks.stanford.edu/file/zh044ts5443/loon_GW_momentum_fluxes.csv'
LOON_NC_MASK_NAME = 'loon_masks.nc'
LOON_NP_MASK_NAME = 'loon_masks.npy'
CACHE_ENV_VAR = 'AD99PY_CACHE_DIR'

def get_loon_nc_mask_path():
    cache_path = get_cache_dir() / LOON_NC_MASK_NAME
    if not cache_path.exists():
        mask_ds = get_xarray_mask()
        mask_ds.to_netcdf(cache_path)
    return cache_path 

def get_loon_np_mask_path():
    cache_path = get_cache_dir() / LOON_NP_MASK_NAME
    if not cache_path.exists():
        mask_np = get_numpy_mask()
        np.save(cache_path,mask_np)
    return cache_path

def get_cache_dir() -> pathlib.Path:
    env_path = os.getenv(CACHE_ENV_VAR)
    if env_path: 
        path = pathlib.Path(env_path).expanduser().resolve()
    else:
        path = pathlib.Path(user_cache_dir(PACKAGE_NAME))
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_data(filename: str, url: str) -> pathlib.Path:
    cache_path = get_cache_dir() / filename
    if not cache_path.exists():
        print(f"Downloading {filename} to {cache_path}")
        r = requests.get(url)
        r.raise_for_status()
        with open(cache_path, 'wb') as f:
            f.write(r.content)
    return cache_path
    
def download_loon_data():
    print(f"[INFO] Downloading Loon data from {LOON_DATA_URL}")
    print("[INFO] Loon data is licensed under CC BY 4.0. Please cite the original source (Rhodes and Candido, 2021) when using this data.")
    return ensure_data('loon_GW_momentum_fluxes.csv', LOON_DATA_URL)

def save_loon_basins():
    loon_data = download_loon_data()
    masks = get_numpy_mask()
    trop_atl_flights,extra_atl_flights,extra_pac_flights,indian_flights,trop_pac_flights,SO_flights = process_flights(loon_data,masks)
    trop_atl_flights_path = get_cache_dir() / 'trop_atl_flights_flux.npy'
    extra_atl_flights_path = get_cache_dir() / 'extra_atl_flights_flux.npy'
    extra_pac_flights_path = get_cache_dir() / 'extra_pac_flights_flux.npy'
    indian_flights_path = get_cache_dir() / 'indian_flights_flux.npy'
    trop_pac_flights_path = get_cache_dir() / 'trop_pac_flights_flux.npy'
    SO_flights_path = get_cache_dir() / 'SO_flights_flux.npy'
    np.save(trop_atl_flights_path,trop_atl_flights) 
    print(f"[INFO] Saved Tropical Atlantic flights to {trop_atl_flights_path}")
    np.save(extra_atl_flights_path,extra_atl_flights)
    print(f"[INFO] Saved Extra Tropical Atlantic flights to {extra_atl_flights_path}")
    np.save(extra_pac_flights_path,extra_pac_flights)
    print(f"[INFO] Saved Extra Tropical Pacific flights to {extra_pac_flights_path}")
    np.save(indian_flights_path,indian_flights)
    print(f"[INFO] Saved Indian flights to {indian_flights_path}")
    np.save(trop_pac_flights_path,trop_pac_flights)
    print(f"[INFO] Saved Tropical Pacific flights to {trop_pac_flights_path}")
    np.save(SO_flights_path,SO_flights)
    print(f"[INFO] Saved Southern Ocean flights to {SO_flights_path}")
    return (trop_atl_flights_path,extra_atl_flights_path,extra_pac_flights_path,indian_flights_path,trop_pac_flights_path,SO_flights_path)

def get_loon_basin_data(basin:str):
    path_name = get_cache_dir() / f"{basin}_flights_flux.npy"
    if not path_name.exists():
        print(f"[INFO] Basin data for '{basin}' not found in cache. Generating basin data...")
        save_loon_basins()
    return path_name