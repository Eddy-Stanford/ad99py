from typing import Optional
import xarray as xr
import os
from functools import cache
from ._data import get_loon_nc_mask_path


@cache
def list_basins(**kwargs):
    mask = load_mask(**kwargs)
    return [str(k) for k in mask.data_vars.keys()]

def default_mask_path(dir=None,name=None)->str:
    if dir is None or name is None:
        return get_loon_nc_mask_path()
    return os.path.join(dir,name)


def load_mask(path:Optional[str]=None,recentering:bool=True,**kwargs)->xr.Dataset:
    if path is None:
        path = default_mask_path(**kwargs)
    ds_mask = xr.open_dataset(path)
    if recentering:
        # map longitudes to be in range 0-360
        ds_mask['lon']= (ds_mask.lon + 360)%360
    return ds_mask


def mask_dataset(ds:xr.Dataset,basins=None,mask:Optional[xr.Dataset]=None,**kwargs):
    if mask is None:
        mask = load_mask(**kwargs)
    if 'latitude' in ds.variables and 'longitude' in ds.variables:
        # we assume either/or for 'latitude'/'longitude' or 'lat'/'lon
        mask = mask.rename(lat='latitude',lon='longitude')
        interp_mask = mask.interp(latitude=ds.latitude,longitude=ds.longitude,method='nearest')
        if basins is None:
            total_mask = sum(interp_mask[d] for d in interp_mask.data_vars)
        else:
            try:
                total_mask = sum(interp_mask[d] for d in basins)
            except KeyError as e:
                raise KeyError(f"No basin named {e.args[0]}. Allowed basins are {set(interp_mask.data_vars.keys())}") from e
                
        masked = ds.where(total_mask).stack(points=['latitude','longitude']).dropna('points',how='all')
    elif 'lat' in ds.variables and 'lon' in ds.variables:
        interp_mask = mask.interp(lat=ds.lat,lon=ds.lon,method='nearest')
        if basins is None:
            total_mask = sum(interp_mask[d] for d in interp_mask.data_vars)
        else:
            try:
                total_mask = sum(interp_mask[d] for d in basins)
            except KeyError as e:
                raise KeyError(f"No basin named {e.args[0]}. Allowed basins are {set(interp_mask.data_vars.keys())}") from e
        masked = ds.where(total_mask).stack(points=['lat','lon']).dropna('points',how='all')
    else:
        raise KeyError("Dataset must have 'latitude'/'longitude' or 'lat'/'lon' coordinates.")
    masked = masked.reset_index('points')
    return masked 
   
