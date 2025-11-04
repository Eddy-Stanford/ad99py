import os 
import numpy as np
from .masks import list_basins
from ._data import get_loon_basin_data


def get_basin_flux_data_path(basin,path=None):
    if basin not in list_basins():
        raise ValueError(f"Basin '{basin}' not found in available basins: {list_basins()}")
    if path is None:
        data_path = get_loon_basin_data(basin)
    else:
        data_path = os.path.join(path,f"{basin}_flights_flux.npy")
    return data_path


def get_fluxes(basins=None,path=None):
    u_flux_ptv = []
    u_flux_ntv = []
    v_flux_ntv = []
    v_flux_ptv = []
    if not basins:
        basins = list_basins() 
    try: 
        paths = [get_basin_flux_data_path(b,path) for b in basins]
    except TypeError as e:
        raise TypeError("If argument not `None` then expected an iterable type of basins.") from e 
    if len(paths) == 0:
        raise FileNotFoundError(f"No flux files found in {path}.")
    for f in paths:
        flux = np.load(f)
        u_flux_ntv.append(flux[0])
        u_flux_ptv.append(flux[1])
        v_flux_ntv.append(flux[2])    
        v_flux_ptv.append(flux[3])

    return u_flux_ntv,u_flux_ptv,v_flux_ntv,v_flux_ptv

def build_dictionary(untv,uptv,vntv,vptv):
    ntv_u = np.concatenate(untv)
    ntv_v = np.concatenate(vntv)
    ptv_u = np.concatenate(uptv)
    ptv_v = np.concatenate(vptv)
    abs_u = np.abs(ntv_u) + np.abs(ptv_u)
    net_u = ntv_u + ptv_u 
    abs_v = np.abs(ntv_v) + np.abs(ptv_v)
    net_v = ntv_v + ptv_v
    tot = np.sqrt(net_u**2 + net_v**2)
    abs_f = abs_u + abs_v
    return {
        'u_flux_ntv':ntv_u[ntv_u < 0],
        'u_flux_ptv':ptv_u[ptv_u > 0],
        'v_flux_ntv':ntv_v[ntv_v < 0],
        'v_flux_ptv':ptv_v[ptv_v > 0],
        'u_flux_abs':abs_u[abs_u > 0],
        'v_flux_abs':abs_v[abs_v > 0],
        'u_flux_net':net_u,
        'v_flux_net':net_v,
        'net_flux':tot[tot>0],
        'abs_flux':abs_f[abs_f>0]
    }

def loon_data(basins=None,path=None):
    return build_dictionary(*get_fluxes(basins,path))