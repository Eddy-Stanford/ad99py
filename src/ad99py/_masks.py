import numpy as np
import xarray as xr

NLAT_GRID = 36  # 5deg, 90N to 90S
NLON_GRID = 36  # 10deg, 180W to 180E


def get_mask_grid():
    lat_grid = 5 * np.arange(NLAT_GRID) - 87.5
    lon_grid = 10 * np.arange(NLON_GRID) - 175
    return lat_grid, lon_grid


def get_mask_grid_cellnumbers():
    grid_cell_numbers = np.arange(NLAT_GRID * NLON_GRID).reshape(NLAT_GRID, NLON_GRID)
    grid_cell_numbers = grid_cell_numbers.astype(float)
    return grid_cell_numbers


def get_xarray_mask():
    masks = get_numpy_mask()
    lat_grid, lon_grid = get_mask_grid()

    tropical_atlantic_mask = xr.DataArray(
        masks == 1, coords={"lat": lat_grid, "lon": lon_grid}, dims=["lat", "lon"]
    )
    extratropical_atlantic_mask = xr.DataArray(
        masks == 2, coords={"lat": lat_grid, "lon": lon_grid}, dims=["lat", "lon"]
    )
    extratropical_pacific_mask = xr.DataArray(
        masks == 3, coords={"lat": lat_grid, "lon": lon_grid}, dims=["lat", "lon"]
    )
    indian_mask = xr.DataArray(
        masks == 4, coords={"lat": lat_grid, "lon": lon_grid}, dims=["lat", "lon"]
    )
    tropical_pacific_mask = xr.DataArray(
        masks == 5, coords={"lat": lat_grid, "lon": lon_grid}, dims=["lat", "lon"]
    )
    southern_ocean_mask = xr.DataArray(
        masks == 6, coords={"lat": lat_grid, "lon": lon_grid}, dims=["lat", "lon"]
    )
    masks_ds = xr.Dataset(
        {
            "tropical_atlantic": tropical_atlantic_mask,
            "extratropical_atlantic": extratropical_atlantic_mask,
            "extratropical_pacific": extratropical_pacific_mask,
            "indian": indian_mask,
            "tropical_pacific": tropical_pacific_mask,
            "southern_ocean": southern_ocean_mask,
        }
    )
    return masks_ds


def get_numpy_mask():
    grid_cell_numbers = get_mask_grid_cellnumbers()
    masks = np.zeros(shape=np.shape(grid_cell_numbers))
    masks[16, 15:19] = 1
    masks[17, 14:19] = 1
    masks[18, 13:19] = 1
    masks[19, 12:17] = 1
    masks[20, 12:16] = 1

    # Extratropical Atlantic
    masks[22, 11:16] = 2
    masks[23, 10:16] = 2
    masks[24, 10:17] = 2
    masks[25, 11:17] = 2
    masks[26, 12:17] = 2
    masks[27, 13:17] = 2

    # Extratropical Pacific
    masks[22, :2] = 3
    masks[22, 3:7] = 3
    masks[23, :6] = 3
    masks[24, :6] = 3
    masks[25, :5] = 3
    masks[26, :5] = 3
    masks[27, :5] = 3
    masks[22, -5:] = 3
    masks[23, -5:] = 3
    masks[24, -4:] = 3
    masks[25, -3:] = 3
    masks[26, -3:] = 3
    masks[27, -3:] = 3

    # Orography mask(for the Southern Ocean data)
    orog_mask = np.zeros(shape=np.shape(grid_cell_numbers))
    # South America
    orog_mask[7:11, 10:12] = 1
    orog_mask[10, 12] = 1
    # Islands
    orog_mask[7, 12] = 1
    orog_mask[6, 14] = 1
    orog_mask[7, 14] = 1
    orog_mask[7, 25] = 1
    orog_mask[8, 24:26] = 1
    # Tazmania, Australia, New Zealand
    orog_mask[9, 32] = 1
    orog_mask[10, 31:34] = 1
    orog_mask[8:11, 35] = 1
    orog_mask[8, 34] = 1
    # Antarctica
    orog_mask[5, 11:13] = 1
    # Eliminate the Southern Ocean data over orography
    orog_cell_numbers = grid_cell_numbers[orog_mask == 1]

    # For plotting a map of the masks
    # Indian Ocean:
    masks[14, 23:30] = 4
    masks[15, 23:30] = 4
    masks[16, 22:28] = 4
    masks[17, 23:28] = 4
    masks[18, 23:27] = 4
    masks[19, 23:25] = 4
    masks[19, 26:27] = 4
    masks[20, 23:25] = 4
    masks[20, 26:27] = 4
    # Tropical Pacific:
    masks[16, 34:] = 5
    masks[17, 34:] = 5
    masks[18, 31:] = 5
    masks[19, 31:] = 5
    masks[20, 31:] = 5
    masks[16, :10] = 5
    masks[17, :10] = 5
    masks[18, :10] = 5
    masks[19, :9] = 5
    masks[20, :9] = 5
    # Southern Ocean:
    masks[10, :] = 6
    masks[9, :] = 6
    masks[8, :] = 6
    masks[7, :] = 6
    masks[6, :] = 6
    masks[5, :] = 6
    masks[orog_mask == 1] = 0
    return masks
