import numpy as np
import pandas as pd
from scipy import interpolate
from ._masks import get_mask_grid, get_mask_grid_cellnumbers, NLAT_GRID, NLON_GRID

LOON_DTYPES = {
    "flight_id": str,
    "time": str,
    "latitude": np.float64,
    "longitude": np.float64,
    "altitude": np.float64,
    "pressure": np.float64,
    "wind_u": np.float64,
    "wind_v": np.float64,
    "T_cosmic": np.float64,
    "dTdz_cosmic": np.float64,
    "N_cosmic": np.float64,
    "segment_id": np.int64,
    "u_smooth": np.float64,
    "v_smooth": np.float64,
    "flux_east": np.float64,
    "flux_west": np.float64,
    "flux_north": np.float64,
    "flux_south": np.float64,
    "flux_east_HF": np.float64,
    "flux_west_HF": np.float64,
    "flux_north_HF": np.float64,
    "flux_south_HF": np.float64,
    "flux_east_MF": np.float64,
    "flux_west_MF": np.float64,
    "flux_north_MF": np.float64,
    "flux_south_MF": np.float64,
    "flux_east_LF": np.float64,
    "flux_west_LF": np.float64,
    "flux_north_LF": np.float64,
    "flux_south_LF": np.float64,
}


def flights_to_numpy_array(flight):
    return (
        np.stack(
            [
                flight.flux_west.values,
                flight.flux_east.values,
                flight.flux_south.values,
                flight.flux_north.values,
            ]
        )
        * 1000
    )


def dt_start_end(segment):
    """
    Chop off the first and last two hours of each segment
    """
    segment["dt_start"] = (segment.time - segment.time.iloc[0]) / pd.Timedelta(
        1, "hours"
    )
    segment["dt_end"] = (segment.time.iloc[-1] - segment.time) / pd.Timedelta(
        1, "hours"
    )

    return segment


def process_flights(flight_path, masks):
    print(
        "[INFO] Processing Loon flight data and applying masks for all basins. This may take some time..."
    )
    flights = pd.read_csv(
        flight_path,
        parse_dates=["time"],
        infer_datetime_format=True,
        dtype=LOON_DTYPES,
        usecols=LOON_DTYPES.keys(),
    )
    flights["time"] = pd.to_datetime(flights["time"], format="ISO8601")
    flights["month"] = flights.time.apply(lambda x: x.month)
    flights["year"] = flights.time.apply(lambda x: x.year)

    # Create a cell "number" for each grid cell, essentially making each cell its own region
    grid_cell_numbers = get_mask_grid_cellnumbers()
    lat_grid, lon_grid = get_mask_grid()

    # Make a list of each grid cell's latitude and longitude
    [temp_lat, temp_lon] = np.meshgrid(lat_grid, lon_grid)
    temp_lat = np.reshape(temp_lat, NLAT_GRID * NLON_GRID)
    temp_lon = np.reshape(temp_lon, NLAT_GRID * NLON_GRID)

    # Apply a cell number to each GW packet
    data_lat = np.array(flights.latitude)
    data_lon = np.array(flights.longitude)
    temp = np.reshape(np.transpose(grid_cell_numbers), NLAT_GRID * NLON_GRID)
    f = interpolate.NearestNDInterpolator((temp_lat, temp_lon), temp)
    # this is how 2D interpolation is done from a grid to a list of points:
    data_cellnum = [float(f(*p)) for p in zip(data_lat, data_lon)]
    data_cellnum = np.round(
        np.array(data_cellnum)
    )  # round to the region's integer value
    flights["grid_cell_number"] = data_cellnum

    flights = flights.groupby("segment_id").progress_apply(dt_start_end)
    flights = flights[flights.dt_start > 2]
    flights = flights[flights.dt_end > 2]

    flights = delete_depressureizations(flights)

    # Select the flights
    trop_atl_flights = flights[
        flights.grid_cell_number.isin(grid_cell_numbers[masks == 1])
    ].copy()
    extra_atl_flights = flights[
        flights.grid_cell_number.isin(grid_cell_numbers[masks == 2])
    ].copy()
    extra_pac_flights = flights[
        flights.grid_cell_number.isin(grid_cell_numbers[masks == 3])
    ].copy()
    indian_flights = flights[
        flights.grid_cell_number.isin(grid_cell_numbers[masks == 4])
    ].copy()
    trop_pac_flights = flights[
        flights.grid_cell_number.isin(grid_cell_numbers[masks == 5])
    ].copy()
    SO_flights = flights[
        flights.grid_cell_number.isin(grid_cell_numbers[masks == 6])
    ].copy()

    return (
        flights_to_numpy_array(trop_atl_flights),
        flights_to_numpy_array(extra_atl_flights),
        flights_to_numpy_array(extra_pac_flights),
        flights_to_numpy_array(indian_flights),
        flights_to_numpy_array(trop_pac_flights),
        flights_to_numpy_array(SO_flights),
    )


def delete_depressureizations(flights):
    # Delete depressurizations
    flights = flights.copy()
    # My approach, based on running a few depressurizations through the wavelet analysis, is to delete
    # both the depressurization and the data within 2 hours (to the nearsest hour, rounding up) or the
    # width of the depressurization, whichever is bigger, on either side of it.
    flights["depressurization"] = 0

    # Tropical Pacific
    # Segment 284 (Flight I-195)
    flights.loc[
        (flights.flight_id == "I-195")
        & (flights.time >= np.datetime64("2014-07-09 07:00:00"))
        & (flights.time <= np.datetime64("2014-07-09 15:00:00")),
        "depressurization",
    ] = 1

    # Segment 7969 (Flight I-488)
    flights.loc[
        (flights.flight_id == "I-488")
        & (flights.time >= np.datetime64("2015-05-21 18:00:00"))
        & (flights.time <= np.datetime64("2015-05-22 10:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "I-488")
        & (flights.time >= np.datetime64("2015-05-23 04:00:00"))
        & (flights.time <= np.datetime64("2015-05-23 16:00:00")),
        "depressurization",
    ] = 1

    # Segment 8041 (Flight I-490)
    flights.loc[
        (flights.flight_id == "I-490")
        & (flights.time >= np.datetime64("2015-06-24 07:00:00"))
        & (flights.time <= np.datetime64("2015-06-24 13:00:00")),
        "depressurization",
    ] = 1

    # Tropical Atlantic
    # Segment 6506 (Flight I-436)
    flights.loc[
        (flights.flight_id == "I-436")
        & (flights.time >= np.datetime64("2015-04-09 01:00:00"))
        & (flights.time <= np.datetime64("2015-04-09 10:00:00")),
        "depressurization",
    ] = 1

    # Segment 8957 (Flight NR-215)
    flights.loc[
        (flights.flight_id == "NR-215")
        & (flights.time >= np.datetime64("2016-02-19 00:00:00"))
        & (flights.time <= np.datetime64("2016-02-19 13:00:00")),
        "depressurization",
    ] = 1

    # Indian Ocean
    # Segment 58 (Flight L-002)
    flights.loc[
        (flights.flight_id == "L-002")
        & (flights.time >= np.datetime64("2014-02-15 18:00:00"))
        & (flights.time <= np.datetime64("2014-02-16 04:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "L-002")
        & (flights.time >= np.datetime64("2014-02-20 04:00:00"))
        & (flights.time <= np.datetime64("2014-02-21 08:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "L-002")
        & (flights.time >= np.datetime64("2014-02-26 20:00:00"))
        & (flights.time <= np.datetime64("2014-02-27 01:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "L-002")
        & (flights.time >= np.datetime64("2014-02-28 10:00:00"))
        & (flights.time <= np.datetime64("2014-03-01 04:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "L-002")
        & (flights.time >= np.datetime64("2014-03-03 04:00:00"))
        & (flights.time <= np.datetime64("2014-03-04 06:00:00")),
        "depressurization",
    ] = 1

    # Segment 5672 (Flight I-361)
    flights.loc[
        (flights.flight_id == "I-361")
        & (flights.time >= np.datetime64("2015-01-26 09:00:00"))
        & (flights.time <= np.datetime64("2015-01-27 09:00:00")),
        "depressurization",
    ] = 1

    # Extratropical Pacific
    # Segment 5782 (Flight I-379)
    flights.loc[
        (flights.flight_id == "I-379")
        & (flights.time >= np.datetime64("2014-09-22 02:00:00"))
        & (flights.time <= np.datetime64("2014-09-22 08:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "I-379")
        & (flights.time >= np.datetime64("2014-09-22 09:00:00"))
        & (flights.time <= np.datetime64("2014-09-22 18:00:00")),
        "depressurization",
    ] = 1

    # Extratropical Atlantic
    # Segment 41 (Flight I-126)
    flights.loc[
        (flights.flight_id == "I-126")
        & (flights.time >= np.datetime64("2013-12-23 14:00:00"))
        & (flights.time <= np.datetime64("2013-12-24 03:00:00")),
        "depressurization",
    ] = 1

    # Southern Ocean
    # Segment 543 (Flight I-208)
    flights.loc[
        (flights.flight_id == "I-208")
        & (flights.time >= np.datetime64("2014-08-07 23:00:00"))
        & (flights.time <= np.datetime64("2014-08-08 05:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "I-208")
        & (flights.time >= np.datetime64("2014-08-09 11:00:00"))
        & (flights.time <= np.datetime64("2014-08-10 00:00:00")),
        "depressurization",
    ] = 1

    # Segment 1303 (Flight I-248)
    flights.loc[
        (flights.flight_id == "I-248")
        & (flights.time >= np.datetime64("2014-07-31 22:00:00"))
        & (flights.time <= np.datetime64("2014-08-01 04:00:00")),
        "depressurization",
    ] = 1

    # Segment 1585 (Flight I-254)
    flights.loc[
        (flights.flight_id == "I-254")
        & (flights.time >= np.datetime64("2014-07-31 11:00:00"))
        & (flights.time <= np.datetime64("2014-07-31 17:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "I-254")
        & (flights.time >= np.datetime64("2014-08-01 06:00:00"))
        & (flights.time <= np.datetime64("2014-08-01 12:00:00")),
        "depressurization",
    ] = 1

    # Segment 1985 (Flight I-263)
    flights.loc[
        (flights.flight_id == "I-263")
        & (flights.time >= np.datetime64("2014-08-02 02:00:00"))
        & (flights.time <= np.datetime64("2014-08-02 21:00:00")),
        "depressurization",
    ] = 1

    # Segment 2141 (Flight I-267)
    flights.loc[
        (flights.flight_id == "I-267")
        & (flights.time >= np.datetime64("2014-09-18 12:00:00"))
        & (flights.time <= np.datetime64("2014-09-18 19:00:00")),
        "depressurization",
    ] = 1

    # Segment 3559 (Flight I-292)
    flights.loc[
        (flights.flight_id == "I-292")
        & (flights.time >= np.datetime64("2014-11-12 05:00:00"))
        & (flights.time <= np.datetime64("2014-11-13 06:00:00")),
        "depressurization",
    ] = 1

    # Segment 3816 (Flight I-295)
    flights.loc[
        (flights.flight_id == "I-295")
        & (flights.time >= np.datetime64("2014-12-01 06:00:00"))
        & (flights.time <= np.datetime64("2014-12-01 13:00:00")),
        "depressurization",
    ] = 1

    # Segment 3990 (Flight I-296)
    flights.loc[
        (flights.flight_id == "I-296")
        & (flights.time >= np.datetime64("2014-12-26 08:00:00"))
        & (flights.time <= np.datetime64("2014-12-26 14:00:00")),
        "depressurization",
    ] = 1

    # Segment 4060 (Flight I-298)
    flights.loc[
        (flights.flight_id == "I-298")
        & (flights.time >= np.datetime64("2014-07-30 03:00:00"))
        & (flights.time <= np.datetime64("2014-07-30 10:00:00")),
        "depressurization",
    ] = 1

    # Segment 4623 (Flight I-320)
    flights.loc[
        (flights.flight_id == "I-320")
        & (flights.time >= np.datetime64("2014-11-02 05:00:00"))
        & (flights.time <= np.datetime64("2014-11-02 19:00:00")),
        "depressurization",
    ] = 1

    # Segment 4637 (Flight I-320)
    flights.loc[
        (flights.flight_id == "I-320")
        & (flights.time >= np.datetime64("2014-11-29 11:00:00"))
        & (flights.time <= np.datetime64("2014-11-29 19:00:00")),
        "depressurization",
    ] = 1

    # Segment 5216 (Flight I-338)
    flights.loc[
        (flights.flight_id == "I-338")
        & (flights.time >= np.datetime64("2014-08-24 04:00:00"))
        & (flights.time <= np.datetime64("2014-08-24 17:00:00")),
        "depressurization",
    ] = 1

    # Segment 6884 (Flight M-051)
    flights.loc[
        (flights.flight_id == "M-051")
        & (flights.time >= np.datetime64("2015-01-25 22:00:00"))
        & (flights.time <= np.datetime64("2015-01-26 10:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "M-051")
        & (flights.time >= np.datetime64("2015-02-01 19:00:00"))
        & (flights.time <= np.datetime64("2015-02-02 10:00:00")),
        "depressurization",
    ] = 1

    # Segment 6885 (Flight M-051)
    flights.loc[
        (flights.flight_id == "M-051")
        & (flights.time >= np.datetime64("2015-02-05 12:00:00"))
        & (flights.time <= np.datetime64("2015-02-06 09:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "M-051")
        & (flights.time >= np.datetime64("2015-02-06 15:00:00"))
        & (flights.time <= np.datetime64("2015-02-07 08:00:00")),
        "depressurization",
    ] = 1

    # Segment 7020 (Flight M-054)
    flights.loc[
        (flights.flight_id == "M-054")
        & (flights.time >= np.datetime64("2015-03-31 20:00:00"))
        & (flights.time <= np.datetime64("2015-04-01 08:00:00")),
        "depressurization",
    ] = 1
    flights.loc[
        (flights.flight_id == "M-054")
        & (flights.time >= np.datetime64("2015-04-02 14:00:00"))
        & (flights.time <= np.datetime64("2015-04-02 22:00:00")),
        "depressurization",
    ] = 1

    # Delete the depressurizations
    flights = flights[flights.depressurization == 0]
    flights = flights.drop(columns="depressurization")
    return flights
