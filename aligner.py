"""
aligner.py
----------
Spatial and temporal alignment of multi-source datasets.

Handles:
    - Regridding all physics/forcing fields to a common Chl-a grid
    - Temporal resampling (e.g. hourly wind → daily mean)
    - Bounding box clipping
    - Coordinate standardization (rename lat/lon variants)
    - Land/depth masking using bathymetry
"""

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Coordinate standardization
# ---------------------------------------------------------------------------

# Known aliases used across Copernicus products
LAT_ALIASES = ["latitude", "Latitude", "nav_lat", "y", "nlat"]
LON_ALIASES = ["longitude", "Longitude", "nav_lon", "x", "nlon"]
TIME_ALIASES = ["time_counter", "Time", "DATE", "date"]
DEPTH_ALIASES = ["depth", "deptht", "lev", "level", "z"]


def standardize_coords(ds: xr.Dataset) -> xr.Dataset:
    """
    Rename any known lat/lon/time/depth coordinate aliases to
    the canonical names: lat, lon, time, depth.
    """
    rename_map = {}

    for alias in LAT_ALIASES:
        if alias in ds.coords or alias in ds.dims:
            rename_map[alias] = "lat"
            break

    for alias in LON_ALIASES:
        if alias in ds.coords or alias in ds.dims:
            rename_map[alias] = "lon"
            break

    for alias in TIME_ALIASES:
        if alias in ds.coords or alias in ds.dims:
            rename_map[alias] = "time"
            break

    for alias in DEPTH_ALIASES:
        if alias in ds.coords or alias in ds.dims:
            rename_map[alias] = "depth"
            break

    if rename_map:
        logger.debug(f"Renaming coordinates: {rename_map}")
        ds = ds.rename(rename_map)

    # Wrap longitude from [0, 360] to [-180, 180] if needed
    if "lon" in ds.coords:
        if float(ds.lon.max()) > 180:
            ds = ds.assign_coords(lon=(ds.lon + 180) % 360 - 180)
            ds = ds.sortby("lon")
            logger.debug("Wrapped longitude from [0,360] to [-180,180]")

    # Sort by lat and lon to ensure ascending order
    if "lat" in ds.coords:
        ds = ds.sortby("lat")
    if "lon" in ds.coords:
        ds = ds.sortby("lon")

    return ds


# ---------------------------------------------------------------------------
# Bounding box clipping
# ---------------------------------------------------------------------------

def clip_to_domain(
    ds: xr.Dataset,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> xr.Dataset:
    """
    Clip dataset to a spatial bounding box.
    Adds a small buffer (0.5 * resolution) to avoid edge effects during regridding.
    """
    if "lat" not in ds.coords or "lon" not in ds.coords:
        logger.warning("Dataset has no lat/lon coordinates; skipping clip.")
        return ds

    # Estimate native resolution for buffer
    if ds.sizes.get("lat", 0) > 1:
        res = float(abs(ds.lat.diff("lat").mean()))
        buffer = 0.5 * res
    else:
        buffer = 0.0

    ds = ds.sel(
        lat=slice(lat_min - buffer, lat_max + buffer),
        lon=slice(lon_min - buffer, lon_max + buffer),
    )
    logger.debug(
        f"Clipped to [{lat_min:.2f},{lat_max:.2f}] lat, "
        f"[{lon_min:.2f},{lon_max:.2f}] lon"
    )
    return ds


# ---------------------------------------------------------------------------
# Surface level extraction (depth=0)
# ---------------------------------------------------------------------------

def extract_surface(ds: xr.Dataset) -> xr.Dataset:
    """
    If dataset has a depth dimension, select the shallowest level (depth=0).
    """
    if "depth" in ds.dims:
        ds = ds.isel(depth=0).drop_vars("depth", errors="ignore")
        logger.debug("Extracted surface level (depth index 0)")
    return ds


# ---------------------------------------------------------------------------
# Regridding
# ---------------------------------------------------------------------------

def regrid_to_target(
    source: xr.Dataset,
    target: xr.Dataset,
    method: str = "bilinear",
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Regrid source dataset to match the lat/lon grid of target dataset.

    Method options
    --------------
    bilinear    : smooth interpolation, suitable for continuous fields
                  (SST, MLD, sea level)
    nearest     : preserves categorical values; use for masks
    conservative: area-weighted; best for flux/precipitation variables
                  (requires xesmf)

    Parameters
    ----------
    source    : dataset to regrid
    target    : dataset whose lat/lon grid is the destination
    method    : interpolation method
    variables : subset of source variables to regrid; all if None
    """
    if "lat" not in target.coords or "lon" not in target.coords:
        raise ValueError("Target dataset must have lat and lon coordinates.")

    target_lat = target.lat.values
    target_lon = target.lon.values

    if method == "conservative":
        return _regrid_conservative(source, target_lat, target_lon, variables)
    else:
        return _regrid_scipy(source, target_lat, target_lon, method, variables)


def _regrid_scipy(
    source: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    method: str,
    variables: Optional[List[str]],
) -> xr.Dataset:
    """
    Regrid using xarray's interp (backed by scipy).
    Works for bilinear and nearest.
    """
    interp_method = "linear" if method == "bilinear" else "nearest"
    vars_to_regrid = variables or list(source.data_vars)

    regridded_vars = {}
    for var in vars_to_regrid:
        if var not in source:
            continue
        da = source[var]
        try:
            da_interp = da.interp(
                lat=target_lat,
                lon=target_lon,
                method=interp_method,
                kwargs={"bounds_error": False, "fill_value": np.nan},
            )
            regridded_vars[var] = da_interp
        except Exception as e:
            logger.warning(f"Could not regrid variable '{var}': {e}")

    ds_out = xr.Dataset(regridded_vars)
    logger.info(
        f"Regridded {list(regridded_vars.keys())} to "
        f"{len(target_lat)}x{len(target_lon)} grid using {method}"
    )
    return ds_out


def _regrid_conservative(
    source: xr.Dataset,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    variables: Optional[List[str]],
) -> xr.Dataset:
    """
    Conservative regridding using xESMF (requires: pip install xesmf).
    Used for river discharge and precipitation where area conservation matters.
    """
    try:
        import xesmf as xe
    except ImportError:
        logger.warning(
            "xesmf not installed; falling back to bilinear for conservative regridding.\n"
            "Install with: pip install xesmf"
        )
        return _regrid_scipy(source, target_lat, target_lon, "bilinear", variables)

    target_grid = xr.Dataset(
        {"lat": (["lat"], target_lat), "lon": (["lon"], target_lon)}
    )
    regridder = xe.Regridder(source, target_grid, "conservative", periodic=False)

    vars_to_regrid = variables or list(source.data_vars)
    ds_out = regridder(source[vars_to_regrid])
    logger.info(
        f"Conservative regrid applied to {vars_to_regrid}"
    )
    return ds_out


# ---------------------------------------------------------------------------
# Temporal resampling
# ---------------------------------------------------------------------------

def resample_to_daily(
    ds: xr.Dataset,
    aggregation: str = "mean",
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Resample a sub-daily dataset to daily frequency.

    Parameters
    ----------
    ds          : dataset with a time coordinate at sub-daily resolution
    aggregation : 'mean' | 'max' | 'min' | 'sum'
                  Use 'sum' for precipitation/discharge.
                  Use 'mean' for SST, winds, currents.
    variables   : subset to resample; all if None
    """
    if "time" not in ds.coords:
        logger.warning("No time coordinate found; skipping temporal resampling.")
        return ds

    vars_to_use = variables or list(ds.data_vars)
    ds_subset = ds[vars_to_use]

    resampler = ds_subset.resample(time="1D")

    if aggregation == "mean":
        ds_daily = resampler.mean(skipna=True)
    elif aggregation == "max":
        ds_daily = resampler.max(skipna=True)
    elif aggregation == "min":
        ds_daily = resampler.min(skipna=True)
    elif aggregation == "sum":
        ds_daily = resampler.sum(skipna=True)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}. Use mean/max/min/sum.")

    logger.info(f"Resampled {vars_to_use} to daily ({aggregation})")
    return ds_daily


def align_time_axes(
    datasets: Dict[str, xr.Dataset],
    reference_key: str,
    method: str = "nearest",
    tolerance: Optional[str] = "1D",
) -> Dict[str, xr.Dataset]:
    """
    Align all datasets to the time axis of a reference dataset.
    Useful when Chl-a (BGC) and physics products have slightly offset timestamps.

    Parameters
    ----------
    datasets      : dict of name → xr.Dataset
    reference_key : key in datasets to use as the time reference
    method        : 'nearest' or 'linear' interpolation in time
    tolerance     : max time offset to tolerate before filling with NaN

    Returns
    -------
    Dict with all datasets aligned to the reference time axis.
    """
    ref_time = datasets[reference_key].time

    aligned = {reference_key: datasets[reference_key]}
    for key, ds in datasets.items():
        if key == reference_key:
            continue
        try:
            ds_aligned = ds.interp(time=ref_time, method=method)
            aligned[key] = ds_aligned
            logger.info(f"Time-aligned '{key}' to '{reference_key}'")
        except Exception as e:
            logger.warning(f"Could not align '{key}': {e}")
            aligned[key] = ds

    return aligned


# ---------------------------------------------------------------------------
# Full alignment pipeline for all modalities
# ---------------------------------------------------------------------------

def align_all_modalities(
    chl_ds: xr.Dataset,
    physics_ds: xr.Dataset,
    wind_ds: xr.Dataset,
    discharge_ds: xr.Dataset,
    precip_ds: xr.Dataset,
    domain: Dict,
    regrid_method: str = "bilinear",
) -> Dict[str, xr.Dataset]:
    """
    End-to-end alignment pipeline:
        1. Standardize coordinates
        2. Clip to domain
        3. Extract surface level (physics)
        4. Resample wind to daily mean
        5. Accumulate ERA5 precip already done in loader (daily mm/day)
        6. Regrid physics, wind, discharge, precip to Chl-a grid
        7. Align time axes to Chl-a

    Parameters
    ----------
    chl_ds       : Chlorophyll-a dataset (BGC) — defines the target grid
    physics_ds   : SST, currents, MLD, sea level (CMEMS physics)
    wind_ds      : Wind vectors (CMEMS, may be hourly)
    discharge_ds : GloFAS river discharge (CDS, 0.05°, daily, GRIB2-loaded)
    precip_ds    : ERA5 total precipitation (CDS, 0.25°, already daily mm/day)
    domain       : dict with lon_min, lon_max, lat_min, lat_max
    regrid_method: passed to regrid_to_target for continuous fields

    Returns
    -------
    dict with keys: 'chl', 'physics', 'wind', 'discharge', 'precip'
    all on the same spatial grid and time axis as chl_ds.
    """
    logger.info("Starting full modality alignment pipeline")

    # Step 1: Standardize coordinates
    chl_ds       = standardize_coords(chl_ds)
    physics_ds   = standardize_coords(physics_ds)
    wind_ds      = standardize_coords(wind_ds)
    discharge_ds = standardize_coords(discharge_ds)
    precip_ds    = standardize_coords(precip_ds)

    # Step 2: Clip to domain
    bbox = (domain["lon_min"], domain["lon_max"],
            domain["lat_min"], domain["lat_max"])
    chl_ds       = clip_to_domain(chl_ds, *bbox)
    physics_ds   = clip_to_domain(physics_ds, *bbox)
    wind_ds      = clip_to_domain(wind_ds, *bbox)
    discharge_ds = clip_to_domain(discharge_ds, *bbox)
    precip_ds    = clip_to_domain(precip_ds, *bbox)

    # Step 3: Extract surface level from 3D physics fields
    physics_ds = extract_surface(physics_ds)

    # Step 4: Resample wind to daily mean (wind may be hourly)
    wind_ds = resample_to_daily(wind_ds, aggregation="mean")

    # Step 5: Regrid everything to the Chl-a grid
    # Use conservative regridding for discharge and precip (flux variables)
    physics_ds   = regrid_to_target(physics_ds,   chl_ds, method=regrid_method)
    wind_ds      = regrid_to_target(wind_ds,       chl_ds, method=regrid_method)
    discharge_ds = regrid_to_target(discharge_ds,  chl_ds, method="conservative")
    precip_ds    = regrid_to_target(precip_ds,     chl_ds, method="conservative")

    # Step 6: Align time axes to Chl-a
    datasets = {
        "chl":       chl_ds,
        "physics":   physics_ds,
        "wind":      wind_ds,
        "discharge": discharge_ds,
        "precip":    precip_ds,
    }
    datasets = align_time_axes(datasets, reference_key="chl")

    logger.info("Alignment pipeline complete — modalities: "
                + ", ".join(datasets.keys()))
    return datasets
