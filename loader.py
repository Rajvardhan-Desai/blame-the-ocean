"""
loader.py
---------
Handles loading data from all formats produced by Copernicus Marine Service:
    - NetCDF  (.nc)
    - HDF5    (.h5 / .he5 / .hdf5 / .hdf)
    - GeoTIFF (.tif / .tiff)
    - Zarr    (directory store)
    - GRIB    (.grib / .grib2) for GloFAS

All loaders return a standardized xarray.Dataset so every downstream
module works the same way regardless of source format.
"""

import glob
import logging
import os
from pathlib import Path
from typing import Callable, List, Optional, Union

import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------

def detect_format(path: Union[str, Path]) -> str:
    """
    Infer file format from extension or directory structure.

    Returns one of:
        'netcdf' | 'hdf5' | 'geotiff' | 'zarr' | 'grib' | 'unknown'
    """
    path = Path(path)

    if path.is_dir():
        if (path / ".zgroup").exists() or (path / ".zarray").exists():
            return "zarr"
        return "unknown"

    suffix = path.suffix.lower()
    if suffix in [".nc", ".nc4"]:
        return "netcdf"
    if suffix in [".h5", ".he5", ".hdf5", ".hdf"]:
        return "hdf5"
    if suffix in [".tif", ".tiff"]:
        return "geotiff"
    if suffix in [".grib", ".grib2", ".grb", ".grb2"]:
        return "grib"
    return "unknown"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_geographic_crs(da: xr.DataArray) -> bool:
    """
    Best-effort check whether raster coordinates are geographic lon/lat.
    """
    try:
        crs = da.rio.crs
        if crs is None:
            return False
        return bool(getattr(crs, "is_geographic", False))
    except Exception:
        return False


def _infer_dims(shape: tuple, coords: dict) -> tuple:
    """
    Guess dimension names from array shape by matching coordinate lengths.

    Avoids duplicate dimension names. Falls back to dim_0, dim_1, ...
    when ambiguous.
    """
    dims = []
    used = set()

    for i, size in enumerate(shape):
        matches = [
            name for name, values in coords.items()
            if len(values) == size and name not in used
        ]

        if len(matches) == 1:
            dim = matches[0]
            used.add(dim)
        else:
            dim = f"dim_{i}"

        dims.append(dim)

    return tuple(dims)


# ---------------------------------------------------------------------------
# Individual format loaders
# ---------------------------------------------------------------------------

def load_netcdf(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Load a NetCDF file into an xarray Dataset.

    Parameters
    ----------
    path      : path to the .nc file
    variables : subset of variables to load; loads all if None
    chunks    : dask chunking dict e.g. {"time": 10, "lat": 256, "lon": 256}
                Pass None for in-memory loading.
    """
    logger.info(f"Loading NetCDF: {path}")

    kwargs = {}
    if chunks:
        kwargs["chunks"] = chunks

    try:
        ds = xr.open_dataset(path, engine="netcdf4", **kwargs)
    except Exception:
        ds = xr.open_dataset(path, **kwargs)

    if variables:
        available = [v for v in variables if v in ds.data_vars]
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            logger.warning(f"Variables not found in {path}: {missing}")
        ds = ds[available]

    return ds


def load_hdf5(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    group: Optional[str] = None,
) -> xr.Dataset:
    """
    Load an HDF5 file into an xarray Dataset.

    Parameters
    ----------
    path      : path to the .h5 file
    variables : subset of variables to load
    group     : HDF5 group path (e.g. '/geophysical_data'); loads root if None
    """
    logger.info(f"Loading HDF5: {path}")

    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "h5py is required for HDF5 loading. pip install h5py"
        ) from e

    data_vars = {}
    coords = {}
    coord_keys_in_file = set()

    with h5py.File(path, "r") as f:
        root = f[group] if group else f

        # Extract common coordinate datasets and canonicalize names
        for coord_name in [
            "lat", "latitude", "Latitude",
            "lon", "longitude", "Longitude",
            "time", "Time",
        ]:
            if coord_name in root:
                arr = root[coord_name][:]

                if "lat" in coord_name.lower():
                    canonical = "lat"
                elif "lon" in coord_name.lower():
                    canonical = "lon"
                else:
                    canonical = "time"

                coords[canonical] = arr
                coord_keys_in_file.add(coord_name)

        target_vars = variables or list(root.keys())

        for var in target_vars:
            if var not in root:
                logger.warning(f"Variable '{var}' not found in HDF5 group '{group or '/'}'")
                continue

            if var in coord_keys_in_file:
                continue

            obj = root[var]
            if not hasattr(obj, "shape"):
                # Skip non-dataset objects
                continue

            arr = obj[:]
            dims = _infer_dims(arr.shape, coords)
            attrs = dict(obj.attrs) if hasattr(obj, "attrs") else {}
            data_vars[var] = xr.Variable(dims, arr, attrs=attrs)

    return xr.Dataset(data_vars=data_vars, coords=coords)


def load_geotiff(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Load a GeoTIFF into an xarray Dataset using rioxarray.
    Each band becomes a variable named by band index or provided name.

    Parameters
    ----------
    path      : path to the .tif file
    variables : optional list of variable names matching band order
    """
    logger.info(f"Loading GeoTIFF: {path}")

    try:
        import rioxarray
    except ImportError as e:
        raise ImportError(
            "rioxarray is required for GeoTIFF loading. pip install rioxarray"
        ) from e

    da = rioxarray.open_rasterio(path)  # shape: (band, y, x)

    # Only rename spatial dims to lon/lat if raster CRS is geographic
    if _is_geographic_crs(da):
        rename_map = {}
        if "x" in da.dims:
            rename_map["x"] = "lon"
        if "y" in da.dims:
            rename_map["y"] = "lat"
        da = da.rename(rename_map)

    n_bands = da.sizes["band"]

    if variables and len(variables) == n_bands:
        names = variables
    else:
        if variables and len(variables) != n_bands:
            logger.warning(
                f"Provided {len(variables)} variable names but raster has {n_bands} bands. "
                "Using default band names."
            )
        names = [f"band_{i + 1}" for i in range(n_bands)]

    data_vars = {}
    for i in range(n_bands):
        band_da = da.isel(band=i)
        if "band" in band_da.coords:
            band_da = band_da.drop_vars("band")
        data_vars[names[i]] = band_da

    return xr.Dataset(data_vars)


def load_zarr(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
) -> xr.Dataset:
    """
    Load a Zarr store into an xarray Dataset.
    """
    logger.info(f"Loading Zarr: {path}")

    kwargs = {}
    if chunks:
        kwargs["chunks"] = chunks

    try:
        ds = xr.open_zarr(str(path), consolidated=True, **kwargs)
    except Exception:
        ds = xr.open_zarr(str(path), consolidated=False, **kwargs)

    if variables:
        available = [v for v in variables if v in ds.data_vars]
        missing = [v for v in variables if v not in ds.data_vars]
        if missing:
            logger.warning(f"Variables not found in Zarr store {path}: {missing}")
        ds = ds[available]

    return ds


def load_glofas(path: Union[str, Path]) -> xr.Dataset:
    """
    Load a GloFAS GRIB2/GRIB file into an xarray Dataset.
    Renames latitude/longitude to lat/lon for pipeline consistency.

    Requires: pip install cfgrib eccodes
    """
    logger.info(f"Loading GloFAS GRIB: {path}")

    try:
        ds = xr.open_dataset(str(path), engine="cfgrib")
    except Exception as e:
        raise RuntimeError(
            f"Failed to open GloFAS file: {e}\n"
            "Install: pip install cfgrib eccodes"
        ) from e

    rename = {}
    if "latitude" in ds.coords:
        rename["latitude"] = "lat"
    if "longitude" in ds.coords:
        rename["longitude"] = "lon"

    if rename:
        ds = ds.rename(rename)

    return ds


# ---------------------------------------------------------------------------
# Unified entry point
# ---------------------------------------------------------------------------

def load(
    path: Union[str, Path],
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    hdf5_group: Optional[str] = None,
) -> xr.Dataset:
    """
    Auto-detect format and load any supported file into an xarray Dataset.

    Parameters
    ----------
    path       : file or directory path
    variables  : variables/bands to load (loads all if None)
    chunks     : dask chunk dict for lazy loading (NetCDF / Zarr)
    hdf5_group : HDF5 group path (only used for HDF5 files)
    """
    fmt = detect_format(path)
    logger.info(f"Detected format: {fmt} for {path}")

    if fmt == "netcdf":
        return load_netcdf(path, variables, chunks)
    if fmt == "hdf5":
        return load_hdf5(path, variables, hdf5_group)
    if fmt == "geotiff":
        return load_geotiff(path, variables)
    if fmt == "zarr":
        return load_zarr(path, variables, chunks)
    if fmt == "grib":
        return load_glofas(path)

    raise ValueError(
        f"Unsupported or unrecognized format for: {path}\n"
        "Supported: .nc, .nc4, .h5, .he5, .hdf5, .hdf, .tif, .tiff, "
        ".grib, .grib2, .grb, .grb2, zarr directory"
    )


# ---------------------------------------------------------------------------
# Multi-file loader (time-series of daily files)
# ---------------------------------------------------------------------------

def load_time_series(
    directory: Union[str, Path],
    pattern: str = "*.nc",
    variables: Optional[List[str]] = None,
    chunks: Optional[dict] = None,
    date_parser: Optional[Callable[[str], object]] = None,
    hdf5_group: Optional[str] = None,
    strict: bool = False,
) -> xr.Dataset:
    """
    Load a directory of daily files matching a glob pattern and
    concatenate them along the time dimension.

    Parameters
    ----------
    directory   : folder containing the daily files
    pattern     : glob pattern to match files (e.g. "chl_*.nc")
    variables   : variables to load from each file
    chunks      : dask chunk dict
    date_parser : optional function(filename: str) -> timestamp-like object
                  to extract timestamps from filenames when files lack
                  a time coordinate
    hdf5_group  : HDF5 group path for HDF5 files
    strict      : if True, raise immediately on a file load failure;
                  if False, skip bad files and continue

    Example
    -------
    ds = load_time_series(
        "data/raw/chl/",
        pattern="cmems_chl_*.nc",
        variables=["chl"],
        chunks={"lat": 256, "lon": 256},
    )
    """
    files = sorted(glob.glob(str(Path(directory) / pattern)))
    if not files:
        raise FileNotFoundError(
            f"No files matched pattern '{pattern}' in {directory}"
        )

    logger.info(f"Found {len(files)} files in {directory}")

    datasets = []

    for fpath in files:
        try:
            ds = load(
                fpath,
                variables=variables,
                chunks=chunks,
                hdf5_group=hdf5_group,
            )

            if "time" not in ds.coords:
                if date_parser is None:
                    raise ValueError(
                        f"{fpath} has no 'time' coordinate and no date_parser was provided"
                    )

                t = date_parser(Path(fpath).name)
                ds = ds.expand_dims("time").assign_coords(time=("time", [t]))

            datasets.append(ds)

        except Exception as e:
            if strict:
                raise
            logger.warning(f"Skipping {fpath}: {e}")

    if not datasets:
        raise RuntimeError("All files failed to load. Check your data directory.")

    combined = xr.concat(datasets, dim="time")
    combined = combined.sortby("time")

    logger.info(
        f"Combined dataset time range: "
        f"{combined.time.values[0]} to {combined.time.values[-1]}"
    )
    return combined


# ---------------------------------------------------------------------------
# Copernicus-specific loaders (wraps copernicusmarine toolbox)
# ---------------------------------------------------------------------------

def download_copernicus(
    dataset_id: str,
    variables: List[str],
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    depth_min: float = 0.0,
    depth_max: float = 1.0,
    output_dir: str = "data/raw",
    username: Optional[str] = None,
    password: Optional[str] = None,
    skip_depth: bool = False,
) -> Path:
    """
    Download a subset from Copernicus Marine Service.

    Parameters
    ----------
    skip_depth : if True, omit depth arguments entirely from the request.
                 Use this for BGC products (chl, pp) which are already
                 surface-integrated and have no depth dimension, or when
                 the dataset's minimum depth is unknown.
                 If False, pass depth_min/depth_max to get surface level only.
    """
    try:
        import copernicusmarine as cm
    except ImportError as e:
        raise ImportError(
            "copernicusmarine is required.\n"
            "Install: pip install copernicusmarine\n"
            "Auth:    copernicusmarine login"
        ) from e

    os.makedirs(output_dir, exist_ok=True)
    output_filename = f"{dataset_id}_{date_start}_{date_end}.nc"
    output_path = Path(output_dir) / output_filename

    if output_path.exists():
        logger.info(f"File already exists, skipping download: {output_path}")
        return output_path

    logger.info(f"Downloading {dataset_id} | {date_start} to {date_end}")

    subset_kwargs = dict(
        dataset_id=dataset_id,
        variables=variables,
        start_datetime=date_start,
        end_datetime=date_end,
        minimum_longitude=lon_min,
        maximum_longitude=lon_max,
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        output_filename=str(output_path),
        username=username,
        password=password,
    )

    if not skip_depth:
        subset_kwargs["minimum_depth"] = depth_min
        subset_kwargs["maximum_depth"] = depth_max

    cm.subset(**subset_kwargs)

    logger.info(f"Downloaded to: {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CDS downloaders — GloFAS river discharge and ERA5 precipitation
# These use cdsapi (different from copernicusmarine used for CMEMS products)
# ---------------------------------------------------------------------------

def download_glofas(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
    system_version: str = "version_4_0",
) -> Path:
    """
    Download GloFAS v4.0 river discharge reanalysis via cdsapi.

    Dataset : cems-glofas-historical
    Variable: river_discharge_in_the_last_24_hours
    Format  : GRIB2
    """
    try:
        import cdsapi
        import pandas as pd
    except ImportError as e:
        raise ImportError("pip install cdsapi pandas") from e

    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(output_dir) / f"glofas_{date_start}_{date_end}.grib2"

    if output_path.exists():
        logger.info(f"GloFAS file already exists, skipping: {output_path}")
        return output_path

    dates = pd.date_range(date_start, date_end, freq="D")
    years = sorted(set(str(d.year) for d in dates))
    months = sorted(set(f"{d.month:02d}" for d in dates))
    days = sorted(set(f"{d.day:02d}" for d in dates))

    logger.info(f"Downloading GloFAS discharge | {date_start} to {date_end}")

    client = cdsapi.Client(url="https://ewds.climate.copernicus.eu/api")
    client.retrieve(
        "cems-glofas-historical",
        {
            "system_version": [system_version],
            "hydrological_model": ["lisflood"],
            "product_type": ["consolidated"],
            "variable": ["river_discharge_in_the_last_24_hours"],
            "hyear": years,
            "hmonth": months,
            "hday": days,
            "data_format": "grib2",
            "download_format": "unarchived",
            "area": [lat_max, lon_min, lat_min, lon_max],  # N W S E
        },
    ).download(str(output_path))

    logger.info(f"GloFAS saved to: {output_path}")
    return output_path


def _extract_nc_from_zip(zip_path: Path, dest_path: Path) -> None:
    """
    Extract the first .nc file found inside a zip archive to dest_path.
    CDS sometimes returns a zip even when download_format=unarchived is set.
    """
    import zipfile

    with zipfile.ZipFile(str(zip_path), "r") as zf:
        nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
        if not nc_members:
            raise RuntimeError(
                f"No .nc file found inside zip archive: {zip_path}\n"
                f"Contents: {zf.namelist()}"
            )
        with zf.open(nc_members[0]) as src, open(str(dest_path), "wb") as dst:
            dst.write(src.read())


def download_era5_precipitation(
    date_start: str,
    date_end: str,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    output_dir: str = "data/raw",
) -> Path:
    """
    Download ERA5 total precipitation AND 10m wind components from CDS.

    Variables downloaded:
        total_precipitation       -> "tp"
        10m_u_component_of_wind   -> "u10"
        10m_v_component_of_wind   -> "v10"
    """
    try:
        import cdsapi
        import pandas as pd
        import zipfile
    except ImportError as e:
        raise ImportError("pip install cdsapi pandas") from e

    os.makedirs(output_dir, exist_ok=True)
    merged_path = Path(output_dir) / f"era5_precip_wind_{date_start}_{date_end}.nc"

    if merged_path.exists():
        logger.info(f"ERA5 file already exists, skipping: {merged_path}")
        return merged_path

    full_range = pd.date_range(date_start, date_end, freq="MS")
    year_months = [(ts.year, ts.month) for ts in full_range]
    end_ts = pd.Timestamp(date_end)
    if (end_ts.year, end_ts.month) not in year_months:
        year_months.append((end_ts.year, end_ts.month))

    client = cdsapi.Client()
    chunk_paths = []

    for year, month in year_months:
        chunk_path = Path(output_dir) / f"era5_precip_wind_{year}_{month:02d}.nc"

        if chunk_path.exists():
            try:
                ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
                ds_test.close()
                logger.info(f"ERA5 {year}-{month:02d} already exists, skipping.")
                chunk_paths.append(chunk_path)
                continue
            except Exception:
                logger.warning(
                    f"Existing file for {year}-{month:02d} is corrupt; re-downloading."
                )
                chunk_path.unlink()

        month_start = max(
            pd.Timestamp(f"{year}-{month:02d}-01"),
            pd.Timestamp(date_start),
        )
        month_end = min(
            pd.Timestamp(f"{year}-{month:02d}-01") + pd.offsets.MonthEnd(1),
            pd.Timestamp(date_end),
        )

        days = sorted(set(
            f"{d.day:02d}"
            for d in pd.date_range(month_start, month_end, freq="D")
        ))

        logger.info(
            f"Downloading ERA5 | {year}-{month:02d} "
            f"({month_start.date()} to {month_end.date()})"
        )

        tmp_path = chunk_path.with_suffix(".tmp")

        client.retrieve(
            "reanalysis-era5-single-levels",
            {
                "product_type": ["reanalysis"],
                "variable": [
                    "total_precipitation",
                    "10m_u_component_of_wind",
                    "10m_v_component_of_wind",
                ],
                "year": [str(year)],
                "month": [f"{month:02d}"],
                "day": days,
                "time": [f"{h:02d}:00" for h in range(24)],
                "area": [lat_max, lon_min, lat_min, lon_max],  # N W S E
                "data_format": "netcdf",
                "download_format": "unarchived",
                "grid": ["0.25", "0.25"],
            },
            str(tmp_path),
        )

        if zipfile.is_zipfile(str(tmp_path)):
            logger.info(
                f"ERA5 {year}-{month:02d}: CDS returned a zip archive; extracting NetCDF."
            )
            try:
                _extract_nc_from_zip(tmp_path, chunk_path)
            finally:
                tmp_path.unlink(missing_ok=True)
        else:
            tmp_path.rename(chunk_path)

        try:
            ds_test = xr.open_dataset(str(chunk_path), engine="netcdf4")
            ds_test.close()
        except Exception as e:
            chunk_path.unlink(missing_ok=True)
            raise RuntimeError(
                f"Downloaded file for {year}-{month:02d} failed validation: {e}\n"
                "The file has been deleted. Re-run to retry this month."
            ) from e

        logger.info(f"ERA5 {year}-{month:02d} saved to: {chunk_path}")
        chunk_paths.append(chunk_path)

    logger.info(f"Merging {len(chunk_paths)} monthly ERA5 files -> {merged_path}")

    datasets = []
    try:
        for p in chunk_paths:
            ds = xr.open_dataset(str(p), engine="netcdf4")
            datasets.append(ds.load())
            ds.close()

        ds_merged = xr.concat(datasets, dim="time").sortby("time")
        ds_merged.to_netcdf(str(merged_path))
        ds_merged.close()
    finally:
        for ds in datasets:
            try:
                ds.close()
            except Exception:
                pass

    for p in chunk_paths:
        try:
            p.unlink()
        except OSError:
            pass

    logger.info(f"ERA5 merged file saved to: {merged_path}")
    return merged_path


def accumulate_era5_precip_to_daily(ds: xr.Dataset, var: str = "tp") -> xr.Dataset:
    """
    Sum ERA5 hourly precipitation accumulations to daily totals and convert
    from metres to mm/day (* 1000).

    ERA5 'tp' = metres of water accumulated in each 1-hour step.
    Daily total (mm) = sum of 24 hourly values * 1000.
    """
    if var not in ds.data_vars:
        logger.warning(f"Variable '{var}' not found; skipping accumulation.")
        return ds

    daily = ds[var].resample(time="1D").sum(skipna=True) * 1000.0
    daily.attrs.update({
        "units": "mm/day",
        "long_name": "Total precipitation (daily sum)",
    })

    logger.info("ERA5 precipitation accumulated to daily mm/day")
    return ds.assign({var: daily})


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def print_dataset_summary(ds: xr.Dataset, label: str = "") -> None:
    """Print a compact summary of a dataset for quick inspection."""
    print(f"\n{'=' * 60}")
    if label:
        print(f"  {label}")
    print(f"{'=' * 60}")
    print(f"  Variables : {list(ds.data_vars)}")
    print(f"  Coords    : {list(ds.coords)}")

    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        print(f"  Time range: {ds.time.values[0]}  ->  {ds.time.values[-1]}")

    if "lat" in ds.coords:
        try:
            print(f"  Lat range : {float(ds.lat.min()):.2f}  ->  {float(ds.lat.max()):.2f}")
        except Exception:
            print("  Lat range : unavailable")

    if "lon" in ds.coords:
        try:
            print(f"  Lon range : {float(ds.lon.min()):.2f}  ->  {float(ds.lon.max()):.2f}")
        except Exception:
            print("  Lon range : unavailable")

    try:
        print(f"  Size      : {ds.nbytes / 1e6:.1f} MB (in memory)")
    except Exception:
        print("  Size      : unavailable")

    print()