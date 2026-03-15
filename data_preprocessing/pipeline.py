"""
pipeline.py
-----------
Main preprocessing pipeline for MM-MARAS.

Orchestrates all preprocessing steps in order:
    1. Download / load raw data
    2. Standardize coordinates
    3. Clip to spatial domain
    4. Temporal alignment
    5. Spatial regridding to common Chl-a grid
    6. Log-transform and normalization
    7. Mask generation (obs, land, bloom, MCAR, MNAR)
    8. Static context assembly
    9. Patch extraction and train/val/test split
   10. Save to disk

Run from the command line:
    python pipeline.py

Or import and call run_pipeline() programmatically.
"""

import sys
import os

# ---------------------------------------------------------------------------
# Path fix - must be before ALL local imports.
# Ensures masker.py, loader.py, config.py etc. are always found even when
# running from a different working directory on Windows.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr

import config as cfg
from aligner import align_all_modalities
from loader import (load, load_time_series, download_copernicus,
                    download_glofas, download_era5_precipitation,
                    load_glofas, accumulate_era5_precip_to_daily,
                    print_dataset_summary)
from masker import build_all_masks
from normalizer import compute_stats, save_stats, load_stats, normalize_dataset
from patcher import PatchExtractor, save_patches, temporal_split

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")


# ---------------------------------------------------------------------------
# Step 1: Data acquisition
# ---------------------------------------------------------------------------

def step_download(domain: dict) -> dict:
    """
    Download all required data from Copernicus Marine Service (CMEMS) and CDS.

    BGC Chl-a   → CMEMS multiyear reanalysis (cmems_mod_glo_bgc_my_0.25deg_P1D-m)
    Physics      → CMEMS multiyear reanalysis (cmems_mod_glo_phy_my_0.083deg_P1D-m)
    Wind + Precip→ ERA5 single-levels via CDS (one combined download)
    Discharge    → GloFAS v4.0 via CEMS/CDS

    Skips any file that already exists locally.
    Returns dict of paths to downloaded files.
    """
    logger.info("STEP 1: Downloading data")

    paths = {}

    # Chl-a — multiyear BGC reanalysis (variable: chl)
    # No depth subsetting — chl is already a surface-integrated product
    paths["chl"] = download_copernicus(
        dataset_id  = cfg.COPERNICUS_BGC_DATASET,
        variables   = cfg.BGC_VARIABLES,
        date_start  = cfg.DATE_START,
        date_end    = cfg.DATE_END,
        lon_min     = domain["lon_min"],
        lon_max     = domain["lon_max"],
        lat_min     = domain["lat_min"],
        lat_max     = domain["lat_max"],
        output_dir  = cfg.RAW_DATA_DIR,
        skip_depth  = True,
    )

    # Physics — multiyear reanalysis (thetao, uo, vo, mlotst, zos)
    # Select shallowest available level
    paths["physics"] = download_copernicus(
        dataset_id  = cfg.COPERNICUS_PHY_DATASET,
        variables   = cfg.PHY_VARIABLES,
        date_start  = cfg.DATE_START,
        date_end    = cfg.DATE_END,
        lon_min     = domain["lon_min"],
        lon_max     = domain["lon_max"],
        lat_min     = domain["lat_min"],
        lat_max     = domain["lat_max"],
        output_dir  = cfg.RAW_DATA_DIR,
        skip_depth  = False,   # physics fields are 3D; take surface level
    )

    # ERA5 precipitation + 10m wind (u10, v10) — single CDS request
    # wind replaces the old CMEMS hourly wind dataset
    paths["era5"] = download_era5_precipitation(
        date_start = cfg.DATE_START,
        date_end   = cfg.DATE_END,
        lon_min    = domain["lon_min"],
        lon_max    = domain["lon_max"],
        lat_min    = domain["lat_min"],
        lat_max    = domain["lat_max"],
        output_dir = cfg.RAW_DATA_DIR,
    )

    # GloFAS river discharge (CEMS — separate API endpoint from standard CDS)
    paths["discharge"] = download_glofas(
        date_start = cfg.DATE_START,
        date_end   = cfg.DATE_END,
        lon_min    = domain["lon_min"],
        lon_max    = domain["lon_max"],
        lat_min    = domain["lat_min"],
        lat_max    = domain["lat_max"],
        output_dir = cfg.RAW_DATA_DIR,
    )

    return paths


# ---------------------------------------------------------------------------
# Step 2: Load and align all modalities
# ---------------------------------------------------------------------------

def step_load_and_align(paths: dict, domain: dict) -> dict:
    """
    Load raw files and align all modalities to the Chl-a grid and time axis.
    Returns dict of aligned xr.Datasets.
    """
    logger.info("STEP 2: Loading and aligning modalities")

    chl_ds     = load(paths["chl"],     variables=cfg.BGC_VARIABLES)
    physics_ds = load(paths["physics"], variables=cfg.PHY_VARIABLES)

    # ERA5 file contains both wind (u10, v10) and precipitation (tp)
    era5_ds      = load(paths["era5"])
    wind_ds      = era5_ds[cfg.WIND_VARIABLES]   # u10, v10
    precip_ds    = era5_ds[["tp"]]

    # Accumulate ERA5 hourly precip → daily mm/day
    precip_ds    = accumulate_era5_precip_to_daily(precip_ds, var="tp")

    # GloFAS discharge (GRIB2)
    discharge_ds = load_glofas(paths["discharge"])

    print_dataset_summary(chl_ds,       label="Raw Chl-a (BGC multiyear)")
    print_dataset_summary(physics_ds,   label="Raw Physics (multiyear)")
    print_dataset_summary(wind_ds,      label="Raw Wind (ERA5 u10/v10)")
    print_dataset_summary(precip_ds,    label="Raw Precipitation (ERA5 tp)")
    print_dataset_summary(discharge_ds, label="Raw GloFAS Discharge")

    aligned = align_all_modalities(
        chl_ds       = chl_ds,
        physics_ds   = physics_ds,
        wind_ds      = wind_ds,
        discharge_ds = discharge_ds,
        precip_ds    = precip_ds,
        domain       = domain,
        regrid_method = cfg.REGRID_METHOD,
    )

    return aligned


# ---------------------------------------------------------------------------
# Step 3: Build masks
# ---------------------------------------------------------------------------

def step_build_masks(aligned: dict) -> xr.Dataset:
    """
    Generate all binary masks from the Chl-a DataArray.
    """
    logger.info("STEP 3: Building masks")

    chl_da = aligned["chl"]["chl"]  # raw Chl-a DataArray

    mask_ds = build_all_masks(
        chl            = chl_da,
        valid_min      = cfg.CHL_VALID_MIN,
        valid_max      = cfg.CHL_VALID_MAX,
        bloom_threshold= cfg.CHL_BLOOM_THRESH,
        mcar_threshold = cfg.MCAR_SPATIAL_CORR_THRESHOLD,
        mnar_threshold = cfg.MNAR_CHL_CORR_THRESHOLD,
    )

    return mask_ds


# ---------------------------------------------------------------------------
# Step 4: Normalize
# ---------------------------------------------------------------------------

def step_normalize(
    aligned: dict,
    mask_ds: xr.Dataset,
    stats_path: str,
    recompute_stats: bool = True,
) -> dict:
    """
    Compute normalization statistics from training data only,
    then normalize all datasets.

    Parameters
    ----------
    recompute_stats : if False, loads existing stats from stats_path.
                      Set to False at inference or during val/test preprocessing.
    """
    logger.info("STEP 4: Normalizing")

    if recompute_stats:
        # Compute stats on training time slice only
        T_total = aligned["chl"].sizes["time"]
        train_slice, _, _ = temporal_split(T_total)

        chl_train     = aligned["chl"].isel(time=train_slice)
        physics_train = aligned["physics"].isel(time=train_slice)
        wind_train    = aligned["wind"].isel(time=train_slice)

        obs_mask_train = mask_ds["obs_mask"].isel(time=train_slice)

        chl_stats = compute_stats(
            chl_train, cfg.BGC_VARIABLES,
            obs_mask=obs_mask_train,
        )
        phy_stats = compute_stats(
            physics_train, cfg.PHY_VARIABLES,
            obs_mask=None,
        )
        wind_stats = compute_stats(
            wind_train, cfg.WIND_VARIABLES,
            obs_mask=None,
        )

        all_stats = {**chl_stats, **phy_stats, **wind_stats}
        os.makedirs(cfg.STATS_DIR, exist_ok=True)
        save_stats(all_stats, stats_path)

    else:
        all_stats = load_stats(stats_path)

    # Apply normalization
    chl_norm     = normalize_dataset(aligned["chl"],     all_stats, cfg.BGC_VARIABLES)
    physics_norm = normalize_dataset(aligned["physics"], all_stats, cfg.PHY_VARIABLES)
    wind_norm    = normalize_dataset(aligned["wind"],    all_stats, cfg.WIND_VARIABLES)

    return {
        "chl":     chl_norm,
        "physics": physics_norm,
        "wind":    wind_norm,
        "stats":   all_stats,
    }


# ---------------------------------------------------------------------------
# Step 5: Build static context arrays
# ---------------------------------------------------------------------------

def step_build_static(
    chl_ds: xr.Dataset,
    bathymetry_path: Optional[str] = None,
) -> np.ndarray:
    """
    Assemble static context channels: bathymetry and distance-to-coast.
    If files are not provided, zeros are used as placeholders.

    Returns
    -------
    static_arr : (2, H, W) float32 array (normalized)
    """
    logger.info("STEP 5: Building static context")

    H = chl_ds.sizes["lat"]
    W = chl_ds.sizes["lon"]

    # Bathymetry
    if bathymetry_path and Path(bathymetry_path).exists():
        bathy_ds = load(bathymetry_path)
        from aligner import standardize_coords, clip_to_domain, regrid_to_target
        bathy_ds = standardize_coords(bathy_ds)
        bathy_ds = regrid_to_target(bathy_ds, chl_ds, method="bilinear")
        bathy_arr = bathy_ds.isel(time=0).values if "time" in bathy_ds.dims \
                    else list(bathy_ds.data_vars.values())[0].values
    else:
        logger.warning("No bathymetry file provided; using zeros as placeholder.")
        bathy_arr = np.zeros((H, W), dtype=np.float32)

    # Distance-to-coast: approximate from land mask (Euclidean distance transform)
    from masker import build_land_mask
    land_mask = build_land_mask(chl_ds["chl"])
    ocean_bool = land_mask.values.astype(bool)
    from scipy import ndimage
    dist_arr = ndimage.distance_transform_edt(ocean_bool).astype(np.float32)

    # Stack → (2, H, W)
    static_arr = np.stack([bathy_arr, dist_arr], axis=0)

    # Min-max normalize each channel independently
    for c in range(static_arr.shape[0]):
        ch = static_arr[c]
        finite = ch[np.isfinite(ch)]
        if len(finite) > 0:
            vmin, vmax = finite.min(), finite.max()
            static_arr[c] = (ch - vmin) / max(vmax - vmin, 1e-8)

    logger.info(f"Static context shape: {static_arr.shape}")
    return static_arr


# ---------------------------------------------------------------------------
# Step 6: Extract and save patches
# ---------------------------------------------------------------------------

def step_extract_patches(
    normalized: dict,
    mask_ds: xr.Dataset,
    static_arr: np.ndarray,
) -> None:
    """
    Stack all modalities into NumPy arrays, split into train/val/test,
    and save patches.
    """
    logger.info("STEP 6: Extracting and saving patches")

    chl_np  = normalized["chl"]["chl"].values             # (T, H, W)
    obs_np  = mask_ds["obs_mask"].values                  # (T, H, W)
    mcar_np = mask_ds["mcar_mask"].values                 # (T, H, W)
    mnar_np = mask_ds["mnar_mask"].values                 # (T, H, W)
    bloom_np= mask_ds["bloom_mask"].values                # (T, H, W)
    land_np = mask_ds["land_mask"].values                 # (H, W)

    # Stack physics: (T, C_phy, H, W)
    phy_arrays = [normalized["physics"][v].values for v in cfg.PHY_VARIABLES]
    physics_np = np.stack(phy_arrays, axis=1).astype(np.float32)

    # Stack wind: (T, 2, H, W)
    wind_arrays = [normalized["wind"][v].values for v in cfg.WIND_VARIABLES]
    wind_np = np.stack(wind_arrays, axis=1).astype(np.float32)

    lats  = normalized["chl"]["chl"].lat.values
    lons  = normalized["chl"]["chl"].lon.values
    times = normalized["chl"]["chl"].time.values

    T_total = len(times)
    train_sl, val_sl, test_sl = temporal_split(T_total)

    extractor = PatchExtractor(
        patch_size       = cfg.PATCH_SIZE,
        stride           = cfg.PATCH_STRIDE,
        time_window      = cfg.TIME_WINDOW,
        forecast_horizon = cfg.FORECAST_HORIZON,
        min_valid_frac   = cfg.MIN_VALID_FRACTION,
        land_mask        = land_np,
    )

    os.makedirs(cfg.PATCHES_DIR, exist_ok=True)

    for split, sl in [("train", train_sl), ("val", val_sl), ("test", test_sl)]:
        logger.info(f"Extracting {split} patches...")
        count = save_patches(
            extractor  = extractor,
            chl_norm   = chl_np[sl],
            obs_mask   = obs_np[sl],
            mcar_mask  = mcar_np[sl],
            mnar_mask  = mnar_np[sl],
            physics    = physics_np[sl],
            wind       = wind_np[sl],
            static     = static_arr,
            bloom_mask = bloom_np[sl],
            lats       = lats,
            lons       = lons,
            times      = times[sl],
            output_dir = cfg.PATCHES_DIR,
            split      = split,
        )
        logger.info(f"{split}: {count} patches saved")


# ---------------------------------------------------------------------------
# Full pipeline entry point
# ---------------------------------------------------------------------------

def run_pipeline(
    domain_name: str = cfg.ACTIVE_DOMAIN,
    bathymetry_path: Optional[str] = None,
    download: bool = True,
    chl_path: Optional[str] = None,
    physics_path: Optional[str] = None,
    wind_path: Optional[str] = None,
    recompute_stats: bool = True,
) -> None:
    """
    Run the full MM-MARAS preprocessing pipeline.

    Parameters
    ----------
    domain_name      : key in cfg.DOMAINS to use as spatial bounding box
    bathymetry_path  : path to bathymetry NetCDF/GeoTIFF (optional)
    download         : if True, download data from Copernicus (requires login)
                       if False, paths must be provided via chl_path etc.
    chl_path         : path to existing Chl-a file (used when download=False)
    physics_path     : path to existing physics file
    wind_path        : path to existing wind file
    recompute_stats  : if True, recompute normalization stats from training data
    """
    domain = cfg.DOMAINS[domain_name]
    logger.info(
        f"Starting MM-MARAS preprocessing pipeline | "
        f"domain: {domain_name} | dates: {cfg.DATE_START} to {cfg.DATE_END}"
    )

    # Step 1
    if download:
        paths = step_download(domain)
    else:
        if not all([chl_path, physics_path, wind_path]):
            raise ValueError(
                "When download=False, provide chl_path, physics_path, and wind_path."
            )
        paths = {
            "chl":     chl_path,
            "physics": physics_path,
            "wind":    wind_path,
        }

    # Step 2
    aligned = step_load_and_align(paths, domain)

    # Step 3
    mask_ds = step_build_masks(aligned)

    # Step 4
    stats_path = os.path.join(cfg.STATS_DIR, f"norm_stats_{domain_name}.json")
    normalized = step_normalize(aligned, mask_ds, stats_path, recompute_stats)

    # Step 5
    static_arr = step_build_static(aligned["chl"], bathymetry_path)

    # Step 6
    step_extract_patches(normalized, mask_ds, static_arr)

    logger.info("Preprocessing pipeline complete.")
    logger.info(f"Patches saved to: {cfg.PATCHES_DIR}")
    logger.info(f"Normalization stats at: {stats_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="MM-MARAS Data Preprocessing Pipeline"
    )
    parser.add_argument(
        "--domain", type=str, default=cfg.ACTIVE_DOMAIN,
        choices=list(cfg.DOMAINS.keys()),
        help="Spatial domain preset from config.py",
    )
    parser.add_argument(
        "--no-download", action="store_true",
        help="Skip download; use local files instead",
    )
    parser.add_argument("--chl",     type=str, default=None, help="Path to Chl-a NetCDF")
    parser.add_argument("--physics", type=str, default=None, help="Path to physics NetCDF")
    parser.add_argument("--wind",    type=str, default=None, help="Path to wind NetCDF")
    parser.add_argument("--bathy",   type=str, default=None, help="Path to bathymetry file")
    parser.add_argument(
        "--load-stats", action="store_true",
        help="Load existing normalization stats (skip recomputation)",
    )
    args = parser.parse_args()

    run_pipeline(
        domain_name      = args.domain,
        bathymetry_path  = args.bathy,
        download         = not args.no_download,
        chl_path         = args.chl,
        physics_path     = args.physics,
        wind_path        = args.wind,
        recompute_stats  = not args.load_stats,
    )