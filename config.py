"""
config.py
---------
Central configuration for the MM-MARAS preprocessing pipeline.
All paths, variable names, normalization stats, and spatial/temporal
settings are defined here so every other module stays clean.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Copernicus Marine Service (CMEMS) dataset identifiers
# API tool: copernicusmarine   |   Auth: copernicusmarine login
#
# IMPORTANT — dataset selection by time range:
#
#   BGC Chl-a:
#     Multiyear reanalysis  cmems_mod_glo_bgc_my_0.25deg_P1D-m      1993 – ~2022
#     Interim               cmems_mod_glo_bgc_myint_0.25deg_P1D-m   ~2021 – present-4mo
#     Analysis/Forecast     cmems_mod_glo_bgc-bio_anfc_0.25deg_P1D-m 2021-11 – present
#     → For training data (2019-2023) use MY + MYINT below.
#
#   Physics:
#     Multiyear reanalysis  cmems_mod_glo_phy_my_0.083deg_P1D-m     1993 – ~2021
#     → Contains thetao, uo, vo, mlotst, zos all in one dataset.
#
#   Wind (10m):
#     ERA5 (via CDS) is used for historical wind instead of CMEMS forecast product.
#     WIND_VARIABLES = ["u10", "v10"] from ERA5 single-levels.
# ---------------------------------------------------------------------------

# BGC — Multiyear reanalysis (1993–~2022) — variable name is "chl"
COPERNICUS_BGC_DATASET        = "cmems_mod_glo_bgc_my_0.25deg_P1D-m"
# BGC — Interim extension (~2021–present-4mo) — same variable name "chl"
COPERNICUS_BGC_INTERIM_DATASET= "cmems_mod_glo_bgc_myint_0.25deg_P1D-m"

# Physics — Multiyear reanalysis (1993–~2021), all variables in one dataset
COPERNICUS_PHY_DATASET        = "cmems_mod_glo_phy_my_0.083deg_P1D-m"

# Wind — pulled from ERA5 alongside precipitation (see CDS section below)
# No separate CMEMS wind dataset needed for historical runs.

# Variables to pull from each CMEMS dataset
BGC_VARIABLES   = ["chl"]                              # Chlorophyll-a (mg/m³)
PHY_VARIABLES   = ["thetao", "uo", "vo", "mlotst", "zos"]  # SST, currents, MLD, SSH
WIND_VARIABLES  = ["u10", "v10"]                       # 10m wind from ERA5 (see CDS block)

# Depth range for subsetting 3D datasets (surface only)
# BGC and physics datasets start at ~0.494 m not 0.0 m — use these exact bounds
DEPTH_MIN = 0.494
DEPTH_MAX = 0.495


# ---------------------------------------------------------------------------
# Copernicus Climate Data Store (CDS) dataset identifiers
# API tool: cdsapi   |   Auth: ~/.cdsapirc  (see https://cds.climate.copernicus.eu/how-to-api)
# ---------------------------------------------------------------------------

# GloFAS v4.0 river discharge reanalysis (ERA5-forced LISFLOOD, 0.05°, daily)
# Source: https://ewds.climate.copernicus.eu/datasets/cems-glofas-historical
GLOFAS_DATASET          = "cems-glofas-historical"
GLOFAS_SYSTEM_VERSION   = "version_4_0"
GLOFAS_MODEL            = "lisflood"
GLOFAS_PRODUCT_TYPE     = "consolidated"       # use 'intermediate' for near-real-time
GLOFAS_VARIABLE         = "river_discharge_in_the_last_24_hours"
GLOFAS_RESOLUTION       = 0.05                 # degrees (~5 km native)
GLOFAS_FORMAT           = "grib2"              # native format; pipeline converts to NetCDF

# ERA5 single-levels — precipitation AND 10m wind (pulled together in one request)
# Source: https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
ERA5_DATASET            = "reanalysis-era5-single-levels"
ERA5_PRODUCT_TYPE       = "reanalysis"
ERA5_VARIABLES          = ["total_precipitation",
                            "10m_u_component_of_wind",
                            "10m_v_component_of_wind"]  # downloaded together
# Variable short names in the downloaded NetCDF:
#   total_precipitation → "tp"   (m/hour accumulated → summed to mm/day)
#   10m_u_component    → "u10"  (m/s)
#   10m_v_component    → "v10"  (m/s)
ERA5_RESOLUTION         = 0.25                  # degrees (native)
ERA5_FORMAT             = "netcdf"

# Combined runoff + wind variable names used internally after loading
RUNOFF_VARIABLES        = ["dis24", "tp"]       # GloFAS discharge + ERA5 precipitation


# ---------------------------------------------------------------------------
# Spatial domain presets
# Extend or add custom bounding boxes as needed.
# ---------------------------------------------------------------------------
DOMAINS: Dict[str, Dict] = {
    "global": {
        "lon_min": -180, "lon_max": 180,
        "lat_min":  -90, "lat_max":  90,
    },
    "north_atlantic": {
        "lon_min": -80,  "lon_max":  20,
        "lat_min":  20,  "lat_max":  70,
    },
    "gulf_of_mexico": {
        "lon_min": -98,  "lon_max": -80,
        "lat_min":  18,  "lat_max":  31,
    },
    "chesapeake_bay": {
        "lon_min": -77.5,"lon_max": -75.5,
        "lat_min":  36.8,"lat_max":  39.6,
    },
    "baltic_sea": {
        "lon_min":   9.0,"lon_max":  30.0,
        "lat_min":  53.0,"lat_max":  66.0,
    },
}

# Default domain used throughout the pipeline
ACTIVE_DOMAIN = "bay_of_bengal"


# ---------------------------------------------------------------------------
# Temporal settings
# ---------------------------------------------------------------------------
DATE_START   = "2019-01-01"
DATE_END     = "2023-12-31"
TEMPORAL_RES = "1D"          # Daily composites


# ---------------------------------------------------------------------------
# Spatial resolution & grid
# ---------------------------------------------------------------------------
TARGET_RESOLUTION = 0.25     # degrees (matches BGC product native resolution)
REGRID_METHOD     = "bilinear"   # Options: bilinear | nearest | conservative


# ---------------------------------------------------------------------------
# Patch extraction (for model training)
# ---------------------------------------------------------------------------
PATCH_SIZE    = 64           # spatial pixels (H x W)
PATCH_STRIDE  = 32           # overlap stride
TIME_WINDOW   = 10           # number of time steps per sample (T)
FORECAST_HORIZON = 5         # number of steps to forecast (H)


# ---------------------------------------------------------------------------
# Chlorophyll-a processing
# ---------------------------------------------------------------------------
CHL_LOG_OFFSET   = 1e-4      # added before log to avoid log(0)
CHL_VALID_MIN    = 0.001     # mg/m³ — values below this are flagged
CHL_VALID_MAX    = 100.0     # mg/m³ — values above this are flagged as outliers
CHL_BLOOM_THRESH = 10.0      # mg/m³ — threshold for bloom classification in ERI head


# ---------------------------------------------------------------------------
# Per-variable normalization statistics
# Computed from Copernicus global climatology; update with your regional stats.
# ---------------------------------------------------------------------------
NORM_STATS: Dict[str, Dict[str, float]] = {
    # Regional stats computed from Bay of Bengal training data (2019-01-01 to 2023-09-04).
    # Update these if you switch domains — run the pipeline once and copy from
    # data/stats/norm_stats_{domain}.json.
    "chl_log": {        # log1p(chl), then z-scored
        "mean":  0.146,
        "std":   0.199,
    },
    "thetao": {         # SST (°C) — BoB is consistently warm
        "mean": 29.10,
        "std":   1.24,
    },
    "uo": {             # zonal current (m/s)
        "mean": -0.005,
        "std":   0.241,
    },
    "vo": {             # meridional current (m/s)
        "mean":  0.007,
        "std":   0.216,
    },
    "mlotst": {         # MLD (m), log1p-transformed — shallow in BoB
        "mean":  2.852,
        "std":   0.439,
    },
    "zos": {            # sea surface height (m) — elevated in BoB
        "mean":  0.596,
        "std":   0.108,
    },
    "u10": {            # zonal 10m wind (m/s) — strong SW monsoon signal
        "mean":  0.503,
        "std":   3.628,
    },
    "v10": {            # meridional 10m wind (m/s)
        "mean":  0.821,
        "std":   3.306,
    },
    # Aliases used in some pipeline paths
    "uas": {"mean":  0.503, "std":  3.628},
    "vas": {"mean":  0.821, "std":  3.306},
    # Runoff forcing (both log1p-transformed before z-scoring)
    # These are estimated — recompute from data/stats/ after first pipeline run
    "dis24": {          # GloFAS river discharge (m³/s), log1p-transformed
        "mean":  3.5,
        "std":   2.5,
    },
    "tp": {             # ERA5 total precipitation (mm/day), log1p-transformed
        "mean": -4.5,
        "std":   2.0,
    },
}

# Static variables (not normalized the same way — min-max scaled)
STATIC_VARIABLES = ["bathymetry", "distance_to_coast"]


# ---------------------------------------------------------------------------
# Mask settings
# ---------------------------------------------------------------------------
# Missingness type thresholds
MCAR_SPATIAL_CORR_THRESHOLD = 0.1   # below this → MCAR (random, uncorrelated)
MNAR_CHL_CORR_THRESHOLD     = 0.3   # above this → MNAR (missingness correlated with chl value)

# Minimum valid pixels per time step to keep the scene (else skip)
MIN_VALID_FRACTION = 0.10            # at least 10 % valid pixels required


# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
RAW_DATA_DIR        = "data/raw"
INTERIM_DATA_DIR    = "data/interim"
PROCESSED_DATA_DIR  = "data/processed"
PATCHES_DIR         = "data/patches"
STATS_DIR           = "data/stats"

# Final output format
OUTPUT_FORMAT = "zarr"       # Options: zarr | netcdf


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42