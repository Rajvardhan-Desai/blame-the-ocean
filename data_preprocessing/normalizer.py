"""
normalizer.py
-------------
Variable-specific normalization for all MM-MARAS input channels.

Chl-a:      log1p transform → z-score standardization
Physics:    z-score standardization (per-variable)
Wind:       convert to speed + direction OR keep u/v components and z-score
Static:     min-max scaling (bathymetry, distance-to-coast)
Discharge:  log1p → z-score (heavy-tailed distribution)

All statistics are computed from training data only and saved to disk
so the same stats can be applied at inference without data leakage.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import xarray as xr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-variable transform definitions
# ---------------------------------------------------------------------------

# Tells the normalizer which pre-transform to apply before z-scoring
PRETRANSFORMS = {
    "chl":       "log1p",    # log(chl + 1), handles zeros and heavy tail
    "thetao":    "none",
    "uo":        "none",
    "vo":        "none",
    "mlotst":    "log1p",    # MLD is right-skewed
    "zos":       "none",
    "uas":       "none",
    "vas":       "none",
    "dis24":     "log1p",    # GloFAS river discharge (m³/s) — heavy-tailed
    "tp":        "log1p",    # ERA5 precipitation (mm/day) — zero-inflated, heavy tail
    "discharge": "log1p",
    "precip":    "log1p",
    "bathymetry": "minmax",  # static — treated separately
    "dist_coast": "minmax",  # static — treated separately
}


# ---------------------------------------------------------------------------
# Statistics computation (training set only)
# ---------------------------------------------------------------------------

def compute_stats(
    ds: xr.Dataset,
    variables: List[str],
    obs_mask: Optional[xr.DataArray] = None,
    land_mask: Optional[xr.DataArray] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-variable normalization statistics from a dataset.
    Only valid, ocean pixels are included in statistics.

    Parameters
    ----------
    ds        : dataset containing variables to normalize
    variables : list of variable names
    obs_mask  : (time, lat, lon) binary mask; only 1-pixels contribute to stats
    land_mask : (lat, lon) binary mask; only ocean pixels contribute

    Returns
    -------
    dict of {variable: {mean, std, min, max, p01, p99}}
    """
    stats = {}

    for var in variables:
        if var not in ds:
            logger.warning(f"Variable '{var}' not in dataset; skipping stats.")
            continue

        values = ds[var].values.astype(np.float64)

        # Apply pre-transform before computing stats
        pretransform = PRETRANSFORMS.get(var, "none")
        if pretransform == "log1p":
            values = np.log1p(np.clip(values, 0, None))
        elif pretransform == "minmax":
            # For min-max variables, just record raw min/max
            finite_vals = values[np.isfinite(values)]
            stats[var] = {
                "min":   float(np.nanmin(finite_vals)),
                "max":   float(np.nanmax(finite_vals)),
                "mean":  float(np.nanmean(finite_vals)),
                "std":   float(np.nanstd(finite_vals)),
                "p01":   float(np.nanpercentile(finite_vals, 1)),
                "p99":   float(np.nanpercentile(finite_vals, 99)),
            }
            continue

        # Build a combined validity mask
        valid = np.isfinite(values)
        if obs_mask is not None and values.ndim == obs_mask.values.ndim:
            valid &= (obs_mask.values == 1)
        if land_mask is not None:
            # Broadcast land mask across time
            lm = land_mask.values
            if values.ndim == 3:
                lm = lm[np.newaxis, :, :]  # (1, H, W)
            valid &= (lm == 1)

        flat = values[valid]
        flat = flat[np.isfinite(flat)]

        if len(flat) == 0:
            logger.warning(f"No valid values found for '{var}'; stats set to 0/1.")
            stats[var] = {"mean": 0.0, "std": 1.0, "min": 0.0,
                          "max": 1.0, "p01": 0.0, "p99": 1.0}
            continue

        stats[var] = {
            "mean": float(np.mean(flat)),
            "std":  max(float(np.std(flat)), 1e-8),  # avoid division by zero
            "min":  float(np.min(flat)),
            "max":  float(np.max(flat)),
            "p01":  float(np.percentile(flat, 1)),
            "p99":  float(np.percentile(flat, 99)),
        }
        logger.debug(
            f"{var:>12s} | mean={stats[var]['mean']:+.3f} "
            f"std={stats[var]['std']:.3f} "
            f"[{stats[var]['p01']:.3f}, {stats[var]['p99']:.3f}]"
        )

    return stats


def save_stats(stats: Dict, path: Union[str, Path]) -> None:
    """Save normalization statistics to a JSON file."""
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Normalization stats saved to {path}")


def load_stats(path: Union[str, Path]) -> Dict:
    """Load normalization statistics from a JSON file."""
    with open(path) as f:
        stats = json.load(f)
    logger.info(f"Normalization stats loaded from {path}")
    return stats


# ---------------------------------------------------------------------------
# Normalization transforms
# ---------------------------------------------------------------------------

def normalize_variable(
    da: xr.DataArray,
    var_name: str,
    stats: Dict[str, Dict[str, float]],
    clip_outliers: bool = True,
) -> xr.DataArray:
    """
    Apply pre-transform + normalization to a single DataArray.

    Pre-transforms applied before normalization:
        log1p  : log(x + 1) for skewed positive variables
        minmax : (x - min) / (max - min) for static context variables
        none   : raw value passed directly to z-scoring

    Then z-score standardization: (x - mean) / std

    Parameters
    ----------
    da            : input DataArray (may contain NaN for missing pixels)
    var_name      : variable name (used to look up stats and pretransform)
    stats         : dict from compute_stats or load_stats
    clip_outliers : if True, clip at p01/p99 BEFORE normalization to
                    limit influence of extreme values

    Returns
    -------
    Normalized DataArray. NaN pixels remain NaN.
    """
    if var_name not in stats:
        logger.warning(
            f"No stats found for '{var_name}'; returning unnormalized."
        )
        return da

    s = stats[var_name]
    values = da.values.astype(np.float32).copy()
    finite = np.isfinite(values)

    # Optional outlier clipping (on raw values, before transform)
    if clip_outliers and "p01" in s and "p99" in s:
        values[finite] = np.clip(values[finite], s["p01"], s["p99"])

    # Pre-transform
    pretransform = PRETRANSFORMS.get(var_name, "none")
    if pretransform == "log1p":
        values[finite] = np.log1p(np.clip(values[finite], 0, None))
    elif pretransform == "minmax":
        denom = max(s["max"] - s["min"], 1e-8)
        values[finite] = (values[finite] - s["min"]) / denom
        return da.copy(data=values)  # min-max: no further z-scoring

    # Z-score standardization
    values[finite] = (values[finite] - s["mean"]) / s["std"]

    return da.copy(data=values)


def denormalize_variable(
    da: xr.DataArray,
    var_name: str,
    stats: Dict[str, Dict[str, float]],
) -> xr.DataArray:
    """
    Inverse transform: go from normalized back to physical units.
    Used for model outputs (reconstruction head) to recover Chl-a in mg/m³.

    Steps (reverse order of normalize_variable):
        1. Reverse z-score: x_pre = x_norm * std + mean
        2. Reverse pre-transform:
            log1p → expm1(x_pre)  i.e. exp(x_pre) - 1
            minmax → x_pre * (max - min) + min
    """
    if var_name not in stats:
        logger.warning(f"No stats for '{var_name}'; returning as-is.")
        return da

    s = stats[var_name]
    values = da.values.astype(np.float32).copy()
    finite = np.isfinite(values)

    pretransform = PRETRANSFORMS.get(var_name, "none")

    if pretransform == "minmax":
        denom = max(s["max"] - s["min"], 1e-8)
        values[finite] = values[finite] * denom + s["min"]
        return da.copy(data=values)

    # Reverse z-score
    values[finite] = values[finite] * s["std"] + s["mean"]

    # Reverse pre-transform
    if pretransform == "log1p":
        values[finite] = np.expm1(values[finite])
        values[finite] = np.clip(values[finite], 0, None)  # physical constraint

    return da.copy(data=values)


# ---------------------------------------------------------------------------
# Dataset-level normalization
# ---------------------------------------------------------------------------

def normalize_dataset(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
    variables: Optional[List[str]] = None,
    clip_outliers: bool = True,
) -> xr.Dataset:
    """
    Normalize all (or selected) variables in a dataset.

    Parameters
    ----------
    ds            : input dataset
    stats         : normalization statistics (from compute_stats or load_stats)
    variables     : subset of variables to normalize; all if None
    clip_outliers : clip to [p01, p99] before normalizing

    Returns
    -------
    New dataset with normalized variables. Coordinates and masks are unchanged.
    """
    vars_to_norm = variables or list(ds.data_vars)
    normalized_vars = {}

    for var in vars_to_norm:
        if var not in ds:
            continue
        normalized_vars[var] = normalize_variable(
            ds[var], var, stats, clip_outliers
        )
        logger.debug(f"Normalized: {var}")

    # Keep any variables not normalized (e.g. masks) unchanged
    unchanged = {v: ds[v] for v in ds.data_vars if v not in vars_to_norm}
    normalized_vars.update(unchanged)

    return xr.Dataset(normalized_vars, coords=ds.coords, attrs=ds.attrs)


def denormalize_dataset(
    ds: xr.Dataset,
    stats: Dict[str, Dict[str, float]],
    variables: Optional[List[str]] = None,
) -> xr.Dataset:
    """
    Inverse normalize a dataset (model output → physical units).
    """
    vars_to_denorm = variables or list(ds.data_vars)
    result = {}

    for var in vars_to_denorm:
        if var not in ds:
            continue
        result[var] = denormalize_variable(ds[var], var, stats)

    unchanged = {v: ds[v] for v in ds.data_vars if v not in vars_to_denorm}
    result.update(unchanged)

    return xr.Dataset(result, coords=ds.coords, attrs=ds.attrs)


# ---------------------------------------------------------------------------
# Wind: u/v → speed + direction (optional preprocessing step)
# ---------------------------------------------------------------------------

def compute_wind_speed_direction(
    ds: xr.Dataset,
    u_var: str = "uas",
    v_var: str = "vas",
    drop_components: bool = False,
) -> xr.Dataset:
    """
    Compute wind speed and meteorological direction from u/v components.

    Wind speed = sqrt(u² + v²)
    Wind direction = atan2(-u, -v) in degrees (meteorological convention:
    direction FROM which wind blows, 0°=N, 90°=E)

    Parameters
    ----------
    drop_components : if True, removes u/v after computing speed/direction
    """
    if u_var not in ds or v_var not in ds:
        logger.warning(f"Wind components {u_var}/{v_var} not found; skipping.")
        return ds

    u = ds[u_var].values
    v = ds[v_var].values

    speed = np.sqrt(u**2 + v**2).astype(np.float32)
    direction = (
        np.degrees(np.arctan2(-u, -v)) % 360
    ).astype(np.float32)

    ds = ds.assign({
        "wind_speed": xr.DataArray(
            speed, coords=ds[u_var].coords, dims=ds[u_var].dims,
            attrs={"units": "m/s", "long_name": "10m wind speed"}
        ),
        "wind_dir": xr.DataArray(
            direction, coords=ds[u_var].coords, dims=ds[u_var].dims,
            attrs={"units": "degrees", "long_name": "10m wind direction (met convention)"}
        ),
    })

    if drop_components:
        ds = ds.drop_vars([u_var, v_var])

    return ds
