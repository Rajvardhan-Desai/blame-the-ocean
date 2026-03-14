"""
masker.py
---------
Generates binary validity masks and classifies missingness type
(MCAR vs MNAR) for the MaskNet branch of MM-MARAS.

Mask convention (consistent throughout the pipeline):
    1 = valid pixel
    0 = missing / masked pixel

Outputs per time step:
    - obs_mask     : 1 where Chl-a is observed and valid
    - mcar_mask    : 1 where missingness appears random (clouds, scan gaps)
    - mnar_mask    : 1 where missingness is correlated with Chl-a value
                     (sensor saturation during blooms, sun glint)
    - land_mask    : 1 over ocean pixels (0 = land), static
    - bloom_mask   : 1 where Chl-a exceeds bloom threshold (used for ERI)
"""

import logging
from typing import Optional, Tuple

import numpy as np
import xarray as xr
from scipy import ndimage

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Observation mask (valid data detection)
# ---------------------------------------------------------------------------

def build_obs_mask(
    chl: xr.DataArray,
    valid_min: float = 0.001,
    valid_max: float = 100.0,
) -> xr.DataArray:
    """
    Build a binary observation mask: 1 = valid, 0 = missing/invalid.

    A pixel is marked invalid if:
        - it is NaN
        - it is below valid_min (sensor noise floor)
        - it is above valid_max (implausible value or fill flag)
        - it is non-positive (can't take log)

    Parameters
    ----------
    chl       : raw Chl-a DataArray (mg/m³)
    valid_min : minimum physically plausible value
    valid_max : maximum physically plausible value
    """
    mask = (
        np.isfinite(chl.values) &
        (chl.values >= valid_min) &
        (chl.values <= valid_max)
    ).astype(np.float32)

    return xr.DataArray(
        mask,
        coords=chl.coords,
        dims=chl.dims,
        name="obs_mask",
        attrs={"description": "1=valid, 0=missing/invalid", "convention": "1=valid"},
    )


# ---------------------------------------------------------------------------
# Land mask (static — built once from bathymetry or Chl-a temporal coverage)
# ---------------------------------------------------------------------------

def build_land_mask(
    chl: xr.DataArray,
    min_valid_fraction: float = 0.05,
    bathymetry: Optional[xr.DataArray] = None,
) -> xr.DataArray:
    """
    Build a static land mask: 1 = ocean pixel, 0 = land.

    Strategy
    --------
    If bathymetry is provided: ocean = depth < 0 (bathymetry < 0 by convention).
    Otherwise: a pixel is considered ocean if it has at least min_valid_fraction
    of valid Chl-a observations across the full time series.

    Parameters
    ----------
    chl                 : Chl-a DataArray with (time, lat, lon) dims
    min_valid_fraction  : fraction of timesteps that must be valid
    bathymetry          : optional 2D DataArray of seafloor depth (negative = ocean)
    """
    if bathymetry is not None:
        land_mask = (bathymetry.values < 0).astype(np.float32)
        logger.info("Built land mask from bathymetry (depth < 0 = ocean)")
    else:
        obs = build_obs_mask(chl)
        valid_fraction = obs.mean(dim="time").values
        land_mask = (valid_fraction >= min_valid_fraction).astype(np.float32)
        logger.info(
            f"Built land mask from temporal coverage "
            f"(threshold={min_valid_fraction:.0%})"
        )

    lat = chl.lat if "lat" in chl.coords else chl.coords[chl.dims[-2]]
    lon = chl.lon if "lon" in chl.coords else chl.coords[chl.dims[-1]]

    return xr.DataArray(
        land_mask,
        coords={"lat": lat, "lon": lon},
        dims=["lat", "lon"],
        name="land_mask",
        attrs={"description": "Static ocean mask: 1=ocean, 0=land"},
    )


# ---------------------------------------------------------------------------
# Missingness classification: MCAR vs MNAR
# ---------------------------------------------------------------------------

def classify_missingness(
    obs_mask: xr.DataArray,
    chl: xr.DataArray,
    mcar_spatial_corr_thresh: float = 0.1,
    mnar_chl_corr_thresh: float = 0.3,
    window_days: int = 30,
) -> Tuple[xr.DataArray, xr.DataArray]:
    """
    Classify missing pixels into MCAR (Missing Completely At Random)
    and MNAR (Missing Not At Random).

    MCAR heuristic
    --------------
    A missing region is MCAR if the spatial pattern of missingness shows
    low autocorrelation — consistent with random cloud cover or scan-line
    gaps rather than systematic patterns.
    We measure this with Moran's I approximation: spatially coherent
    missingness (clouds, orbits) is NOT MCAR; isolated random pixels are.

    MNAR heuristic
    --------------
    A missing pixel is MNAR if, in a rolling window of surrounding valid
    observations, high Chl-a values are over-represented near the missing
    region. This captures bloom-induced sensor saturation and sun-glint
    flags that systematically occur when Chl-a is high.

    Parameters
    ----------
    obs_mask              : (time, lat, lon) binary mask, 1=valid
    chl                   : (time, lat, lon) raw Chl-a
    mcar_spatial_corr_thresh : Moran's I threshold below which = MCAR
    mnar_chl_corr_thresh     : correlation threshold above which = MNAR
    window_days           : rolling window length for MNAR estimation

    Returns
    -------
    mcar_mask, mnar_mask  : both (time, lat, lon) binary DataArrays
                            1 = pixel is missing AND classified as that type
    """
    obs_np  = obs_mask.values.astype(np.float32)   # (T, H, W)
    chl_np  = chl.values.astype(np.float32)        # (T, H, W)
    T, H, W = obs_np.shape

    missing_np = 1.0 - obs_np   # 1 where data is missing

    mcar_arr = np.zeros_like(missing_np)
    mnar_arr = np.zeros_like(missing_np)

    for t in range(T):
        miss_t = missing_np[t]   # (H, W)

        # ---- MCAR detection ------------------------------------------------
        # Spatial autocorrelation proxy: compare actual missingness with
        # a spatially smoothed version. Low difference → high spatial structure
        # → NOT MCAR. High difference (noisy) → MCAR.
        smoothed    = ndimage.uniform_filter(miss_t, size=5)
        noise_ratio = np.nanstd(miss_t - smoothed) / (np.nanstd(miss_t) + 1e-8)

        if noise_ratio > (1.0 - mcar_spatial_corr_thresh):
            # High noise ratio → missingness is spatially unstructured → MCAR
            mcar_arr[t] = miss_t
        # else: structured missingness (orbital gaps, cloud systems) → not MCAR

        # ---- MNAR detection ------------------------------------------------
        # For each missing pixel, check whether the chl values in a local
        # neighborhood (5x5) are elevated, suggesting bloom-related masking.
        if t < window_days:
            window_slice = slice(0, t + 1)
        else:
            window_slice = slice(t - window_days + 1, t + 1)

        chl_window = chl_np[window_slice]           # (W, H, W)
        obs_window = obs_np[window_slice]

        # Mean Chl-a in local 5x5 neighborhood using valid pixels only
        chl_valid = np.where(obs_window == 1, chl_window, np.nan)
        local_mean_chl = np.nanmean(
            ndimage.uniform_filter(
                np.nanmean(chl_valid, axis=0), size=5
            )
        )
        global_mean_chl = np.nanmean(chl_valid)
        global_std_chl  = np.nanstd(chl_valid)

        if global_std_chl > 0:
            z_score = (local_mean_chl - global_mean_chl) / global_std_chl
        else:
            z_score = 0.0

        if z_score > mnar_chl_corr_thresh:
            # Missing pixels in high-Chl-a neighborhood → MNAR
            mnar_arr[t] = miss_t

    mcar_da = xr.DataArray(
        mcar_arr,
        coords=obs_mask.coords,
        dims=obs_mask.dims,
        name="mcar_mask",
        attrs={"description": "1 = missing AND classified as MCAR"},
    )
    mnar_da = xr.DataArray(
        mnar_arr,
        coords=obs_mask.coords,
        dims=obs_mask.dims,
        name="mnar_mask",
        attrs={"description": "1 = missing AND classified as MNAR"},
    )

    _log_missingness_stats(missing_np, mcar_arr, mnar_arr)
    return mcar_da, mnar_da


def _log_missingness_stats(
    missing: np.ndarray,
    mcar: np.ndarray,
    mnar: np.ndarray,
) -> None:
    total_missing = missing.sum()
    pct_mcar = 100 * mcar.sum() / (total_missing + 1e-8)
    pct_mnar = 100 * mnar.sum() / (total_missing + 1e-8)
    pct_other = 100 - pct_mcar - pct_mnar
    overall_miss = 100 * missing.mean()
    logger.info(
        f"Missingness summary | overall: {overall_miss:.1f}% | "
        f"MCAR: {pct_mcar:.1f}% | MNAR: {pct_mnar:.1f}% | "
        f"structured (other): {pct_other:.1f}%"
    )


# ---------------------------------------------------------------------------
# Bloom mask (for ERI head labeling)
# ---------------------------------------------------------------------------

def build_bloom_mask(
    chl: xr.DataArray,
    bloom_threshold: float = 10.0,
    min_bloom_pixels: int = 10,
    spatial_dilation_px: int = 2,
) -> xr.DataArray:
    """
    Flag bloom events: contiguous patches above bloom_threshold.

    Parameters
    ----------
    chl                : (time, lat, lon) Chl-a DataArray (mg/m³)
    bloom_threshold    : mg/m³ above which a pixel is "bloom-level"
    min_bloom_pixels   : minimum connected region size to count as a bloom
                         (filters single-pixel noise)
    spatial_dilation_px: dilate bloom regions by this many pixels to
                         capture bloom edges

    Returns
    -------
    bloom_mask: (time, lat, lon) binary DataArray, 1 = bloom pixel
    """
    chl_np = chl.values.astype(np.float32)
    T, H, W = chl_np.shape
    bloom_arr = np.zeros((T, H, W), dtype=np.float32)

    struct = ndimage.generate_binary_structure(2, 2)  # 8-connectivity

    for t in range(T):
        frame = chl_np[t]
        raw_bloom = np.isfinite(frame) & (frame >= bloom_threshold)

        # Label connected components
        labeled, n_features = ndimage.label(raw_bloom, structure=struct)

        # Remove small patches (likely noise)
        for comp_id in range(1, n_features + 1):
            if (labeled == comp_id).sum() >= min_bloom_pixels:
                bloom_arr[t][labeled == comp_id] = 1.0

        # Dilate to include bloom edges
        if spatial_dilation_px > 0 and bloom_arr[t].sum() > 0:
            bloom_arr[t] = ndimage.binary_dilation(
                bloom_arr[t],
                iterations=spatial_dilation_px,
            ).astype(np.float32)

    bloom_da = xr.DataArray(
        bloom_arr,
        coords=chl.coords,
        dims=chl.dims,
        name="bloom_mask",
        attrs={
            "description": f"1=bloom pixel (Chl > {bloom_threshold} mg/m³)",
            "bloom_threshold_mg_m3": bloom_threshold,
        },
    )

    total_bloom_pct = 100 * bloom_arr.mean()
    logger.info(
        f"Bloom mask built | threshold={bloom_threshold} mg/m³ | "
        f"bloom coverage: {total_bloom_pct:.2f}%"
    )
    return bloom_da


# ---------------------------------------------------------------------------
# Composite mask dataset builder
# ---------------------------------------------------------------------------

def build_all_masks(
    chl: xr.DataArray,
    valid_min: float = 0.001,
    valid_max: float = 100.0,
    bloom_threshold: float = 10.0,
    bathymetry: Optional[xr.DataArray] = None,
    mcar_threshold: float = 0.1,
    mnar_threshold: float = 0.3,
) -> xr.Dataset:
    """
    Convenience function: build all masks and return as a single Dataset.

    Returns
    -------
    xr.Dataset with variables:
        obs_mask, land_mask, bloom_mask, mcar_mask, mnar_mask
    """
    logger.info("Building all masks")

    obs_mask   = build_obs_mask(chl, valid_min, valid_max)
    land_mask  = build_land_mask(chl, bathymetry=bathymetry)
    bloom_mask = build_bloom_mask(chl, bloom_threshold)
    mcar_mask, mnar_mask = classify_missingness(
        obs_mask, chl, mcar_threshold, mnar_threshold
    )

    ds_masks = xr.Dataset({
        "obs_mask":   obs_mask,
        "land_mask":  land_mask,
        "bloom_mask": bloom_mask,
        "mcar_mask":  mcar_mask,
        "mnar_mask":  mnar_mask,
    })

    logger.info("All masks built successfully")
    return ds_masks
