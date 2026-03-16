# MM-MARAS Preprocessing Pipeline

Data preprocessing pipeline for the **MM-MARAS** (Multi-Modal Marine Analysis and Reconstruction System) model. Downloads, aligns, normalizes, and packages multi-source oceanographic data into spatiotemporal training patches for deep learning.

---

## What it does

The pipeline ingests five data modalities from Copernicus Marine Service (CMEMS) and the Copernicus Climate Data Store (CDS), aligns them to a common spatial grid and daily time axis, then slices them into overlapping spatiotemporal patches ready for model training.

| Modality | Source | Variables | Resolution |
|---|---|---|---|
| Chlorophyll-a | CMEMS BGC reanalysis | `chl` | 0.25°, daily |
| Ocean physics | CMEMS physics reanalysis | `thetao`, `uo`, `vo`, `mlotst`, `zos` | 0.083°, daily |
| Wind (10m) | ERA5 via CDS | `u10`, `v10` | 0.25°, hourly → daily |
| Precipitation | ERA5 via CDS | `tp` | 0.25°, hourly → daily mm |
| River discharge | GloFAS v4.0 via CDS | `dis24` | 0.05°, daily |

---

## Project structure

```
.
├── pipeline.py       # Main entry point — orchestrates all steps
├── loader.py         # Data download and format loading (NetCDF, HDF5, GeoTIFF, Zarr, GRIB2)
├── aligner.py        # Coordinate standardization, regridding, temporal resampling
├── masker.py         # Observation mask, land mask, bloom mask, MCAR/MNAR classification
├── normalizer.py     # Per-variable transforms (log1p, z-score, min-max)
├── patcher.py        # Spatiotemporal patch extraction and train/val/test splitting
├── config.py         # All settings — domains, dates, variables, normalization stats
└── requirements.txt  # Python dependencies
```

---

## Pipeline steps

```
1. Download     CMEMS (copernicusmarine) + CDS (cdsapi)
2. Load         Auto-detect format → xarray.Dataset
3. Align        Standardize coords → clip → resample wind to daily → regrid to Chl-a grid
4. Mask         obs_mask, land_mask, bloom_mask, mcar_mask, mnar_mask
5. Normalize    log1p + z-score (Chl-a, discharge, precip) | z-score (physics, wind)
6. Static       Bathymetry + distance-to-coast → (2, H, W)
7. Patch        Sliding (T, H, W) window → .npz files + index JSON
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For conservative regridding of flux variables (recommended):

```bash
conda install -c conda-forge xesmf
```

### 2. Authenticate with data services

**Copernicus Marine Service (CMEMS):**

```bash
copernicusmarine login
```

**Copernicus Climate Data Store (CDS):**

Create `~/.cdsapirc` with your credentials from [cds.climate.copernicus.eu](https://cds.climate.copernicus.eu/profile):

```
url: https://cds.climate.copernicus.eu/api
key: your-api-key-here
```

Then accept the ERA5 data licence at:
[reanalysis-era5-single-levels licences](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download#manage-licences)

---

## Usage

### Run the full pipeline

```bash
python pipeline.py --domain gulf_of_mexico
```

### Available domains

| Key | Region |
|---|---|
| `gulf_of_mexico` | Gulf of Mexico |
| `north_atlantic` | North Atlantic |
| `arabian_sea` | Arabian Sea |
| `bay_of_bengal` | Bay of Bengal |
| `lakshadweep_sea` | Southwest India coast |
| `andaman_sea` | Andaman Sea |
| `somali_basin` | Somali / NW Indian Ocean |
| `baltic_sea` | Baltic Sea |
| `chesapeake_bay` | Chesapeake Bay |

### Skip download (use local files)

```bash
python pipeline.py --no-download \
  --chl data/raw/chl.nc \
  --physics data/raw/physics.nc \
  --era5 data/raw/era5_precip_wind.nc \
  --discharge data/raw/glofas_discharge.grib2
```

### Reuse existing normalization stats

```bash
python pipeline.py --domain bay_of_bengal --load-stats
```

### All CLI options

```
--domain        Spatial domain preset (default: gulf_of_mexico)
--no-download   Skip data download; use --chl, --physics, --era5, --discharge paths
--chl           Path to Chl-a NetCDF (when --no-download)
--physics       Path to physics NetCDF (when --no-download)
--era5          Path to ERA5 NetCDF containing tp, u10, v10 (when --no-download)
--discharge     Path to GloFAS GRIB2 file (when --no-download)
--wind          Deprecated alias for --era5 (backward compatibility)
--bathy         Path to bathymetry file (optional; zeros used if omitted)
--load-stats    Load existing normalization stats instead of recomputing
```

---

## Output

```
data/
├── raw/                          Downloaded source files
├── patches/
│   ├── train/
│   │   ├── train_000000.npz
│   │   └── ...
│   ├── val/
│   └── test/
├── train_index.json              Patch metadata (center lat/lon, time_start)
└── stats/
    └── norm_stats_gulf_of_mexico.json
```

Each `.npz` patch contains:

| Key | Shape | Description |
|---|---|---|
| `chl_obs` | `(T, H, W)` | Normalized log-Chl-a (NaN where missing) |
| `obs_mask` | `(T, H, W)` | Binary validity mask |
| `mcar_mask` | `(T, H, W)` | MCAR missingness classification |
| `mnar_mask` | `(T, H, W)` | MNAR missingness classification |
| `physics` | `(T, 5, H, W)` | SST, currents, MLD, sea surface height |
| `wind` | `(T, 2, H, W)` | u10, v10 wind components |
| `static` | `(2, H, W)` | Bathymetry, distance-to-coast |
| `bloom_mask` | `(T, H, W)` | Bloom event labels |
| `target_chl` | `(H_fcast, H, W)` | Future Chl-a for forecast head |

Default patch size: `T=10`, `H=W=64`, forecast horizon `H_fcast=5`, stride `32`.

---

## Configuration

All settings are in `config.py`. Key ones to check before running:

```python
DATE_START   = "2019-01-01"
DATE_END     = "2023-12-31"
ACTIVE_DOMAIN = "gulf_of_mexico"

PATCH_SIZE       = 64
PATCH_STRIDE     = 32
TIME_WINDOW      = 10
FORECAST_HORIZON = 5
```

Normalization statistics in `NORM_STATS` are global Copernicus climatology defaults. After your first run, replace them with the regional stats saved to `data/stats/norm_stats_{domain}.json`.

---

## Notes on ERA5 download

ERA5 is downloaded one calendar month at a time to stay within CDS API size limits. For a 5-year run this produces 60 sequential requests (~1–3 hours total depending on CDS queue load). Completed months are cached in `data/raw/` so the pipeline can resume safely if interrupted.

---

## Requirements

- Python 3.10+
- See `requirements.txt` for full list
- `xesmf` (optional, conda) for conservative regridding of flux variables
- `cfgrib` + `eccodes` for GloFAS GRIB2 loading

---
