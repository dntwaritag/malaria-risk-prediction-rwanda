"""Interview-month dynamic climate features per survey cluster, from the downloaded external data.

Sources (all in data/external/, git-ignored):
  * CHIRPS v3 monthly rainfall GeoTIFFs, mean over pixels whose centres lie within 5 km of the displaced cluster point
  * ERA5-Land monthly 2 m dewpoint temperature (0.1 degree grid, nearest cell)
  * MODIS MOD13Q1.061 EVI and NDVI composites extracted by AppEEARS (quality filtered, monthly mean)
Features are the value in the interview month (lag 0) and 1 to 3 month lags where data exist (July to December 2023).
No survey label is used, so building these features cannot leak the target.
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXT = Path("data/external")


def _chirps(points: pd.DataFrame, radius_km=5.0) -> pd.DataFrame:
    import rasterio
    rows = []
    for f in sorted(glob.glob(str(EXT / "chirps" / "chirps-v3.0.*.tif"))):
        y, m = map(int, re.search(r"(\d{4})\.(\d{2})\.tif", f).groups())
        with rasterio.open(f) as ds:
            arr = ds.read(1).astype(float)
            nod = ds.nodata
            if nod is not None:
                arr[arr == nod] = np.nan
            arr[arr < 0] = np.nan
            for _, p in points.iterrows():
                r, c = ds.index(p.long, p.lat)
                vals = []
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        x, yy = ds.xy(r + dr, c + dc)
                        dkm = np.hypot((x - p.long) * 111.32 * np.cos(np.radians(p.lat)), (yy - p.lat) * 110.57)
                        if dkm <= radius_km:
                            vals.append(arr[r + dr, c + dc])
                v = np.nanmean(vals) if len(vals) and not np.all(np.isnan(vals)) else arr[r, c]
                rows.append((int(p.cluster), y, m, float(v)))
    return pd.DataFrame(rows, columns=["cluster", "year", "month", "rain_mm"])


def _era5(points: pd.DataFrame) -> pd.DataFrame:
    import netCDF4
    rows = []
    for f in sorted(glob.glob(str(EXT / "era5land" / "*.nc"))):
        d = netCDF4.Dataset(f)
        t = pd.to_datetime(d["valid_time"][:], unit="s")
        lat, lon = d["latitude"][:], d["longitude"][:]
        if "d2m" not in d.variables:
            continue
        v = d["d2m"][:] - 273.15
        for _, p in points.iterrows():
            i, j = int(np.argmin(abs(lat - p.lat))), int(np.argmin(abs(lon - p.long)))
            for k in range(len(t)):
                rows.append((int(p.cluster), t[k].year, t[k].month, float(v[k, i, j])))
    return pd.DataFrame(rows, columns=["cluster", "year", "month", "dewpoint_c"])


def _modis() -> pd.DataFrame:
    m = pd.read_csv(EXT / "modis_appeears" / "rwanda-mis2023-vi-MOD13Q1-061-results.csv")
    m = m[(m["MOD13Q1_061__250m_16_days_VI_Quality_Mixed_Clouds"] == "0b0") &
          (m["MOD13Q1_061__250m_16_days_VI_Quality_MODLAND"].isin(["0b00", "0b01"]))]
    m["mid"] = pd.to_datetime(m.Date) + pd.Timedelta(days=8)
    out = m.assign(year=m.mid.dt.year, month=m.mid.dt.month).groupby(["ID", "year", "month"]).agg(
        evi=("MOD13Q1_061__250m_16_days_EVI", "mean"), ndvi=("MOD13Q1_061__250m_16_days_NDVI", "mean")).reset_index()
    return out.rename(columns={"ID": "cluster"})


def build_dynamic_features(interview: pd.DataFrame, points_csv=EXT / "appeears_points_mis2023.csv") -> pd.DataFrame:
    """interview: DataFrame with columns cluster, month (hv006), year (hv007). Returns dyn_* columns, one row per interview row."""
    pts = pd.read_csv(points_csv).rename(columns={"ID": "cluster", "Latitude": "lat", "Longitude": "long"})
    tbl = _chirps(pts).merge(_era5(pts), on=["cluster", "year", "month"], how="outer").merge(
        _modis(), on=["cluster", "year", "month"], how="outer")
    tbl["t"] = tbl.year * 12 + tbl.month
    vars_ = {"rain_mm": "dyn_rain", "dewpoint_c": "dyn_dewpoint", "evi": "dyn_evi", "ndvi": "dyn_ndvi"}
    key = interview[["cluster", "year", "month"]].drop_duplicates().copy()
    key["t"] = key.year * 12 + key.month
    out = key.copy()
    for lag in (0, 1, 2, 3):
        sh = tbl[["cluster", "t"] + list(vars_)].copy(); sh["t"] = sh.t + lag
        sh = sh.rename(columns={k: f"{v}_lag{lag}" for k, v in vars_.items()})
        out = out.merge(sh, on=["cluster", "t"], how="left")
    out["dyn_rain_sum3"] = out[["dyn_rain_lag1", "dyn_rain_lag2", "dyn_rain_lag3"]].sum(axis=1, min_count=3)
    return interview[["cluster", "year", "month"]].merge(out.drop(columns="t"), on=["cluster", "year", "month"], how="left")
