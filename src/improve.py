"""Improvement experiments: combine every downloaded source and test it with the SAME repeated cluster-grouped CV.

Candidate additions to the current best set (household features + cluster means):
  dyn_*   interview-month CHIRPS rainfall, ERA5-Land dewpoint, MODIS EVI/NDVI (lags 0-3)       [src/climate_features.py]
  x_temp  DHS monthly temperature climatology for the interview month (cluster level, no label)
  gc      the DHS geospatial covariates (static / 2020 rasters)
  sp_*    spatial-neighbour burden: distance-weighted case rate of the k nearest TRAINING clusters, computed inside
          each fold (never uses validation labels; a training cluster never sees its own label)
All folds and seeds match src/repeated.py (seeds 100-104), so differences are paired.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import data as D
from src.climate_features import build_dynamic_features
from src.pipeline_builder import build_model_pipeline
from src.pooled import HOUSEHOLD, MEAN_COLS, add_cluster_means

MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
DYN_COMPACT = ["dyn_rain_lag0", "dyn_rain_sum3", "dyn_evi_lag0", "dyn_ndvi_lag0", "dyn_dewpoint_lag0"]


def load_all():
    r = D.load_mis2023(); d = r["data"].reset_index(drop=True); F = r["feature_columns"]
    iv = d[["cluster", "int_month", "int_year"]].rename(columns={"int_month": "month", "int_year": "year"})
    f = build_dynamic_features(iv)
    dyn = [c for c in f.columns if c.startswith("dyn_")]
    d = pd.concat([d, f[dyn].reset_index(drop=True)], axis=1)
    gc = pd.read_csv("data/raw/mis2023/RWGC8AFL.csv")
    tm = gc[["DHSCLUST"] + [f"Temperature_{m}" for m in MONTHS]].rename(columns={"DHSCLUST": "cluster"})
    d = d.merge(tm, on="cluster", how="left")
    d["x_temp_month"] = [row[f"Temperature_{MONTHS[int(m) - 1]}"] for m, (_, row) in zip(d.int_month, d.iterrows())]
    d = d.drop(columns=[f"Temperature_{m}" for m in MONTHS])
    pts = pd.read_csv("data/external/cluster_points_mis2023.csv").set_index("cluster")[["lat", "long"]]
    return d, F, dyn, pts


def spatial_rate(train: pd.DataFrame, targets: np.ndarray, pts: pd.DataFrame, k=5, m=200.0):
    """Distance-weighted, shrunken case rate of the k nearest training clusters (own cluster excluded)."""
    st = train.groupby("cluster").y.agg(n="size", pos="sum")
    st = st[st.index.isin(pts.index)]
    prior = train.y.mean()
    lat0 = np.radians(pts.loc[st.index, "lat"].values); lon0 = np.radians(pts.loc[st.index, "long"].values)
    out = {}
    for c in targets:
        if c not in pts.index:
            out[c] = prior; continue
        la, lo = np.radians(pts.loc[c, "lat"]), np.radians(pts.loc[c, "long"])
        dist = 6371 * np.hypot((lon0 - lo) * np.cos(la), lat0 - la)
        mask = st.index.values != c
        idx = np.argsort(np.where(mask, dist, np.inf))[:k]
        w = 1.0 / np.maximum(dist[idx], 1.0)
        out[c] = (np.sum(w * st.pos.values[idx]) + m * prior * w.mean()) / (np.sum(w * st.n.values[idx]) + m * w.mean())
    return out


def one_repeat(model, d, feats, seed, pts, cluster_means=True, spatial=False, C=None, k=5):
    X = d[["cluster", "y", "weight"] + feats].copy(); use = list(feats)
    if cluster_means:
        X = add_cluster_means(X, "cluster"); use += [f"cm_{c}" for c in MEAN_COLS if c in X]
    if spatial:
        use += ["sp_rate"]
    y, g = X.y.values, X.cluster.values
    oof = np.zeros(len(X)); ncols = []
    for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
        t, v = X.iloc[tr].copy(), X.iloc[va].copy()
        if spatial:
            tc = t.cluster.unique(); sr = spatial_rate(t, np.r_[tc, v.cluster.unique()], pts, k=k)
            t["sp_rate"] = t.cluster.map(sr); v["sp_rate"] = v.cluster.map(sr)
        pipe = build_model_pipeline(model, t[use], use)
        if C is not None and model == "logistic":
            pipe[-1].set_params(C=C)
        pipe.fit(t[use], t.y.values)
        oof[va] = pipe.predict_proba(v[use])[:, 1]
        ncols.append(pipe[:-1].transform(t[use].head(5)).shape[1])
    thr = np.quantile(oof, 0.8)
    return {"pr_auc": float(average_precision_score(y, oof)), "roc_auc": float(roc_auc_score(y, oof)),
            "lift_top20": float(y[oof >= thr].mean() / y.mean()), "n_cols": float(np.mean(ncols)), "oof": oof.tolist()}


def repeated(model, d, pts, repeats=5, seed0=100, **cfg):
    warnings.filterwarnings("ignore")
    runs = [one_repeat(model, d, pts=pts, seed=seed0 + i, **cfg) for i in range(repeats)]
    return {m: {"mean": float(np.mean([r[m] for r in runs])), "sd": float(np.std([r[m] for r in runs], ddof=1)),
                "per_repeat": [r[m] for r in runs]} for m in ("pr_auc", "roc_auc", "lift_top20", "n_cols")}, runs


CONFIGS = lambda hh, F, dyn: {
    "base_household_cm": dict(feats=hh),
    "plus_dynamic_compact": dict(feats=hh + DYN_COMPACT),
    "plus_dynamic_all": dict(feats=hh + dyn),
    "plus_temp_month": dict(feats=hh + ["x_temp_month"]),
    "plus_gc_static": dict(feats=F),
    "plus_spatial_k5": dict(feats=hh, spatial=True),
    "plus_spatial_k10": dict(feats=hh, spatial=True, k=10),
    "plus_spatial_dyn_compact": dict(feats=hh + DYN_COMPACT, spatial=True),
    "plus_spatial_temp": dict(feats=hh + ["x_temp_month"], spatial=True),
    "base_C0.01": dict(feats=hh, C=0.01),
    "base_C0.03": dict(feats=hh, C=0.03),
    "base_C0.3": dict(feats=hh, C=0.3),
    "spatial_k5_C0.03": dict(feats=hh, spatial=True, C=0.03),
    "everything": dict(feats=F + dyn + ["x_temp_month"], spatial=True),
}


def run(out="reports/improvement_experiments.json", models=("logistic",), only=None):
    d, F, dyn, pts = load_all(); hh = [c for c in F if c in HOUSEHOLD]
    res = json.load(open(out)) if Path(out).exists() else {"configs": {}}
    for name, cfg in CONFIGS(hh, F, dyn).items():
        if only and name not in only:
            continue
        for m in models:
            if m in res["configs"].get(name, {}):
                continue
            r, runs = repeated(m, d, pts, **cfg)
            r["oof_first_repeat"] = runs[0]["oof"]
            res["configs"].setdefault(name, {})[m] = r
            print(name, m, round(r["pr_auc"]["mean"], 4), "+-", round(r["pr_auc"]["sd"], 4), "roc", round(r["roc_auc"]["mean"], 3),
                  "lift", round(r["lift_top20"]["mean"], 2), "cols", round(r["n_cols"]["mean"], 1), flush=True)
            Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"))
    return res


if __name__ == "__main__":
    only = sys.argv[1].split(",") if len(sys.argv) > 1 and sys.argv[1] != "all" else None
    models = tuple(sys.argv[2].split(",")) if len(sys.argv) > 2 else ("logistic",)
    import os
    run(out=os.environ.get("OUT", "reports/improvement_experiments.json"), only=only, models=models)
