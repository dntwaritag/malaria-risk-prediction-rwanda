"""Pooled out-of-fold evaluation with interview-month climate features (CHIRPS, ERA5-Land dewpoint, MODIS EVI/NDVI)."""
import json
import warnings
from pathlib import Path

import pandas as pd

from src import data as D
from src.climate_features import build_dynamic_features
from src.pooled import HOUSEHOLD, pooled_cv


def load_with_dynamic():
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    iv = d[["cluster", "int_month", "int_year"]].rename(columns={"int_month": "month", "int_year": "year"})
    f = build_dynamic_features(iv)
    dyn = [c for c in f.columns if c.startswith("dyn_")]
    d = pd.concat([d.reset_index(drop=True), f[dyn].reset_index(drop=True)], axis=1)
    return d, F, dyn


def run(out="reports/pooled_dynamic_evaluation.json", models=("logistic", "lightgbm", "xgboost")):
    warnings.filterwarnings("ignore")
    d, F, dyn = load_with_dynamic(); hh = [c for c in F if c in HOUSEHOLD]
    configs = {"F_household_cluster_means_dynamic": dict(feats=hh + dyn, cluster_means=True),
               "G_household_dynamic": dict(feats=hh + dyn, cluster_means=False),
               "H_all_static_plus_dynamic": dict(feats=F + dyn, cluster_means=False)}
    res = {"n": int(len(d)), "positives": int(d.y.sum()), "baseline_pr_auc": float(d.y.mean()), "dynamic_features": dyn, "configs": {}}
    for name, cfg in configs.items():
        res["configs"][name] = {}
        for m in models:
            r, _ = pooled_cv(m, d, **cfg)
            res["configs"][name][m] = r
            print(name, m, round(r["pr_auc"], 4), round(r["roc_auc"], 3), round(r["lift_top20pct"], 2), flush=True)
            Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=2)
    return res


if __name__ == "__main__":
    run()
