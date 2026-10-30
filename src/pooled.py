"""Improved evaluation: pooled out-of-fold results over ALL 179 MIS 2023 positives, with optional MIS 2017 augmentation.

For each cluster-grouped fold of MIS 2023, the model trains on the other 2023 folds (optionally plus all of MIS 2017)
and predicts the held-out fold, so every 2023 person is predicted exactly once by a model that never saw their cluster.
MIS 2017 has no GPS or climate covariates, so augmented models use household features only.
Cluster mean features (age, wealth score, household size, rooms, nets) summarize local context without using any label.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import data as D
from src.evaluation import Calibrator, cluster_bootstrap
from src.pipeline_builder import build_model_pipeline

HOUSEHOLD = D.CONTINUOUS_PERSON + D.DISCRETE_PERSON + D.ORDINAL_PERSON + D.NOMINAL_PERSON
MEAN_COLS = ["hv105", "hv271", "hv009", "hv216", "hml1"]


def add_cluster_means(df: pd.DataFrame, cluster_col: str) -> pd.DataFrame:
    out = df.copy()
    for c in MEAN_COLS:
        if c in out:
            out[f"cm_{c}"] = out.groupby(cluster_col)[c].transform("mean")
    return out


def load_2017(raw="data/raw/mis2017/RWPR7AFL.DTA", cols=None):
    p7 = pd.read_stata(raw, convert_categoricals=False)
    t = p7[p7.hml35.isin([0, 1])].copy()
    t["y"] = t.hml35.astype(int); t["cluster"] = 100000 + t.hv001.astype(int)
    return t[["y", "cluster"] + [c for c in cols if c in t.columns]]


def pooled_cv(model, d23, feats, d17=None, cluster_means=False, n_splits=5, seed=42):
    warnings.filterwarnings("ignore")
    X23 = d23[["cluster", "y", "weight"] + feats].copy()
    X17 = d17[["cluster", "y"] + feats].copy() if d17 is not None else None
    use = list(feats)
    if cluster_means:
        X23 = add_cluster_means(X23, "cluster"); use += [f"cm_{c}" for c in MEAN_COLS if c in X23]
        if X17 is not None:
            X17 = add_cluster_means(X17, "cluster")
    y, g = X23.y.values, X23.cluster.values
    oof = np.zeros(len(X23))
    for tr, va in StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed).split(X23, y, g):
        tr_df = X23.iloc[tr]
        if X17 is not None:
            tr_df = pd.concat([tr_df[["y"] + use], X17[["y"] + use]], ignore_index=True)
        pipe = build_model_pipeline(model, tr_df[use], use).fit(tr_df[use], tr_df.y.values)
        oof[va] = pipe.predict_proba(X23.iloc[va][use])[:, 1]
    # Platt recalibration on the pooled OOF scores (rank metrics are unchanged; Brier improves)
    cal = Calibrator("platt").fit(oof, y)
    pc = cal.predict(oof)
    res = {"pr_auc": float(average_precision_score(y, oof)), "roc_auc": float(roc_auc_score(y, oof)),
           "pr_auc_weighted": float(average_precision_score(y, oof, sample_weight=X23.weight)),
           "brier_raw": float(brier_score_loss(y, oof)), "brier_platt": float(brier_score_loss(y, pc)),
           "top20pct_recall": float(y[oof >= np.quantile(oof, 0.8)].sum() / y.sum()),
           "lift_top20pct": float(y[oof >= np.quantile(oof, 0.8)].mean() / y.mean())}
    thr = float(np.quantile(pc, 0.8))
    res["ci"] = cluster_bootstrap(y, pc, X23.weight.values, g, thr, n_boot=300)["unweighted"]
    return res, oof


def run_pooled(out="reports/pooled_evaluation.json", models=("logistic", "lightgbm", "xgboost")):
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    hh = [c for c in F if c in HOUSEHOLD]
    d17 = load_2017(cols=hh)
    configs = {"A_2023_all_features": dict(feats=F, d17=None, cluster_means=False),
               "B_2023_household": dict(feats=hh, d17=None, cluster_means=False),
               "C_2023_household_cluster_means": dict(feats=hh, d17=None, cluster_means=True),
               "D_2017_plus_2023_household": dict(feats=hh, d17=d17, cluster_means=False),
               "E_2017_plus_2023_household_cluster_means": dict(feats=hh, d17=d17, cluster_means=True)}
    results = {"n": int(len(d)), "positives": int(d.y.sum()), "baseline_pr_auc": float(d.y.mean()), "configs": {}}
    for name, cfg in configs.items():
        results["configs"][name] = {}
        for m in models:
            res, _ = pooled_cv(m, d, **cfg)
            results["configs"][name][m] = res
            print(name, m, round(res["pr_auc"], 4), round(res["roc_auc"], 3), round(res["lift_top20pct"], 2), flush=True)
            Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(results, open(out, "w"), indent=2)
    return results


if __name__ == "__main__":
    run_pooled()
