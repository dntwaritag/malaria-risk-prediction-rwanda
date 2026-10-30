"""Temporal transfer experiment: train on MIS 2017 (1,765 positives), evaluate on all of MIS 2023 (179 positives).

Why: the 2023 grouped hold-out has only 36 positives, so its metrics are dominated by a few clusters.
2017 and 2023 are different survey rounds with disjoint clusters, so evaluating 2017 -> 2023 is leakage
free and uses all 2023 positives. Only household features common to both surveys are used (no GPS or
climate covariates exist for 2017).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src import data as D
from src.evaluation import cluster_bootstrap, metrics
from src.pipeline_builder import build_model_pipeline

HOUSEHOLD = D.CONTINUOUS_PERSON + D.DISCRETE_PERSON + D.ORDINAL_PERSON + D.NOMINAL_PERSON


def run_transfer(raw2017="data/raw/mis2017/RWPR7AFL.DTA", out="reports/transfer_2017_to_2023.json", n_boot=500,
                 models=("logistic", "random_forest", "lightgbm", "xgboost")):
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    cols = [c for c in F if c in HOUSEHOLD]
    p7 = pd.read_stata(raw2017, convert_categoricals=False)
    t = p7[p7.hml35.isin([0, 1])].copy(); t["y"] = t.hml35.astype(int)
    cols = [c for c in cols if c in t.columns]
    res = {"n_features": len(cols), "train_2017": {"n": int(len(t)), "positives": int(t.y.sum())},
           "test_2023": {"n": int(len(d)), "positives": int(d.y.sum()), "baseline_pr_auc": float(d.y.mean())}, "models": {}}
    for m in models:
        pipe = build_model_pipeline(m, t[cols], cols).fit(t[cols], t.y.values)
        p = pipe.predict_proba(d[cols])[:, 1]
        thr = float(np.quantile(p, 0.80))  # flag the top 20% highest-risk people (operational budget rule)
        res["models"][m] = {"unweighted": metrics(d.y.values, p, None, thr), "weighted": metrics(d.y.values, p, d.weight.values, thr),
                            "ci": cluster_bootstrap(d.y.values, p, d.weight.values, d.cluster.values, thr, n_boot)}
        print(m, round(res["models"][m]["unweighted"]["pr_auc"], 4), round(res["models"][m]["unweighted"]["roc_auc"], 3))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(out, "w"), indent=2)
    return res
