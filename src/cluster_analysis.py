"""Why is predictive skill low? Positive concentration by cluster, province and age, plus a cluster level model."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src import data as D


def run_cluster_analysis(out="reports/cluster_analysis.json"):
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    clim = D.climate_feature_columns()
    num = [c for c in F if c in D.CONTINUOUS_PERSON + D.DISCRETE_PERSON + D.ORDINAL_PERSON and c not in clim]
    cl = d.groupby("cluster").agg(n=("y", "size"), pos=("y", "sum"), **{c: (c, "first") for c in clim}, **{c: (c, "mean") for c in num})
    top = cl.pos.sort_values(ascending=False)
    res = {"clusters": int(len(cl)), "clusters_with_positive": int((cl.pos > 0).sum()), "positives": int(cl.pos.sum()),
           "top1_cluster_positives": int(top.iloc[0]), "top3_share": float(top.iloc[:3].sum() / top.sum()),
           "top1_share": float(top.iloc[0] / top.sum()),
           "by_province": d.groupby("hv024").y.agg(["size", "sum", "mean"]).round(4).reset_index().to_dict("records"),
           "by_age_group": d.assign(g=pd.cut(d.hv105, [-1, 4, 14, 29, 49, 100])).groupby("g", observed=True).y.agg(["size", "sum", "mean"]).round(4).reset_index().astype({"g": str}).to_dict("records")}
    X = cl[clim + num].fillna(cl[clim + num].median()); y = (cl.pos > 0).astype(int).values
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=0)
    mk = {"logistic_l2": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000)),
          "random_forest": lambda: RandomForestClassifier(500, min_samples_leaf=5, class_weight="balanced_subsample", random_state=0, n_jobs=-1)}
    res["cluster_level_model"] = {}
    for name, f in mk.items():
        oof = np.zeros((10, len(y)))
        for i, (a, b) in enumerate(cv.split(X, y)):
            oof[i // 5, b] = f().fit(X.iloc[a], y[a]).predict_proba(X.iloc[b])[:, 1]
        auc = [roc_auc_score(y, o) for o in oof]; ap = [average_precision_score(y, o) for o in oof]
        res["cluster_level_model"][name] = {"auc_mean": float(np.mean(auc)), "auc_sd": float(np.std(auc)), "ap_mean": float(np.mean(ap)),
                                            "baseline_ap": float(y.mean()), "spearman_with_prevalence": float(spearmanr(oof.mean(0), cl.pos / cl.n)[0])}
    Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=2)
    return res
