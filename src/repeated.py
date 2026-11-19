"""Headline evaluation: repeated, stratified, cluster-grouped out-of-fold CV (5 repeats x 5 folds).

Each repeat draws a different random cluster partition (StratifiedGroupKFold), so no single favourable seed
decides the result and the influence of the cluster holding 39 positives is averaged over partitions. Every person
is predicted once per repeat by a model that never saw their cluster; metrics are computed per repeat and then
summarized as mean and standard deviation across repeats.

Survey-round pooling (MIS 2017 + MIS 2023) uses an explicit ``survey_round`` indicator and covariate-shift
importance weights: a domain classifier separates 2017 from 2023 rows on household features (fitted on the training
fold only), and 2017 rows receive weights p(2023|x)/p(2017|x), clipped to [0.1, 10] and normalized to mean 1.
Also reports events per variable (EPV): training positives divided by model columns after preprocessing and screening.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import data as D
from src.pipeline_builder import build_model_pipeline
from src.pooled import HOUSEHOLD, MEAN_COLS, add_cluster_means, load_2017

REPEATS, FOLDS = 5, 5
# Compact set (events-per-variable): drop nominal features with more than 3 levels after preprocessing
# (water source, wall, toilet, province, cooking fuel, floor, net type). Cluster means carry local context instead.
HIGH_CARDINALITY = ["hv201", "hv214", "hv205", "hv024", "hv226", "hv213", "hml12"]


def shift_weights(Z23: pd.DataFrame, Z17: pd.DataFrame, clip=(0.1, 10.0)) -> np.ndarray:
    """Importance weights for 2017 rows toward the 2023 covariate distribution (training data only)."""
    cols = [c for c in Z23.columns if "survey_round" not in c]
    X = pd.concat([Z23[cols], Z17[cols]], ignore_index=True).values
    d = np.r_[np.ones(len(Z23)), np.zeros(len(Z17))]
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(X, d)
    p = np.clip(clf.predict_proba(Z17[cols].values)[:, 1], 1e-3, 1 - 1e-3)
    r = p / (1 - p)
    r = r / r.mean()          # relative importance; the raw odds are tiny because the domains are well separated
    w = np.clip(r, *clip)
    return w / w.mean()


def one_repeat(model, d23, feats, d17, cluster_means, round_indicator, shift_weight, seed):
    X23 = d23[["cluster", "y", "weight"] + feats].copy()
    X17 = d17[["cluster", "y"] + feats].copy() if d17 is not None else None
    use = list(feats)
    if cluster_means:
        X23 = add_cluster_means(X23, "cluster"); use += [f"cm_{c}" for c in MEAN_COLS if c in X23]
        if X17 is not None:
            X17 = add_cluster_means(X17, "cluster")
    if X17 is not None and round_indicator:
        X23["survey_round"] = 0.0; X17["survey_round"] = 1.0; use += ["survey_round"]
    y, g = X23.y.values, X23.cluster.values
    oof = np.zeros(len(X23)); ncols = []; ntrain_pos = []
    for tr, va in StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed).split(X23, y, g):
        t23 = X23.iloc[tr]
        if X17 is not None:
            tr_df = pd.concat([t23[["y"] + use], X17[["y"] + use]], ignore_index=True)
            is17 = np.r_[np.zeros(len(t23), bool), np.ones(len(X17), bool)]
        else:
            tr_df, is17 = t23[["y"] + use], np.zeros(len(t23), bool)
        pipe = build_model_pipeline(model, tr_df[use], use)
        prep = pipe[:-1].fit(tr_df[use], tr_df.y.values)
        Ztr = prep.transform(tr_df[use])
        w = np.ones(len(tr_df))
        if shift_weight and is17.any():
            w[is17] = shift_weights(Ztr[~is17], Ztr[is17])
        pipe[-1].fit(Ztr, tr_df.y.values, sample_weight=w)
        oof[va] = pipe[-1].predict_proba(prep.transform(X23.iloc[va][use]))[:, 1]
        ncols.append(Ztr.shape[1]); ntrain_pos.append(int(t23.y.sum()))
    thr = np.quantile(oof, 0.8)
    return {"pr_auc": float(average_precision_score(y, oof)), "roc_auc": float(roc_auc_score(y, oof)),
            "pr_auc_weighted": float(average_precision_score(y, oof, sample_weight=X23.weight.values)),
            "lift_top20": float(y[oof >= thr].mean() / y.mean()), "top20_recall": float(y[oof >= thr].sum() / y.sum()),
            "n_cols": float(np.mean(ncols)), "train_pos": float(np.mean(ntrain_pos))}


def repeated_cv(model, d23, feats, d17=None, cluster_means=False, round_indicator=False, shift_weight=False,
                repeats=REPEATS, seed0=100):
    warnings.filterwarnings("ignore")
    runs = [one_repeat(model, d23, feats, d17, cluster_means, round_indicator, shift_weight, seed0 + i) for i in range(repeats)]
    out = {m: {"mean": float(np.mean([r[m] for r in runs])), "sd": float(np.std([r[m] for r in runs], ddof=1))} for m in runs[0]}
    out["epv"] = out["train_pos"]["mean"] / out["n_cols"]["mean"]
    out["runs"] = runs
    return out


def run(out="reports/repeated_evaluation.json", models=("logistic", "xgboost", "lightgbm"), only=None):
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    hh = [c for c in F if c in HOUSEHOLD]
    d17 = load_2017(cols=hh)
    configs = {"A_all_features": dict(feats=F),
               "B_household": dict(feats=hh),
               "C_household_cluster_means": dict(feats=hh, cluster_means=True),
               "D_pool2017_unweighted": dict(feats=hh, d17=d17),
               "E_pool2017_round_indicator": dict(feats=hh, d17=d17, round_indicator=True),
               "F_pool2017_round_indicator_shift_weights": dict(feats=hh, d17=d17, round_indicator=True, shift_weight=True),
               "G_compact_cluster_means": dict(feats=[c for c in hh if c not in HIGH_CARDINALITY], cluster_means=True),
               "H_compact_no_context": dict(feats=[c for c in hh if c not in HIGH_CARDINALITY])}
    res = {"n": int(len(d)), "positives": int(d.y.sum()), "baseline_pr_auc": float(d.y.mean()),
           "repeats": REPEATS, "folds": FOLDS, "configs": {}}
    if Path(out).exists():
        res = json.load(open(out))
    for name, cfg in configs.items():
        if only and name not in only:
            continue
        for m in models:
            if m in res["configs"].get(name, {}):
                continue
            if name.startswith(("D", "E", "F")) and m == "lightgbm":
                continue
            v = repeated_cv(m, d, **cfg)
            res["configs"].setdefault(name, {})[m] = v
            print(name, m, round(v["pr_auc"]["mean"], 4), "+-", round(v["pr_auc"]["sd"], 4), round(v["roc_auc"]["mean"], 3),
                  "EPV", round(v["epv"], 2), flush=True)
            Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=2)
    return res


if __name__ == "__main__":
    import sys
    only = sys.argv[1].split(",") if len(sys.argv) > 1 else None
    models = tuple(sys.argv[3].split(",")) if len(sys.argv) > 3 else ("logistic", "xgboost", "lightgbm")
    run(out=f"reports/repeated_evaluation_{sys.argv[2]}.json" if len(sys.argv) > 2 else "reports/repeated_evaluation.json", only=only, models=models)


def domain_separability(out="reports/survey_round_shift.json"):
    """How well household covariates alone separate MIS 2017 from MIS 2023 (5-fold CV AUC of a domain classifier).

    AUC near 0.5 means the rounds overlap and importance weighting can help; AUC near 1 means the covariate
    distributions barely overlap, so weighting cannot transport 2017 rows to 2023 (or variables are coded differently).
    """
    from sklearn.model_selection import cross_val_predict
    warnings.filterwarnings("ignore")
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]; hh = [c for c in F if c in HOUSEHOLD]
    d17 = load_2017(cols=hh)
    a = d[["y"] + hh].copy(); b = d17[["y"] + hh].copy()
    df = pd.concat([a, b], ignore_index=True); dom = np.r_[np.ones(len(a)), np.zeros(len(b))]
    prep = build_model_pipeline("logistic", df[hh], hh)[:-1].fit(df[hh], df.y.values)
    Z = prep.transform(df[hh])
    p = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"), Z.values, dom, cv=5,
                          method="predict_proba")[:, 1]
    lr = LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced").fit(Z.values, dom)
    top = sorted(zip(Z.columns, lr.coef_[0]), key=lambda t: -abs(t[1]))[:6]
    res = {"domain_auc": float(roc_auc_score(dom, p)), "n_2023": int(len(a)), "n_2017": int(len(b)),
           "top_separating_columns": [[c, float(v)] for c, v in top]}
    json.dump(res, open(out, "w"), indent=2)
    return res
