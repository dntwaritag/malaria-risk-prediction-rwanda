"""Stage 2 model improvement: nested, repeated, cluster-grouped tuning of the leading candidates.

Why: the pooled evaluation (src/pooled.py) compared fixed configurations on one split. Here hyperparameters are chosen
by an inner cluster-grouped CV that only sees the outer training fold, and the whole procedure is repeated over several
split seeds, so the reported PR-AUC is not tuned on the data it is scored on and the spread across seeds shows how
much the ranking depends on the split.
Candidates: L2 logistic, elastic-net logistic, shallow gradient boosting, and a rank-average ensemble of the best two.
Feature sets: household (B), household + cluster means (C), household + extended cluster context (X).
Preprocessing and the collinearity screen are fit on each outer training fold only; the inner search reuses that
fitted matrix (hyperparameter choice only), while outer validation rows never touch any fitted statistic.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src import data as D
from src.evaluation import cluster_bootstrap, Calibrator
from src.pipeline_builder import build_model_pipeline
from src.pooled import HOUSEHOLD, MEAN_COLS, add_cluster_means

SEED = 42
OUTER_SEEDS = (42, 7)


def add_extended_context(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Label-free cluster context: means of numeric household columns, wealth spread, cluster size."""
    out = df.copy()
    num = [c for c in cols if c in out and pd.api.types.is_numeric_dtype(out[c])]
    g = out.groupby("cluster")
    for c in num:
        out[f"cm_{c}"] = g[c].transform("mean")
    if "hv271" in out:
        out["cm_sd_hv271"] = g["hv271"].transform("std").fillna(0.0)
    out["cm_n"] = g["cluster"].transform("size")
    return out


def _lr(C, penalty="l2"):
    if penalty == "l2":
        return LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    return LogisticRegression(C=C, penalty="elasticnet", l1_ratio=0.5, solver="saga",
                              class_weight="balanced", max_iter=300, tol=1e-3, random_state=SEED)


def _xgb(depth, n):
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=n, learning_rate=0.03, max_depth=depth, subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=20, reg_lambda=10.0, scale_pos_weight=100.0, eval_metric="aucpr",
                         random_state=SEED, n_jobs=2)


GRIDS = {
    "logit_l2": [("C=%g" % c, (lambda c=c: _lr(c))) for c in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3)],
    "logit_enet": [("C=%g" % c, (lambda c=c: _lr(c, "enet"))) for c in (0.03, 0.1)],
    "xgb_shallow": [("d%d_n%d" % (d, n), (lambda d=d, n=n: _xgb(d, n))) for d in (1, 2) for n in (100, 300)],
}


def _inner_select(Z, y, g, make_grid, seed):
    best, best_score = None, -1.0
    for name, mk in make_grid:
        oof = np.zeros(len(y))
        for tr, va in StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed).split(Z, y, g):
            m = mk().fit(Z.iloc[tr], y[tr]); oof[va] = m.predict_proba(Z.iloc[va])[:, 1]
        s = average_precision_score(y, oof)
        if s > best_score:
            best, best_score = (name, mk), s
    return best


def _rank01(p):
    return (rankdata(p) - 1) / (len(p) - 1)


def nested_run(d23: pd.DataFrame, feats: list[str], context: str, outer_seed: int):
    X = d23[["cluster", "y", "weight"] + feats].copy()
    use = list(feats)
    if context == "means":
        X = add_cluster_means(X, "cluster"); use += [f"cm_{c}" for c in MEAN_COLS if c in X]
    elif context == "extended":
        X = add_extended_context(X, feats); use += [c for c in X.columns if c.startswith("cm_")]
    y, g = X.y.values, X.cluster.values
    oofs = {k: np.zeros(len(X)) for k in list(GRIDS) + ["ensemble"]}
    chosen = {k: [] for k in GRIDS}
    for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=outer_seed).split(X, y, g):
        pipe = build_model_pipeline("logistic", X.iloc[tr][use], use)
        prep = pipe[:-1].fit(X.iloc[tr][use], y[tr])
        Ztr, Zva = prep.transform(X.iloc[tr][use]), prep.transform(X.iloc[va][use])
        preds = {}
        for k, grid in GRIDS.items():
            name, mk = _inner_select(Ztr, y[tr], g[tr], grid, outer_seed)
            chosen[k].append(name)
            preds[k] = mk().fit(Ztr, y[tr]).predict_proba(Zva)[:, 1]
            oofs[k][va] = preds[k]
        oofs["ensemble"][va] = 0.5 * _rank01(preds["logit_l2"]) + 0.5 * _rank01(preds["xgb_shallow"])
    return X, y, g, oofs, chosen


def _metrics(y, p, w, g):
    thr = float(np.quantile(p, 0.8))
    return {"pr_auc": float(average_precision_score(y, p)), "roc_auc": float(roc_auc_score(y, p)),
            "pr_auc_weighted": float(average_precision_score(y, p, sample_weight=w)),
            "top20_recall": float(y[p >= thr].sum() / y.sum()), "lift_top20": float(y[p >= thr].mean() / y.mean())}


def run(out="reports/tuning_evaluation.json", contexts=("none", "means", "extended")):
    warnings.filterwarnings("ignore")
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    hh = [c for c in F if c in HOUSEHOLD]
    res = {"n": int(len(d)), "positives": int(d.y.sum()), "baseline_pr_auc": float(d.y.mean()),
           "outer_seeds": list(OUTER_SEEDS), "sets": {}}
    for ctx in contexts:
        res["sets"][ctx] = {}
        runs = {}
        for s in OUTER_SEEDS:
            X, y, g, oofs, chosen = nested_run(d, hh, ctx, s)
            for k, p in oofs.items():
                runs.setdefault(k, []).append(_metrics(y, p, X.weight.values, g))
                if s == OUTER_SEEDS[0]:
                    res["sets"][ctx].setdefault("_oof_seed0", {})[k] = None
            res["sets"][ctx].setdefault("chosen_seed0", chosen) if s == OUTER_SEEDS[0] else None
            for k in oofs:
                print(ctx, s, k, round(runs[k][-1]["pr_auc"], 4), round(runs[k][-1]["roc_auc"], 3), flush=True)
        for k, lst in runs.items():
            res["sets"][ctx][k] = {m: {"mean": float(np.mean([x[m] for x in lst])), "sd": float(np.std([x[m] for x in lst]))}
                                   for m in lst[0]}
        res["sets"][ctx].pop("_oof_seed0", None)
        Path(out).parent.mkdir(parents=True, exist_ok=True); json.dump(res, open(out, "w"), indent=2)
    return res


if __name__ == "__main__":
    import sys
    ctx = tuple(sys.argv[1].split(",")) if len(sys.argv) > 1 else ("none", "means", "extended")
    run(out=f"reports/tuning_{'_'.join(ctx)}.json", contexts=ctx)


def c_path(out="reports/tuning_c_path.json", Cs=(0.003, 0.01, 0.03, 0.1, 0.3, 1.0)):
    """Sensitivity of the household + cluster-mean logistic to regularization strength (same outer folds per seed)."""
    warnings.filterwarnings("ignore")
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]; hh = [c for c in F if c in HOUSEHOLD]
    X = add_cluster_means(d[["cluster", "y", "weight"] + hh].copy(), "cluster")
    use = hh + [f"cm_{c}" for c in MEAN_COLS if c in X]
    y, g = X.y.values, X.cluster.values
    res = {}
    for s in OUTER_SEEDS:
        oofs = {C: np.zeros(len(X)) for C in Cs}
        for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s).split(X, y, g):
            prep = build_model_pipeline("logistic", X.iloc[tr][use], use)[:-1].fit(X.iloc[tr][use], y[tr])
            Ztr, Zva = prep.transform(X.iloc[tr][use]), prep.transform(X.iloc[va][use])
            for C in Cs:
                oofs[C][va] = _lr(C).fit(Ztr, y[tr]).predict_proba(Zva)[:, 1]
        for C in Cs:
            res.setdefault(str(C), {})[str(s)] = {"pr_auc": float(average_precision_score(y, oofs[C])),
                                                 "roc_auc": float(roc_auc_score(y, oofs[C]))}
            print(s, C, round(res[str(C)][str(s)]["pr_auc"], 4), round(res[str(C)][str(s)]["roc_auc"], 3), flush=True)
    json.dump(res, open(out, "w"), indent=2)
    return res
