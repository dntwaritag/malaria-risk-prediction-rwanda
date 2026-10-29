"""Model evaluation: cluster-aware out-of-fold predictions, calibration, weighted metrics, bootstrap CIs."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, brier_score_loss, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold

from src.pipeline_builder import build_model_pipeline

MODELS = ["logistic", "random_forest", "lightgbm", "xgboost"]


def oof_predictions(model_name, train, feature_columns, n_splits=5, seed=42):
    """Out-of-fold probabilities; every fold keeps whole clusters together (no cluster spans train/valid)."""
    X, y, g = train[feature_columns], train["y"].values, train["cluster"].values
    oof = np.zeros(len(train))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for tr, va in cv.split(X, y, g):
        pipe = build_model_pipeline(model_name, X.iloc[tr], feature_columns)
        pipe.fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])[:, 1]
    return oof


class Calibrator:
    """Platt (sigmoid on logit score) or isotonic, fitted only on out-of-fold predictions."""
    def __init__(self, kind="platt"):
        self.kind = kind

    @staticmethod
    def _logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p, y, w=None):
        if self.kind == "platt":
            self.m = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p), y, sample_weight=w)
        else:
            self.m = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y, sample_weight=w)
        return self

    def predict(self, p):
        return self.m.predict_proba(self._logit(p))[:, 1] if self.kind == "platt" else self.m.predict(p)


def metrics(y, p, w=None, thr=0.5):
    pred = (p >= thr).astype(int)
    out = {"pr_auc": average_precision_score(y, p, sample_weight=w), "roc_auc": roc_auc_score(y, p, sample_weight=w),
           "brier": brier_score_loss(y, p, sample_weight=w), "recall": recall_score(y, pred, sample_weight=w, zero_division=0),
           "precision": precision_score(y, pred, sample_weight=w, zero_division=0),
           "f1": f1_score(y, pred, sample_weight=w, zero_division=0)}
    return {k: float(v) for k, v in out.items()}


def choose_threshold(y, p, target_recall=0.70):
    """Operational threshold chosen on out-of-fold data: highest threshold that still reaches the recall target."""
    cand = np.unique(np.quantile(p, np.linspace(0.5, 0.999, 400)))
    best = cand[0]
    for t in cand:
        if recall_score(y, (p >= t).astype(int), zero_division=0) >= target_recall:
            best = t
    return float(best)


def cluster_bootstrap(y, p, w, clusters, thr, n_boot=500, seed=42):
    rng = np.random.default_rng(seed)
    ids = np.unique(clusters)
    idx_by = {c: np.where(clusters == c)[0] for c in ids}
    rows = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        ix = np.concatenate([idx_by[c] for c in pick])
        if y[ix].sum() < 2:
            continue
        for tag, ww in (("unweighted", None), ("weighted", w[ix])):
            m = metrics(y[ix], p[ix], ww, thr)
            rows.append({"kind": tag, **m})
    df = pd.DataFrame(rows)
    return {k: {m: [float(df[df.kind == k][m].quantile(.025)), float(df[df.kind == k][m].quantile(.975))]
                for m in ["pr_auc", "roc_auc", "brier", "recall", "precision", "f1"]} for k in ("unweighted", "weighted")}


def run_evaluation(train, test, feature_columns, models=MODELS, out_path="reports/evaluation.json", n_boot=500):
    warnings.filterwarnings("ignore")
    y_tr, y_te = train["y"].values, test["y"].values
    w_tr, w_te = train["weight"].values, test["weight"].values
    base = float(y_te.mean())
    results = {"test_positives": int(y_te.sum()), "test_n": int(len(y_te)), "baseline_pr_auc": base, "models": {}}
    preds = {}
    for name in models:
        oof = oof_predictions(name, train, feature_columns)
        cal = Calibrator("platt").fit(oof, y_tr)
        cal_iso = Calibrator("isotonic").fit(oof, y_tr)
        thr = choose_threshold(y_tr, cal.predict(oof))
        pipe = build_model_pipeline(name, train[feature_columns], feature_columns).fit(train[feature_columns], y_tr)
        raw = pipe.predict_proba(test[feature_columns])[:, 1]
        variants = {"raw": raw, "platt": cal.predict(raw), "isotonic": cal_iso.predict(raw)}
        thr_raw = choose_threshold(y_tr, oof)
        entry = {"oof": {"pr_auc": float(average_precision_score(y_tr, oof)), "roc_auc": float(roc_auc_score(y_tr, oof)),
                         "brier_raw": float(brier_score_loss(y_tr, oof)),
                         "brier_platt": float(brier_score_loss(y_tr, cal.predict(oof)))},
                 "threshold_platt": thr, "threshold_raw": thr_raw, "test": {}}
        for v, p in variants.items():
            t = thr if v == "platt" else (thr_raw if v == "raw" else choose_threshold(y_tr, cal_iso.predict(oof)))
            entry["test"][v] = {"unweighted": metrics(y_te, p, None, t), "weighted": metrics(y_te, p, w_te, t)}
        entry["bootstrap_ci_platt"] = cluster_bootstrap(y_te, variants["platt"], w_te, test["cluster"].values, thr, n_boot)
        # calibration curve bins on pooled OOF (quantile bins)
        q = pd.qcut(cal.predict(oof), 8, duplicates="drop")
        cc = pd.DataFrame({"p": cal.predict(oof), "y": y_tr, "b": q}).groupby("b", observed=True).agg(p=("p", "mean"), y=("y", "mean"), n=("y", "size"))
        entry["calibration_bins"] = cc.reset_index(drop=True).to_dict("list")
        results["models"][name] = entry
        preds[name] = variants["platt"]
        print(name, "test PR-AUC", round(entry["test"]["platt"]["unweighted"]["pr_auc"], 4), "ROC", round(entry["test"]["platt"]["unweighted"]["roc_auc"], 3))
    # geographic robustness: province-level ROC/PR for best model by OOF PR-AUC
    best = max(results["models"], key=lambda m: results["models"][m]["oof"]["pr_auc"])
    results["best_by_oof_pr_auc"] = best
    reg = {}
    for r, df in test.assign(p=preds[best]).groupby("hv024"):
        reg[str(r)] = {"n": int(len(df)), "positives": int(df.y.sum()),
                       "mean_pred": float(df.p.mean()), "obs_rate": float(df.y.mean())}
    results["province_test_best"] = reg
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(out_path, "w"), indent=2)
    return results, preds
