"""Pre implementation data audit: dimensions, types, missingness, target balance.

Run:  python -m src.audit data/raw/mis2023 --out reports/audit_mis2023.json
Writes aggregate statistics only (no microdata), so the output is safe to commit.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import data as D


def infer_kind(col: str, s: pd.Series) -> str:
    if col in D.ORDINAL_PERSON:
        return "ordinal"
    if col in D.NOMINAL_PERSON:
        return "nominal"
    if col in D.DISCRETE_PERSON:
        return "discrete"
    if col in D.CONTINUOUS_PERSON or col in D.climate_feature_columns():
        return "continuous"
    return "other"


def audit(raw_dir: str = "data/raw/mis2023") -> dict:
    r = D.load_mis2023(raw_dir)
    raw, df, feats = r["raw_pr"], r["data"], r["feature_columns"]
    X = df[feats]
    n, p = X.shape
    miss = X.isna().sum()
    pos = int(df.y.sum())
    neg = int(len(df) - pos)
    w = df.weight
    out = {
        "source_file_shape": {"rows": int(raw.shape[0]), "columns": int(raw.shape[1])},
        "dropped_rows": r["dropped_rows"],
        "model_table": {"N": int(n), "P": int(p)},
        "overall_missing_pct": float(miss.sum() / (n * p) * 100),
        "complete_rows": int(X.notna().all(axis=1).sum()),
        "rows_with_missing": int(X.isna().any(axis=1).sum()),
        "missing_per_column": {c: {"count": int(miss[c]), "pct": round(float(miss[c] / n * 100), 3)} for c in feats},
        "kind_counts": pd.Series({c: infer_kind(c, X[c]) for c in feats}).value_counts().to_dict(),
        "kind_by_column": {c: infer_kind(c, X[c]) for c in feats},
        "target": {
            "n_valid_tests": int(len(df)), "positives": pos, "negatives": neg,
            "prevalence_pct": round(pos / len(df) * 100, 3),
            "imbalance_ratio_neg_to_pos": round(neg / pos, 1),
            "weighted_prevalence_pct": round(float(np.average(df.y, weights=w) * 100), 3),
            "clusters": int(df.cluster.nunique()),
            "clusters_with_a_positive": int(df[df.y == 1].cluster.nunique()),
            "positives_under_5": int(((df.hv105 < 5) & (df.y == 1)).sum()),
            "tests_under_5": int((df.hv105 < 5).sum()),
        },
        "excluded_with_reason": D.EXCLUDED_WITH_REASON,
    }
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", default="data/raw/mis2023")
    ap.add_argument("--out", default="reports/audit_mis2023.json")
    a = ap.parse_args()
    res = audit(a.raw_dir)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(json.dumps({k: res[k] for k in ("model_table", "overall_missing_pct", "complete_rows", "target")}, indent=1))
