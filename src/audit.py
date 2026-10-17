"""Pre implementation data audit for the Phase 1 report tables.

Run once the DHS PR file is available:
    python -m src.audit data/raw/<file>.dta --out phase1_strategy/audit.json

Produces N, P, dtype inventory, per column and overall missingness,
complete rows, and target imbalance. It reads the file but writes only
aggregate statistics (no microdata), so the output is safe to commit.
Nothing here has been run on real DHS data; it is tested on synthetic
data only (see tests in the Phase 2 notebook).
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src.transformers import DHSSentinelRecoder, TARGET_COLUMN_RDT


def audit(df: pd.DataFrame, target: str = TARGET_COLUMN_RDT, max_discrete: int = 20) -> dict:
    df = DHSSentinelRecoder().transform(df)
    n, p = df.shape
    miss = df.isna().sum()
    kinds = {}
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            kinds[c] = "datetime"
        elif isinstance(s.dtype, pd.CategoricalDtype) or s.dtype == object:
            kinds[c] = "categorical (nominal or ordinal, decide from codebook)"
        elif pd.api.types.is_integer_dtype(s) or s.dropna().mod(1).eq(0).all():
            kinds[c] = "discrete" if s.nunique() > 2 else "binary or categorical code"
        else:
            kinds[c] = "continuous"
    out = {
        "N": int(n), "P": int(p),
        "overall_missing_pct": float(miss.sum() / (n * p) * 100),
        "complete_rows": int(df.notna().all(axis=1).sum()),
        "rows_with_missing": int(df.isna().any(axis=1).sum()),
        "missing_per_column": {c: {"count": int(miss[c]), "pct": float(miss[c] / n * 100)}
                               for c in df.columns},
        "inferred_kind": kinds,
    }
    if target in df.columns:
        y = df[target].dropna()
        pos, neg = int((y == 1).sum()), int((y == 0).sum())
        out["target"] = {"n_labeled": int(len(y)), "positives": pos, "negatives": neg,
                         "prevalence_pct": float(pos / max(len(y), 1) * 100),
                         "imbalance_ratio": float(neg / pos) if pos else None}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--out", default="phase1_strategy/audit.json")
    a = ap.parse_args()
    frame = pd.read_stata(a.path, convert_categoricals=False)
    with open(a.out, "w") as f:
        json.dump(audit(frame), f, indent=2)
