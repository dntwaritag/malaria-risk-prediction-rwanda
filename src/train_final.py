"""Train the final deployable model and export the artefacts used by the web app.

Model: the headline configuration (household features + cluster means, regularized class-weighted logistic regression).

Outputs (regenerated, never committed, because they are derived from DHS microdata):
  models/malaria_risk_model.joblib   fitted scikit-learn pipeline
  models/model_card.json             features, evaluation numbers (from reports/repeated_evaluation.json), limits
  app/cluster_scores.json            out-of-fold risk score and rank per survey cluster (no coordinates, no person rows)

Usage:  python -m src.train_final
"""
from __future__ import annotations

import json
import platform
import sys
import warnings
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from src import data as D
from src.pipeline_builder import build_model_pipeline
from src.pooled import HOUSEHOLD, MEAN_COLS, add_cluster_means

SEEDS = range(100, 105)   # same seeds as the repeated evaluation
TIERS = [(0.20, "Highest"), (0.50, "High"), (0.80, "Medium"), (1.01, "Lower")]


def tier_for(frac_rank: float) -> str:
    return next(name for cut, name in TIERS if frac_rank < cut)


def main(out_models="models", out_app="app"):
    warnings.filterwarnings("ignore")
    r = D.load_mis2023(); d = r["data"].reset_index(drop=True); F = r["feature_columns"]
    hh = [c for c in F if c in HOUSEHOLD]
    X = add_cluster_means(d[["cluster", "y", "weight"] + hh], "cluster")
    use = hh + [f"cm_{c}" for c in MEAN_COLS if c in X]
    y, g = X.y.values, X.cluster.values

    # 1) final model on all data
    pipe = build_model_pipeline("logistic", X[use], use).fit(X[use], y)

    # 2) out-of-fold scores (each person scored by models that never saw their cluster), averaged over 5 repeats
    oof = np.zeros(len(X))
    for seed in SEEDS:
        for tr, va in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(X, y, g):
            p = build_model_pipeline("logistic", X.iloc[tr][use], use).fit(X.iloc[tr][use], y[tr])
            oof[va] += p.predict_proba(X.iloc[va][use])[:, 1] / len(SEEDS)
    cs = pd.DataFrame({"cluster": g, "s": oof}).groupby("cluster").s.mean().sort_values(ascending=False)
    n = len(cs)
    clusters = [{"cluster": int(c), "risk_score": round(float(s), 4), "rank": i + 1, "tier": tier_for(i / n)}
                for i, (c, s) in enumerate(cs.items())]

    rep = json.load(open("reports/repeated_evaluation.json"))
    m = rep["configs"]["C_household_cluster_means"]["logistic"]
    card = {
        "name": "Rwanda malaria risk ranking model", "version": date.today().isoformat(),
        "model": "Regularized (L2, C=0.1), class-weighted logistic regression after a collinearity screen",
        "task": "Rank people / survey clusters by relative malaria risk (RDT result hml35) for targeting prevention",
        "training_data": {"source": "DHS Rwanda Malaria Indicator Survey 2023", "n": int(rep["n"]), "positives": int(rep["positives"])},
        "input_features": hh, "cluster_mean_features": [f"cm_{c}" for c in MEAN_COLS if c in X], "model_inputs": use,
        "cluster_mean_source_columns": [c for c in MEAN_COLS if c in X],
        "evaluation": {"protocol": "5 repeats x 5-fold StratifiedGroupKFold by cluster (seeds 100-104)",
                       "pr_auc_mean": round(m["pr_auc"]["mean"], 4), "pr_auc_sd": round(m["pr_auc"]["sd"], 4),
                       "roc_auc_mean": round(m["roc_auc"]["mean"], 3), "lift_top20": round(m["lift_top20"]["mean"], 2),
                       "baseline_pr_auc": round(rep["baseline_pr_auc"], 4), "events_per_variable": round(m["epv"], 2)},
        "limitations": [
            "Scores come from a class-weighted model, so they run much higher than real infection rates: a score of 0.8 does NOT mean an 80% chance of malaria. Use them only to rank.",
            "Only 179 positive tests: uncertainty is wide (PR-AUC sd 0.03 across repeats).",
            "Trained on one survey round (Jul-Dec 2023); performance on other years or districts is unknown.",
            "Use as decision support alongside local knowledge, not as a diagnosis tool."],
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    Path(out_models).mkdir(exist_ok=True); Path(out_app).mkdir(exist_ok=True)
    joblib.dump(pipe, f"{out_models}/malaria_risk_model.joblib", compress=3)
    json.dump(card, open(f"{out_models}/model_card.json", "w"), indent=2)
    json.dump({"generated": date.today().isoformat(), "note": "Out-of-fold relative risk scores per survey cluster. No coordinates or person-level data.",
               "clusters": clusters}, open(f"{out_app}/cluster_scores.json", "w"))
    print("saved model + card + cluster scores for", n, "clusters")


if __name__ == "__main__":
    main()
