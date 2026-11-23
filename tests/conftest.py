"""Fixtures build a tiny model on SYNTHETIC data, so tests never need DHS microdata."""
import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.pipeline_builder import build_model_pipeline
from src.pooled import MEAN_COLS, add_cluster_means

HH = ['hv105', 'hv040', 'hv271', 'hv009', 'hv216', 'hml1', 'hml4', 'hv270', 'hv104', 'hv024', 'hv025', 'hv201', 'hv205', 'hv213',
      'hv214', 'hv215', 'hv226', 'hv206', 'hv207', 'hv208', 'hv209', 'hv210', 'hv211', 'hv212', 'hv221', 'hv247', 'hv227', 'hml10',
      'hml12', 'hv219']


def synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 4, n).astype(float) for c in HH})
    df["hv105"] = rng.integers(0, 80, n).astype(float); df["hv040"] = rng.normal(1600, 250, n)
    df["hv271"] = rng.normal(0, 90000, n); df["hv009"] = rng.integers(1, 10, n).astype(float)
    df["hv216"] = rng.integers(1, 6, n).astype(float); df["hml1"] = rng.integers(0, 4, n).astype(float)
    df["hv270"] = rng.integers(1, 6, n).astype(float); df["hv104"] = rng.integers(1, 3, n).astype(float)
    df["hv024"] = rng.integers(1, 6, n).astype(float); df["hv025"] = rng.integers(1, 3, n).astype(float)
    df["cluster"] = rng.integers(1, 31, n)
    df["y"] = (rng.random(n) < 0.05 + 0.03 * (df.cluster % 3 == 0)).astype(int)
    return df


@pytest.fixture(scope="session")
def artefacts(tmp_path_factory):
    d = tmp_path_factory.mktemp("art")
    df = synthetic(); X = add_cluster_means(df, "cluster")
    use = HH + [f"cm_{c}" for c in MEAN_COLS]
    pipe = build_model_pipeline("logistic", X[use], use).fit(X[use], X.y)
    joblib.dump(pipe, d / "malaria_risk_model.joblib")
    card = {"name": "test", "model": "logistic", "input_features": HH, "model_inputs": use, "cluster_mean_source_columns": MEAN_COLS,
            "training_data": {"n": 900, "positives": int(df.y.sum())},
            "evaluation": {"protocol": "test", "pr_auc_mean": 0.08, "pr_auc_sd": 0.03, "roc_auc_mean": 0.75, "lift_top20": 2.9, "baseline_pr_auc": 0.0096},
            "limitations": ["test limitation"]}
    (d / "model_card.json").write_text(json.dumps(card))
    (d / "scores.json").write_text(json.dumps({"note": "n", "generated": "2026-01-01", "clusters": [
        {"cluster": 1, "risk_score": 0.2, "rank": 1, "tier": "Highest"}, {"cluster": 2, "risk_score": 0.1, "rank": 2, "tier": "Lower"}]}))
    return d, df


@pytest.fixture()
def client(artefacts):
    from app import main
    d, _ = artefacts
    main.load_artifacts(d, d / "scores.json")
    return TestClient(main.app)
