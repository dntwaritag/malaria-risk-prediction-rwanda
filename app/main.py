"""Malaria risk ranking service (FastAPI).

Serves a small web page and a JSON API around the trained logistic model.
No DHS person-level data and no coordinates are shipped: only the fitted model, its model card and
out-of-fold cluster risk scores.
"""
from __future__ import annotations

import io
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware

from src.pooled import add_cluster_means

BASE = Path(__file__).resolve().parent
MODEL_DIR = Path(os.environ.get("MODEL_DIR", BASE.parent / "models"))
SCORES_FILE = Path(os.environ.get("CLUSTER_SCORES", BASE / "cluster_scores.json"))
MAX_RECORDS = int(os.environ.get("MAX_RECORDS", "5000"))
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(5 * 1024 * 1024)))


class State:
    model = None
    card: dict = {}
    clusters: dict = {}


state = State()


def load_artifacts(model_dir: Path = MODEL_DIR, scores_file: Path = SCORES_FILE) -> None:
    model_path, card_path = Path(model_dir) / "malaria_risk_model.joblib", Path(model_dir) / "model_card.json"
    if model_path.exists() and card_path.exists():
        state.model = joblib.load(model_path)
        state.card = json.loads(card_path.read_text())
    if Path(scores_file).exists():
        state.clusters = json.loads(Path(scores_file).read_text())


@asynccontextmanager
async def lifespan(_: FastAPI):
    load_artifacts()
    yield


class SecurityHeaders(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        resp = await call_next(request)
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "DENY"
        resp.headers["Referrer-Policy"] = "no-referrer"
        if request.url.path not in ("/docs", "/redoc", "/openapi.json"):  # Swagger UI loads assets from a CDN
            resp.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'"
        return resp


app = FastAPI(title="Rwanda malaria risk ranking", version="1.0.0", lifespan=lifespan,
              description="Decision-support ranking of malaria risk from the DHS Rwanda MIS 2023. Scores are relative, not probabilities.")
app.add_middleware(SecurityHeaders)
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")


class ScoreRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1, description="DHS-coded household/person records; optional 'cluster' key groups them")


def require_model():
    if state.model is None:
        raise HTTPException(503, "Model artefacts are not loaded")


def score_frame(df: pd.DataFrame) -> dict:
    require_model()
    if len(df) == 0 or len(df) > MAX_RECORDS:
        raise HTTPException(422, f"Provide between 1 and {MAX_RECORDS} records")
    feats = state.card["input_features"]
    src_cols = state.card["cluster_mean_source_columns"]
    use = state.card["model_inputs"]
    df = df.copy()
    if "cluster" not in df:
        df["cluster"] = "all"
    df["cluster"] = df["cluster"].fillna("all").astype(str)
    ignored = [c for c in df.columns if c not in feats and c != "cluster"]
    missing = [c for c in feats if c not in df.columns]
    for c in feats:
        df[c] = pd.to_numeric(df[c], errors="coerce") if c in df else np.nan
    X = add_cluster_means(df[["cluster"] + feats], "cluster")
    for c in src_cols:
        if f"cm_{c}" not in X:
            X[f"cm_{c}"] = np.nan
    p = state.model.predict_proba(X[use])[:, 1]
    out = df[["cluster"]].copy(); out["risk_score"] = p
    byc = out.groupby("cluster").risk_score.agg(["mean", "size"]).sort_values("mean", ascending=False)
    clusters = [{"cluster": c, "mean_risk": round(float(r["mean"]), 4), "n": int(r["size"]), "rank": i + 1} for i, (c, r) in enumerate(byc.iterrows())]
    return {"n": int(len(out)), "scores_are_relative": True,
            "results": [{"index": i, "cluster": c, "risk_score": round(float(s), 4)} for i, (c, s) in enumerate(zip(out.cluster, out.risk_score))],
            "clusters": clusters, "ignored_fields": ignored, "missing_features": missing}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": state.model is not None}


@app.get("/api/model")
def model_card():
    if not state.card:
        raise HTTPException(503, "Model card not loaded")
    return state.card


@app.get("/api/clusters")
def clusters(tier: str | None = None):
    if not state.clusters:
        raise HTTPException(503, "Cluster scores not loaded")
    rows = state.clusters["clusters"]
    if tier:
        rows = [r for r in rows if r["tier"].lower() == tier.lower()]
    return {"note": state.clusters.get("note"), "generated": state.clusters.get("generated"), "clusters": rows}


@app.get("/api/template.csv", response_class=PlainTextResponse)
def template():
    require_model()
    return "cluster," + ",".join(state.card["input_features"]) + "\n"


@app.post("/api/score")
def score(req: ScoreRequest):
    return score_frame(pd.DataFrame(req.records))


@app.post("/api/score/csv")
async def score_csv(file: UploadFile = File(...)):
    raw = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File too large")
    try:
        df = pd.read_csv(io.BytesIO(raw))
    except Exception:
        raise HTTPException(422, "Could not read the file as CSV")
    return score_frame(df)


@app.get("/", include_in_schema=False)
def index():
    return FileResponse(BASE / "static" / "index.html")
