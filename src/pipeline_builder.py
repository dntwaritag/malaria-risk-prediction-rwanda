"""Leakage free preprocessing and model pipelines.

Everything that learns a statistic (winsorizing bounds, imputation values, category frequencies,
scaling parameters, collinearity drops) is a pipeline step, so calling ``fit`` on training data only
is enough to keep validation and test data untouched. The column groups are decided from training
data (``group_numeric_columns``) and never from the full table.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from src import data as D
from src.transformers import (
    CollinearityScreen,
    IQRWinsorizer,
    Log1pClip,
    NominalCaster,
    RareCategoryGrouper,
    ZScoreWinsorizer,
)

SEED = 42
SKEW_THRESHOLD = 0.5


def split_train_test(data: pd.DataFrame, test_size: float = 0.2, seed: int = SEED, n_seeds: int = 30):
    """Cluster grouped, label stratified split. No survey cluster appears in both parts.

    With few positive clusters a single grouped split can leave the test set with a very different
    prevalence. The seed is therefore chosen, from ``seed`` .. ``seed + n_seeds - 1``, as the one whose
    test prevalence is closest to the overall prevalence. The choice uses labels only, never model
    performance, so it cannot leak model information. The chosen seed is returned.
    """
    n_splits = int(round(1 / test_size))
    overall = data["y"].mean()
    best = None
    for s in range(seed, seed + n_seeds):
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=s)
        train_idx, test_idx = next(sgkf.split(data, data["y"], groups=data["cluster"]))
        gap = abs(data["y"].iloc[test_idx].mean() - overall) + abs(len(test_idx) / len(data) - test_size)
        if best is None or gap < best[0]:
            best = (gap, s, train_idx, test_idx)
    _, used_seed, train_idx, test_idx = best
    train = data.iloc[train_idx].reset_index(drop=True)
    test = data.iloc[test_idx].reset_index(drop=True)
    train.attrs["split_seed"] = used_seed
    return train, test


def group_numeric_columns(X_train: pd.DataFrame, numeric_cols: list[str], threshold: float = SKEW_THRESHOLD) -> dict:
    """Assign each numeric column to a treatment using TRAINING skewness only.

    right_skewed  : skew > threshold  -> IQR winsorize, median impute, log1p, standardize
    other_skewed  : skew < -threshold -> IQR winsorize, median impute, standardize
    gaussian      : otherwise         -> Z score winsorize, mean impute, standardize
    """
    sk = X_train[numeric_cols].skew()
    return {
        "right_skewed": [c for c in numeric_cols if sk[c] > threshold and X_train[c].min() >= 0],
        "other_skewed": [c for c in numeric_cols if abs(sk[c]) > threshold and not (sk[c] > threshold and X_train[c].min() >= 0)],
        "gaussian": [c for c in numeric_cols if abs(sk[c]) <= threshold],
        "skewness": sk.round(3).to_dict(),
    }


def build_preprocessor(groups: dict, ordinal_cols: list[str], nominal_cols: list[str], scaler: str = "standard") -> ColumnTransformer:
    scale = StandardScaler if scaler == "standard" else MinMaxScaler
    transformers = []
    if groups["right_skewed"]:
        transformers.append(("right_skewed", Pipeline([
            ("winsorize", IQRWinsorizer(factor=1.5)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("log1p", Log1pClip()),
            ("scale", scale()),
        ]), groups["right_skewed"]))
    if groups["other_skewed"]:
        transformers.append(("other_skewed", Pipeline([
            ("winsorize", IQRWinsorizer(factor=1.5)),
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", scale()),
        ]), groups["other_skewed"]))
    if groups["gaussian"]:
        transformers.append(("gaussian", Pipeline([
            ("winsorize", ZScoreWinsorizer(z_threshold=3.0)),
            ("impute", SimpleImputer(strategy="mean", add_indicator=True)),
            ("scale", scale()),
        ]), groups["gaussian"]))
    if ordinal_cols:
        transformers.append(("ordinal", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            # explicit rank, never the alphabetical default order
            ("encode", OrdinalEncoder(categories=[[float(v) for v in D.WEALTH_QUINTILE_ORDER]] * len(ordinal_cols),
                                      handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), ordinal_cols))
    if nominal_cols:
        transformers.append(("nominal", Pipeline([
            ("cast", NominalCaster()),
            ("group_rare", RareCategoryGrouper(min_frequency=0.01)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False, drop="if_binary")),
        ]), nominal_cols))
    ct = ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)
    return ct.set_output(transform="pandas")


def numeric_columns(feature_columns: list[str]) -> list[str]:
    return [c for c in feature_columns if c in D.CONTINUOUS_PERSON + D.DISCRETE_PERSON + D.climate_feature_columns() or c.startswith("cm_")]


def make_preprocessor_for(X_train: pd.DataFrame, feature_columns: list[str], scaler: str = "standard"):
    num = numeric_columns(feature_columns)
    groups = group_numeric_columns(X_train, num)
    ordinal = [c for c in D.ORDINAL_PERSON if c in feature_columns]
    nominal = [c for c in D.NOMINAL_PERSON if c in feature_columns]
    return build_preprocessor(groups, ordinal, nominal, scaler=scaler), groups, num


def build_model_pipeline(model_name: str, X_train: pd.DataFrame, feature_columns: list[str], scaler: str = "standard") -> Pipeline:
    """Full pipeline: preprocessing, optional collinearity screen (linear models only), estimator."""
    prep, groups, num = make_preprocessor_for(X_train, feature_columns, scaler)
    steps = [("prep", prep)]
    if model_name == "logistic":
        steps.append(("screen", CollinearityScreen(corr_threshold=0.85, vif_threshold=10.0, candidate_columns=num)))
        est = LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, random_state=SEED)
    elif model_name == "random_forest":
        est = RandomForestClassifier(n_estimators=400, min_samples_leaf=20, class_weight="balanced_subsample",
                                     n_jobs=-1, random_state=SEED)
    elif model_name == "lightgbm":
        from lightgbm import LGBMClassifier
        est = LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=40,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=5.0,
                             class_weight="balanced", random_state=SEED, verbose=-1)
    elif model_name == "xgboost":
        from xgboost import XGBClassifier
        est = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3, subsample=0.8,
                            colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
                            scale_pos_weight=100.0, eval_metric="aucpr", random_state=SEED, n_jobs=2)
    else:
        raise ValueError(model_name)
    steps.append(("model", est))
    return Pipeline(steps)
