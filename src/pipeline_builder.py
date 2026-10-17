"""Modular ColumnTransformer and Pipeline definitions.

This module composes the custom transformers in transformers.py plus
standard Scikit Learn components into a single leakage resistant
preprocessing pipeline. The guiding rule throughout: build the pipeline
object once, then call .fit() on it only with X_train (or only within
a single cross validation training fold). Never call .fit() or
.fit_transform() on the full, unsplit dataset. Calling .transform()
(not .fit_transform()) on X_test or X_val is always safe, since no
statistic is learned at that point, only applied.

COLUMN LISTS ARE PLACEHOLDERS. See transformers.py for the same
warning. Update CONTINUOUS_COLUMNS, NOMINAL_COLUMNS, and
ORDINAL_COLUMNS below once the actual Rwanda PR file codebook has been
inspected in the Phase 2 notebook's exploratory data analysis section,
and document the final, verified lists in the Phase 1 Strategy Report
and the Phase 2 Decision Log, including which columns were dropped and
why.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
    MinMaxScaler,
)

from src.transformers import (
    DHSSentinelRecoder,
    RareCategoryGrouper,
    IQRWinsorizer,
    ZScoreWinsorizer,
)


# ---------------------------------------------------------------------------
# PLACEHOLDER COLUMN GROUPS. Replace with verified columns from the real
# codebook before trusting any pipeline output.
# ---------------------------------------------------------------------------

# Continuous features expected to be right skewed (count like or bounded
# below at zero): candidate for the IQR winsorizer and, if modeling with a
# distance based or linear algorithm, StandardScaler after a log transform
# is examined in Phase 2 (see Figure 4.1, the transformation stage grid).
CONTINUOUS_SKEWED_COLUMNS: list[str] = [
    # "hc1",   # child's age in months: example only, confirm distribution
]

# Continuous features expected to be approximately Gaussian: candidate for
# the Z score winsorizer and StandardScaler.
CONTINUOUS_GAUSSIAN_COLUMNS: list[str] = []

# Ordinal categorical features with a defined rank (for example a wealth
# index quintile): OrdinalEncoder with an explicit, manually specified
# category order, never the alphabetical default order.
ORDINAL_COLUMNS: list[str] = [
    # "hv270",  # household wealth index quintile: example only, confirm coding
]

# Nominal categorical features with no inherent order (for example region,
# residence type): RareCategoryGrouper then OneHotEncoder.
NOMINAL_COLUMNS: list[str] = [
    # "hv025",  # type of residence, urban/rural: example only, confirm coding
    # "hv024",  # region/province: example only, confirm coding
]


def build_preprocessing_pipeline(
    scaler: str = "standard",
    imputer_strategy: str = "median",
    rare_category_min_frequency: float = 0.01,
) -> Pipeline:
    """Build the full, leakage resistant preprocessing pipeline.

    Parameters
    ----------
    scaler : {"standard", "minmax"}
        "standard" applies StandardScaler, appropriate when the target
        model is distance based (KNN, SVM) or a regularized linear
        model (Lasso, Ridge) and the continuous features are
        approximately Gaussian after any needed transformation.
        "minmax" applies MinMaxScaler, appropriate when a bounded
        [0, 1] range is required (for example feeding a neural
        network) or when preserving exact zero values matters.
        This choice must be justified in the Phase 1 report against
        the actual distribution shapes found in exploratory analysis,
        not left at a library default.
    imputer_strategy : {"median", "mean", "knn"}
        "median" is the safer default for skewed continuous features,
        since the mean is pulled by outliers. "knn" uses KNNImputer,
        which borrows information from similar rows and is worth
        comparing against median imputation if missingness is
        substantial and plausibly MAR (related to other observed
        features) rather than MCAR.
    rare_category_min_frequency : float
        Passed through to RareCategoryGrouper. See that class's
        docstring for why this threshold must be chosen from the
        actual category frequency table, not guessed.

    Returns
    -------
    sklearn.pipeline.Pipeline
        A single pipeline whose .fit() must be called only on X_train.
    """
    scaler_obj = StandardScaler() if scaler == "standard" else MinMaxScaler()
    numeric_imputer = (
        KNNImputer(n_neighbors=5)
        if imputer_strategy == "knn"
        else SimpleImputer(strategy=imputer_strategy)
    )

    skewed_pipeline = Pipeline(
        steps=[
            ("winsorize", IQRWinsorizer(columns=CONTINUOUS_SKEWED_COLUMNS)),
            ("impute", numeric_imputer),
            ("scale", scaler_obj),
        ]
    )

    gaussian_pipeline = Pipeline(
        steps=[
            ("winsorize", ZScoreWinsorizer(columns=CONTINUOUS_GAUSSIAN_COLUMNS)),
            ("impute", numeric_imputer),
            ("scale", scaler_obj),
        ]
    )

    ordinal_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            # TODO: replace categories="auto" with an explicit, manually
            # ordered category list per ordinal column once the real
            # coding scheme (e.g. wealth quintile labels) is confirmed.
            # Using the automatic alphabetical order on an ordinal
            # column would silently produce a meaningless rank.
            ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    nominal_pipeline = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("group_rare", RareCategoryGrouper(min_frequency=rare_category_min_frequency)),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("skewed_continuous", skewed_pipeline, CONTINUOUS_SKEWED_COLUMNS),
            ("gaussian_continuous", gaussian_pipeline, CONTINUOUS_GAUSSIAN_COLUMNS),
            ("ordinal", ordinal_pipeline, ORDINAL_COLUMNS),
            ("nominal", nominal_pipeline, NOMINAL_COLUMNS),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    full_pipeline = Pipeline(
        steps=[
            ("recode_sentinels", DHSSentinelRecoder()),
            ("column_transform", column_transformer),
        ]
    )

    return full_pipeline
