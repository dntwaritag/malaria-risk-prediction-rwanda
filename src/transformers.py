"""Custom Scikit Learn transformers for the Rwanda DHS malaria pipeline.

Every transformer here exists because a generic library default would
silently do the wrong thing on this specific dataset. Each docstring
states which real characteristic of DHS survey data motivates it.
All transformers follow the fit on train only contract: fit() must only
ever see X_train (or a single cross validation training fold), never the
full dataset, so that no information from validation or test rows leaks
into a learned statistic (a threshold, a mean, a category frequency).

IMPORTANT, READ BEFORE USE:
The exact column names and sentinel missing codes below are drafted from
DHS's publicly documented Recode Manual conventions and the DHS Guide to
Statistics page for malaria prevalence in children. They have not yet
been verified against the actual Rwanda PR (Person Recode) file, because
that file had not yet been approved and downloaded at the time this
module was written. Treat every value in DHS_SENTINEL_CODES and every
column name constant as a placeholder to confirm against your own
codebook (the .do file or value label file that ships with your DHS
download) before trusting any output from this module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# ---------------------------------------------------------------------------
# PLACEHOLDER CONFIGURATION. Verify every entry against the real codebook.
# ---------------------------------------------------------------------------

# DHS Person Recode (PR) file, standard variable names per the DHS Guide to
# Statistics "Prevalence of Malaria in Children" page. Confirm these exist
# with these exact names in the Rwanda file you were approved for; variable
# availability can vary slightly by survey round.
TARGET_COLUMN_RDT = "hml35"       # Result of malaria rapid diagnostic test
TARGET_COLUMN_MICROSCOPY = "hml32"  # Final result of malaria from blood smear
ELIGIBILITY_AGE_MONTHS = "hc1"      # Child's age in months
ELIGIBILITY_SLEPT_HERE = "hv103"    # Slept in household last night (1 = yes)
ELIGIBILITY_SELECTED_FOR_TEST = "hv042"  # Household selected for testing module
SAMPLE_WEIGHT = "hv005"             # Household sample weight, for weighted stats

# DHS numeric sentinel codes that are NOT the same as a true blank cell in
# the file, but represent a documented, non random missingness category
# (child not present, refused, test not completed, equipment failure, etc).
# pandas read_stata / read_spss will load these as ordinary integers, not as
# NaN, unless you recode them explicitly. Getting this wrong means your
# missingness audit in Phase 1 will undercount real missing data.
# These specific integer codes are the DHS wide convention for this kind of
# field (commonly 6 = other/inconclusive, 9 = missing); confirm the exact
# values for hml35 and hml32 in your codebook before relying on this dict.
DHS_SENTINEL_CODES = {
    TARGET_COLUMN_RDT: [9],
    TARGET_COLUMN_MICROSCOPY: [6, 9],
}


class DHSSentinelRecoder(BaseEstimator, TransformerMixin):
    """Replace DHS specific sentinel missing codes with np.nan.

    Why this exists: DHS recode files use small integer codes (for
    example 9 for missing, or 6 for inconclusive) to represent
    non random, documented missingness categories within an otherwise
    valid looking integer column. A generic .isna() check, or an
    imputer applied straight out of the box, would treat these
    sentinel integers as real observed values (for example, would
    average a 9 into a mean), silently corrupting every downstream
    statistic. This transformer must run before any imputation step.

    Parameters
    ----------
    sentinel_map : dict[str, list[int]]
        Maps a column name to the list of integer codes in that column
        which represent missingness rather than a real observed value.
    """

    def __init__(self, sentinel_map: dict | None = None):
        self.sentinel_map = sentinel_map or DHS_SENTINEL_CODES

    def fit(self, X: pd.DataFrame, y=None):
        # Stateless: no statistic is learned from the data, so there is
        # no train/test leakage risk from this particular transformer.
        # It still implements fit to stay compatible with Pipeline.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, codes in self.sentinel_map.items():
            if column in X.columns:
                X[column] = X[column].replace(codes, np.nan)
        return X


class RareCategoryGrouper(BaseEstimator, TransformerMixin):
    """Collapse infrequent levels of a nominal categorical column into "Other".

    Why this exists: DHS nominal fields such as ethnicity, religion, or
    detailed region can carry many levels, some observed only a handful
    of times. One hot encoding such a column directly would create many
    near constant columns that add noise and dimensionality without
    adding predictive signal, and a level seen only once or twice in
    training may not appear at all in a validation fold, causing an
    unseen category error at inference time. Grouping must be fit on
    training data only, since which categories count as rare is itself
    a statistic learned from the training distribution.

    Parameters
    ----------
    min_frequency : float
        Minimum fraction of training rows a category must appear in to
        be kept as its own level. Categories below this threshold are
        grouped into a single "Other" level. A starting value of 0.01
        (roughly one percent of training rows) is a reasonable default
        for survey data, but should be reported and justified in the
        Phase 1 report against the actual category frequency table for
        each nominal column, not left at this default silently.
    """

    def __init__(self, min_frequency: float = 0.01):
        self.min_frequency = min_frequency
        self.keep_categories_: dict[str, set] = {}

    def fit(self, X: pd.DataFrame, y=None):
        self.keep_categories_ = {}
        n = len(X)
        for column in X.columns:
            counts = X[column].value_counts(dropna=True)
            keep = counts[counts / n >= self.min_frequency].index
            self.keep_categories_[column] = set(keep)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, keep in self.keep_categories_.items():
            if column in X.columns:
                X[column] = X[column].where(X[column].isin(keep), other="Other")
        return X


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    """Cap continuous values outside Q1 minus 1.5 IQR to Q3 plus 1.5 IQR.

    Why this exists: the assignment specifies the 1.5 times IQR rule for
    skewed, non normally distributed continuous features, and requires
    capping rather than row deletion to preserve sample size and avoid
    discarding valid information in a row's other columns. The bounds
    (Q1, Q3, IQR) are statistics, so they must be learned from training
    data only and then applied unchanged to validation and test data,
    exactly like a scaler. Applying this to the full dataset before
    splitting would leak test set distribution information into the
    bounds used to clean the training set.
    """

    def __init__(self, columns: list[str] | None = None, factor: float = 1.5):
        self.columns = columns
        self.factor = factor
        self.bounds_: dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame, y=None):
        columns = self.columns or X.columns.tolist()
        self.bounds_ = {}
        for column in columns:
            q1 = X[column].quantile(0.25)
            q3 = X[column].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.factor * iqr
            upper = q3 + self.factor * iqr
            self.bounds_[column] = (lower, upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            if column in X.columns:
                X[column] = X[column].clip(lower=lower, upper=upper)
        return X


class ZScoreWinsorizer(BaseEstimator, TransformerMixin):
    """Cap continuous values beyond plus or minus z standard deviations.

    Why this exists: the assignment specifies the Z score rule for
    approximately normally distributed continuous features, as an
    alternative to the IQR rule for skewed features. Mean and standard
    deviation are learned on training data only, for the same leakage
    reasons as IQRWinsorizer.
    """

    def __init__(self, columns: list[str] | None = None, z_threshold: float = 3.0):
        self.columns = columns
        self.z_threshold = z_threshold
        self.bounds_: dict[str, tuple] = {}

    def fit(self, X: pd.DataFrame, y=None):
        columns = self.columns or X.columns.tolist()
        self.bounds_ = {}
        for column in columns:
            mu = X[column].mean()
            sigma = X[column].std()
            lower = mu - self.z_threshold * sigma
            upper = mu + self.z_threshold * sigma
            self.bounds_[column] = (lower, upper)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for column, (lower, upper) in self.bounds_.items():
            if column in X.columns:
                X[column] = X[column].clip(lower=lower, upper=upper)
        return X


class CollinearityScreen(BaseEstimator, TransformerMixin):
    """Drop one feature from each collinear pair, then iteratively by VIF.

    Why this exists: the Phase 1 protocol fixes the rules before any data
    is seen. Step 1 flags pairs with |Pearson r| >= corr_threshold and
    drops the member with the weaker absolute correlation to the target.
    Step 2 repeatedly drops the feature with the largest VIF while that
    VIF exceeds vif_threshold. Both steps are statistics, so they are
    learned in fit() on training data only, and transform() merely
    selects the surviving columns. The target must be passed to fit();
    it is used only to break ties between collinear features.

    Expects a numeric, already imputed DataFrame (no NaN).
    """

    def __init__(self, corr_threshold: float = 0.85, vif_threshold: float = 10.0):
        self.corr_threshold = corr_threshold
        self.vif_threshold = vif_threshold

    def fit(self, X: pd.DataFrame, y=None):
        from statsmodels.stats.outliers_influence import variance_inflation_factor

        if y is None:
            raise ValueError("CollinearityScreen.fit needs y for tie breaking.")
        X = pd.DataFrame(X).astype(float)
        y = pd.Series(np.asarray(y, dtype=float), index=X.index)
        target_r = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1])
                           if c.std() > 0 else 0.0).fillna(0.0)
        self.dropped_: list[tuple[str, str]] = []
        keep = list(X.columns)

        corr = X.corr().abs()
        for i, a in enumerate(corr.columns):
            for b in corr.columns[i + 1:]:
                if a in keep and b in keep and corr.loc[a, b] >= self.corr_threshold:
                    loser = a if target_r[a] < target_r[b] else b
                    keep.remove(loser)
                    self.dropped_.append((loser, f"|r|={corr.loc[a, b]:.2f} with partner"))

        while len(keep) > 1:
            vifs = pd.Series(
                [variance_inflation_factor(X[keep].values, i) for i in range(len(keep))],
                index=keep,
            )
            worst = vifs.idxmax()
            if not np.isfinite(vifs[worst]) or vifs[worst] > self.vif_threshold:
                keep.remove(worst)
                self.dropped_.append((worst, f"VIF={vifs[worst]:.1f}"))
            else:
                break
        self.keep_columns_ = keep
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(X)[self.keep_columns_]
