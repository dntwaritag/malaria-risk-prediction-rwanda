,# Malaria Risk Prediction for Rwanda: Leakage-Free Preprocessing and Evaluation

A reproducible machine learning pipeline that predicts whether a person tested in the Rwanda
Malaria Indicator Survey (MIS) 2023 has a positive rapid diagnostic test (RDT), using household,
demographic and cluster-level climate covariates. The goal is to support targeted allocation of
bed nets and indoor residual spraying (SDG 3, target 3.3).

> **Honest summary.** Positives are rare (0.96%) and the signal in these covariates is weak.
> Held-out PR-AUC is 0.013 to 0.020 against a 0.0096 prevalence baseline, and the confidence
> intervals of the four models overlap. The value of this repository is a defensible,
> leakage-free workflow and an honest measurement, not a high-accuracy classifier.

## 1. Problem and data

| Item | Value | Why |
|---|---|---|
| Survey | DHS Rwanda MIS 2023 (PR, HR, KR, GE, GC files) | The only Rwanda survey with GPS-linked climate covariates. MIS 2017 has no GPS and is kept as a secondary comparison. |
| Unit | Tested household member (all ages, age is a feature) | Under-5 children alone give only 24 positives, too few to model. |
| Target | `hml35` (RDT result), 1 = positive | Binary, measured for every tested member. Codes 3, 4, 6 (invalid/indeterminate) are set to missing and dropped, never imputed. |
| Size | N = 18,672 rows, P = 53 raw features (26 continuous, 22 nominal, 4 discrete, 1 ordinal) | Produced by `src/audit.py`, stored in `reports/audit_mis2023.json`. |
| Imbalance | 0.96% positive (about 104:1 in the training split) | Drives the metric, split and calibration choices below. |
| Climate | DHS geospatial covariates (5 km rasters, 2020 values) joined by cluster | Pragmatic stand-in: CHIRPS, ERA5 and MODIS were not reachable from the build environment. This is a documented deviation. |

Raw DHS files are licensed and are **not** in the repository (`data/raw/` is git-ignored).
Request access at <https://dhsprogram.com>, then place the MIS 2023 files in `data/raw/mis2023/`.

## 2. Design decisions and justifications

**Leakage control.** Excluded by design: the second test (`hml32`), test-process variables,
modelled malaria columns (Malaria_Incidence_*, Malaria_Prevalence_*, ITN_Coverage_*), anemia and
fever-treatment variables, identifiers, and the survey weight `hv005` as a feature (it is used only
to weight evaluation). *Justification:* each of these is either the label in another form or is
known only after the test, so using them would inflate scores without real predictive value.

**Sentinel codes.** Real DHS codes for `hml35` are 3, 4 and 6, not 9 as assumed in the planning
phase. They are converted to NaN before any imputation so imputers never average code numbers
into real values.

**Structural missingness.** `hml4` and `hml10` are 46.1% missing and are missing exactly when the
household has no net (`hml12` = 0; Cramér's V = 1.0). Little's MCAR test also rejects MCAR
(chi-square 6984, df 29, p < 0.001). *Decision:* an explicit "no net" level plus missing-indicator
flags, not MCAR/MAR imputation, because the value is absent by design.

**Cluster-grouped, label-stratified split.** Households in one survey cluster share geography and
climate, so a random split lets the model memorize clusters. `StratifiedGroupKFold` and the
train/test split keep each cluster (`hv001`) entirely on one side (overlap checked = 0). The split
seed (51) was chosen by label balance only (36 test positives vs 20 for seed 42), never by model
performance.

**Skew-aware numeric treatment (thresholds learned on training data only).**

| Training skewness | Outliers | Imputation | Transform |
|---|---|---|---|
| Right-skewed (> 0.5): 13 columns | IQR winsorize (1.5 x IQR; 1st/99th percentile if IQR = 0) | median | log1p, then standardize |
| Other skewed: 5 columns | IQR winsorize | median | standardize |
| Approximately Gaussian: 12 columns | Z-score winsorize (|z| > 3) | mean | standardize |

`Frost_Days_2020` has Q1 = Q3 = 0, so the plain 1.5 IQR rule would have flagged 20% of rows and flattened the
feature to a constant; the winsorizer therefore falls back to percentile bounds when the IQR is zero.
Rows are capped, never deleted, because with 143 training positives every row matters.
Example: `Livestock_Goats` skew 4.12 becomes -0.19 after log1p (Figure 4.1). All imputers keep
missing indicators.

**Encoding.** Wealth quintile uses an explicit order [1..5] (no alphabetical default, which would
invent a false rank). Nominal columns get an explicit "missing" level, levels under 1% are grouped
into "Other", then one-hot with `drop="if_binary"`. 53 raw features become 95 model columns.

**Collinearity.** 20 climate feature pairs have |r| >= 0.85 (strongest: Mean_Temperature vs PET,
r = 0.993). The screen keeps the feature with the stronger link to the target, then drops by
VIF > 10. *Justification (supervisor feedback):* it is applied to the **logistic regression branch
only**, since tree models tolerate collinearity and lose nothing from keeping correlated inputs.

**Models.** Regularized logistic regression (interpretable baseline), Random Forest (robust
baseline), LightGBM and XGBoost (primary). Class weights handle imbalance without touching data.

**Calibration and evaluation.** Metrics: PR-AUC (primary), ROC-AUC, Brier, recall, precision, F1;
never accuracy. Out-of-fold predictions feed Platt scaling or isotonic regression, so calibration
never sees test data. The operating threshold is chosen on out-of-fold data (recall target 0.70).
Results are reported **weighted** (`hv005 / 1e6`) and **unweighted**, with cluster-bootstrap 95%
confidence intervals, because the test set has only 36 positives.

## 3. Results (held-out test, Platt-calibrated, unweighted)

| Model | PR-AUC | ROC-AUC | Brier | Recall | Precision |
|---|---|---|---|---|---|
| Baseline (prevalence) | 0.0096 | 0.50 | | | |
| Logistic regression | 0.0132 | 0.546 | 0.0096 | 0.528 | 0.011 |
| Random forest | 0.0138 | 0.625 | 0.0096 | 0.472 | 0.013 |
| LightGBM | 0.0156 | 0.656 | 0.0097 | 0.750 | 0.015 |
| XGBoost | 0.0203 | 0.643 | 0.0096 | 0.639 | 0.014 |

Out-of-fold (5-fold grouped) PR-AUC picks **XGBoost** (0.0455), ahead of LightGBM (0.0393), logistic
regression (0.0289) and random forest (0.0275); XGBoost also has the best test PR-AUC (0.0203, weighted 0.0193).
LightGBM has the best test ROC-AUC (0.656) and recall (0.75). Bootstrap intervals overlap, so the ranking among
gradient-boosting models is not statistically established; both clearly beat the linear and bagging baselines out-of-fold. Full numbers, including weighted
results and calibration bins, are in `reports/evaluation.json`; diagnostics are in
`reports/diagnostic_stats.json`.

## 4. Diagnostic figures (`figures/`, 300 DPI, training data only)

| File | Shows |
|---|---|
| `fig1_1_missingness_heatmap.png` | Missingness pattern and top missing columns |
| `fig1_2_target_distribution.png` | Class imbalance and age by result |
| `fig2_1_outlier_boxplots.png` | Raw vs winsorized distributions |
| `fig3_1_correlation_matrix.png` | Upper-triangle correlation, |r| >= 0.85 boxed |
| `fig3_2_collinear_scatterplots.png` | Collinear pairs with fit lines |
| `fig3_3_target_importance.png` | Mutual information and |r| with the target |
| `fig4_1_transformation_grid.png` | Raw, log, min-max, standardized |
| `fig5_1_pca_scree_plot.png` | Scree and cumulative variance (12 components reach 90%, 15 reach 95%) |

## 4b. Why the scores are low, and the 2017 to 2023 transfer test

Only 58 of 170 clusters contain a positive, one cluster holds 39 of the 179 positives (21.8%) and the top three hold 35.8%
(`src/cluster_analysis.py`). A cluster-level model reaches only ROC-AUC 0.52 to 0.58. Training on MIS 2017 (1,765 positives) and
testing on all of MIS 2023 (`src/transfer.py`) gives PR-AUC 0.015 to 0.021 and ROC-AUC 0.63 to 0.67 against a 0.0096 baseline,
a more stable estimate than the 36-positive hold-out. The full report is `docs/report/Assignment3_Phase2.pdf`.

## 5. Repository layout

```
src/transformers.py      custom transformers (sentinel recoder, winsorizers, rare grouper, collinearity screen)
src/data.py              MIS 2023 loader with leakage exclusions and cluster climate join
src/audit.py             data audit -> reports/audit_mis2023.json
src/pipeline_builder.py  grouped split and leakage-free preprocessing/model pipelines
src/diagnostics.py       statistics and the eight figures
src/evaluation.py        OOF predictions, calibration, weighted metrics, bootstrap CIs
notebooks/pipeline_execution.ipynb   end-to-end run
reports/  figures/  docs/            outputs, figures, planning documents
```

## 6. Reproduce

```
pip install -r requirements.txt
# place DHS files in data/raw/mis2023/
python -m src.audit
jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline_execution.ipynb
```

## 7. Limitations

* Only 179 positives in total (143 train, 36 test): estimates are noisy; read intervals, not point values.
* DHS GPS is displaced (urban up to 2 km, rural up to 5 km, 1% of rural up to 10 km), so cluster
  climate values carry location error. Using 5 km covariates partly absorbs this.
* Climate values are single-year (2020) rasters, not matched to the survey month.
* Calibration and thresholds describe this survey, not future years or other countries.
* Survey-weighted results are reported alongside unweighted ones, but weights are not used in training.

## 8. Data and license

DHS microdata cannot be redistributed. Code is provided for research and public-health use.
