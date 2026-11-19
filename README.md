# Malaria Risk Prediction for Rwanda: Leakage-Free Preprocessing and Evaluation

A reproducible machine learning pipeline that predicts whether a person tested in the Rwanda
Malaria Indicator Survey (MIS) 2023 has a positive rapid diagnostic test (RDT), using household,
demographic and cluster-level climate covariates. The goal is to support targeted allocation of
bed nets and indoor residual spraying (SDG 3, target 3.3).

> **Headline result (repeated 5 x 5 cluster-grouped out-of-fold CV, all 179 positives):** regularized logistic
> regression on household features plus cluster-mean context reaches PR-AUC **0.080 +/- 0.032** (about 8x the 0.0096
> baseline), ROC-AUC **0.748 +/- 0.013**, lift 2.9 in the top 20%. Boosted trees score 0.025 to 0.031. Intervals are
> wide and PR-AUC depends strongly on which clusters share a fold; ROC-AUC is more stable. See `reports/repeated_evaluation.json`.

> **Honest summary.** Positives are rare (0.96%) and concentrated: 58 of 170 clusters contain any positive and one
> cluster holds 21.8% of them. The signal in these covariates is modest. The value of this repository is a
> defensible, leakage-free workflow and an honest measurement, not a high-accuracy classifier.

## 1. Problem and data

| Item | Value | Why |
|---|---|---|
| Survey | DHS Rwanda MIS 2023 (PR, HR, KR, GE, GC files) | The only Rwanda survey with GPS-linked climate covariates. MIS 2017 has no GPS and is a secondary comparison. |
| Unit | Tested household member (all ages, age is a feature) | Under-5 children alone give only 24 positives. |
| Target | `hml35` (RDT result), 1 = positive | Codes 3, 4, 6 (invalid/indeterminate) are set to missing and dropped, never imputed. |
| Size | N = 18,672 rows, P = 53 raw features (26 continuous, 22 nominal, 4 discrete, 1 ordinal) | `src/audit.py` -> `reports/audit_mis2023.json`. |
| Imbalance | 0.96% positive (93.9:1 in the training split) | Drives the metric, split and calibration choices. |
| Climate | DHS geospatial covariates (5 km rasters, 2020 values) joined by cluster | Main models use these. Interview-month CHIRPS v3 (5 km buffer), ERA5-Land dewpoint (only variable in the download) and MODIS EVI/NDVI features were tested (`src/climate_features.py`) and did not help. |

Raw DHS files are licensed and are **not** in the repository (`data/raw/` is git-ignored).
Request access at <https://dhsprogram.com>, then place the MIS 2023 files in `data/raw/mis2023/` (and MIS 2017 in `data/raw/mis2017/`).

## 2. Design decisions and justifications

**Leakage control.** Excluded: the second test (`hml32`), test-process variables, modelled malaria columns, anemia and
fever-treatment variables, identifiers, and the survey weight `hv005` as a feature (used only to weight evaluation).
Every statistic (medians, means, scalers, capping bounds, collinearity screen, domain-shift weights) is learned on training data only.

**Sentinel codes.** Real DHS codes for `hml35` are 3, 4 and 6 (not 9); they become NaN before any imputation.

**Structural missingness.** `hml4` and `hml10` are 44.05% missing in training and are missing exactly when the household has
no net (`hml12` = 0; Cramer's V = 1.0). Little's MCAR test rejects MCAR (chi-square 5997, df 29, p < 0.001).
*Decision:* an explicit "no net" level plus missing-indicator flags, not imputation.

**Splits.**
* *Hold-out:* 70/30 cluster-grouped split stratified by cluster burden tier (0, 1-2, 3+ positives), **fixed seed 42, no seed search**
  (`split_burden_tier` in `src/pipeline_builder.py`). Train 13,100 rows / 138 positives, test 5,572 rows / 41 positives, cluster overlap 0.
  The three largest clusters (39, 14, 11 positives) all fall in training, so test has 41 rather than 50-55 positives.
* *Headline evaluation:* repeated 5 x 5 `StratifiedGroupKFold` (seeds 100-104, fixed in advance) over all rows, so no favourable seed
  decides the result (`src/repeated.py`).

**Skew-aware numeric treatment (thresholds learned on training data only).**

| Training skewness | Outliers | Imputation | Transform |
|---|---|---|---|
| Right-skewed (> 0.5): 16 columns | IQR winsorize (1.5 x IQR; 1st/99th percentile if IQR = 0) | median | log1p, then standardize |
| Other skewed: 5 columns | IQR winsorize | median | standardize |
| Approximately Gaussian: 9 columns | Z-score winsorize (abs z > 3) | mean | standardize |

`Frost_Days_2020` has Q1 = Q3 = 0, so the plain rule would flag 18.5% of rows and flatten the feature; the winsorizer falls back to
percentile bounds. Rows are capped, never deleted. Example: `Livestock_Cattle` skew 3.63 becomes 0.13 after log1p (Figure 4.1).

**Encoding.** Explicit wealth order, explicit "missing" level for nominals, levels under 1% grouped into "Other", one-hot with
`drop="if_binary"`. 53 raw features become 96 model columns.

**Collinearity screen (logistic branch only).** 19 climate pairs have |r| >= 0.85 and 14 of 30 numeric features have VIF > 10.
*Supervisor feedback applied:* inside each collinear pair, and when VIF exceeds 10, the feature with the **lower mutual information with the
target** is dropped (MI averaged over 10 seeds because a single kNN estimate has a seed spread of about 0.0006, as large as the gaps between
features). This replaces the earlier |r|-to-target tie-break, which dropped a high-MI feature. 10 features are removed, max VIF falls to 8.47,
and the two highest-MI features (mean temperature, precipitation) are retained. Tree models are not screened.

**Models.** Regularized logistic regression (interpretable, best ranking), random forest, LightGBM and XGBoost. Class weights handle imbalance.

**Calibration and evaluation.** PR-AUC (primary), ROC-AUC, Brier, recall, precision, lift; never accuracy. Platt/isotonic calibration is fit on
out-of-fold predictions. Results are reported weighted (`hv005 / 1e6`) and unweighted, with cluster-bootstrap 95% intervals.

## 3. Results

### 3a. Headline: repeated 5 x 5 cluster-grouped out-of-fold CV (`reports/repeated_evaluation.json`)

| Config | Model | PR-AUC (mean +/- sd) | ROC-AUC | Columns | EPV |
|---|---|---|---|---|---|
| A. All 53 features | Logistic | 0.028 +/- 0.007 | 0.688 | 86 | 1.7 |
| B. Household (30) | Logistic | 0.070 +/- 0.028 | 0.747 | 73 | 1.9 |
| **C. Household + cluster means** | **Logistic** | **0.080 +/- 0.032** | **0.748** | 78 | 1.8 |
| C | XGBoost | 0.025 +/- 0.001 | 0.702 | 78 | 1.8 |
| D. B + MIS 2017, unweighted | Logistic | 0.023 +/- 0.001 | 0.678 | 93 | 1.5 |
| E. D + survey-round indicator | Logistic | 0.023 +/- 0.002 | 0.679 | 94 | 1.5 |
| F. E + covariate-shift weights | Logistic | 0.022 +/- 0.002 | 0.672 | 94 | 1.5 |
| G. Compact (<= 35 columns) + cluster means | Logistic | 0.024 +/- 0.004 | 0.703 | 33 | 4.3 |
| H. Compact, no context | Logistic | 0.027 +/- 0.005 | 0.713 | 28 | 5.1 |

Baseline PR-AUC 0.0096. EPV = training positives (about 143 per fold) / model columns.

### 3b. Hold-out (70/30 burden-tier split, 41 positives, Platt-calibrated, unweighted)

| Model | PR-AUC | ROC-AUC | Brier | Recall |
|---|---|---|---|---|
| Baseline (test prevalence) | 0.0074 | 0.50 | | |
| Logistic regression | 0.0203 | 0.699 | 0.0073 | 0.88 |
| Random forest | 0.0326 | 0.733 | 0.0073 | 0.73 |
| LightGBM | 0.0237 | 0.760 | 0.0073 | 0.93 |
| XGBoost | 0.0255 | 0.745 | 0.0073 | 0.88 |

With 41 positives the bootstrap intervals overlap and the ordering differs from 3a, so models are not ranked on this split.

## 4. Supervisor feedback and what was done

| Comment | Action | Outcome |
|---|---|---|
| Incomplete reanalysis climate (ERA5 has dewpoint only; MODIS 10.6% missing; 3 interview months) | Documented; flagged the full ERA5-Land download (2 m temperature, total precipitation) as the priority acquisition | **Not yet downloaded**; main models keep DHS 5 km covariates |
| Collinearity screen dropped the highest-MI feature | MI tie-break (10-seed average) in both screen steps | Top-2 MI features retained; near ties are within estimator noise |
| Survey-round shift when pooling MIS 2017 | Round indicator + covariate-shift weights (`src/repeated.py`) | Still no gain: domain AUC = 1.00 (rounds perfectly separable; `hv040` altitude missing for all 2017 rows and `hv226` cooking-fuel coded differently). Harmonize codes before DHS 2019-20 |
| 20% hold-out (36 positives) underpowered, seed picked by label balance | Fixed-seed 70/30 burden-tier split; repeated 5 x 5 grouped CV as headline | 41 test positives; headline PR-AUC 0.080 +/- 0.032 |
| Low EPV (95 columns / 143 events) | Compact <= 35-column sets; EPV reported | EPV 4.3-5.1 but ranking falls (ROC 0.70-0.71 vs 0.75); best model keeps the larger set and states its low EPV. More events (RBC district series) is the real fix |

## 5. Stage 2: tuning and robustness (`src/tuning.py`)

Nested, repeated, cluster-grouped tuning of L2 and elastic-net logistic regression, shallow XGBoost, a rank-average ensemble and an
extended cluster-context set (two split seeds): **no reliable gain** over the fixed logistic model (household + cluster means, C = 0.1).
ROC-AUC is flat (0.73-0.75) for C from 0.003 to 1 and PR-AUC differences are within split noise. Outputs: `reports/tuning_*.json`.

## 6. Interview-month climate features (`src/climate_features.py`, `src/pooled_dynamic.py`)

CHIRPS v3 rainfall, ERA5-Land dewpoint and MODIS EVI/NDVI joined by cluster and interview month (lags 0-3). On one 5-fold partition they lower
PR-AUC (0.011-0.021 vs 0.050 for household + cluster means on the same partition). Not rerun under repeated CV; needs the raster files in `data/external/`.

## 7. Diagnostic figures (`figures/`, 300 DPI, training data only)

| File | Shows |
|---|---|
| `fig1_1_missingness_heatmap.png` | Missingness pattern and top missing columns |
| `fig1_2_target_distribution.png` | Class imbalance and age by result |
| `fig2_1_outlier_boxplots.png` | Raw vs winsorized distributions |
| `fig3_1_correlation_matrix.png` | Upper-triangle correlation, |r| >= 0.85 boxed |
| `fig3_2_collinear_scatterplots.png` | Collinear pairs with fit lines |
| `fig3_3_target_importance.png` | Mutual information (10-seed average) and |r| with the target |
| `fig4_1_transformation_grid.png` | Raw, log, min-max, standardized |
| `fig5_1_pca_scree_plot.png` | Scree and cumulative variance (12 components reach 90%, 15 reach 95%) |

## 8. Why the scores are low

58 of 170 clusters contain a positive, one cluster holds 39 of 179 positives (21.8%), the top three hold 35.8% (`src/cluster_analysis.py`).
A cluster-level model reaches only ROC-AUC 0.52-0.58. Training on MIS 2017 and testing on MIS 2023 gives PR-AUC 0.015-0.021 (`src/transfer.py`).

## 9. Data to acquire next

Full ERA5-Land variable set (2 m temperature, total precipitation) (priority 1); harmonized DHS 2019-20 with round indicator and shift weights;
RBC monthly district case series 2014-2024 (more events, longitudinal design); NISR census denominators and district boundaries. Details in the report.

## 10. Repository layout

```
src/transformers.py      custom transformers (sentinel recoder, winsorizers, rare grouper, MI-aware collinearity screen)
src/data.py              MIS 2023 loader with leakage exclusions and cluster climate join
src/audit.py             data audit -> reports/audit_mis2023.json
src/pipeline_builder.py  burden-tier split and leakage-free preprocessing/model pipelines
src/diagnostics.py       statistics and the eight figures
src/evaluation.py        OOF predictions, calibration, weighted metrics, bootstrap CIs
src/holdout.py           70/30 hold-out stage: diagnostics, figures, models
src/repeated.py          headline repeated 5x5 grouped CV, round indicator, shift weights, EPV, domain separability
src/pooled.py            earlier single-partition pooled evaluation (kept for provenance)
src/tuning.py            stage 2 nested tuning, ensembling, penalty sweep
src/climate_features.py  interview-month CHIRPS / ERA5 / MODIS features; src/pooled_dynamic.py evaluates them
src/transfer.py          train 2017, test 2023
src/cluster_analysis.py  why skill is low (cluster concentration)
notebooks/pipeline_execution.ipynb   end-to-end run
reports/  figures/  docs/  outputs, figures, report (docs/report) and original proposal (docs/proposal)
```

## 11. Reproduce

```
pip install -r requirements.txt
# place DHS files in data/raw/mis2023/ and data/raw/mis2017/
python -m src.audit
python -m src.holdout              # split, figures, hold-out models
python -m src.repeated             # headline repeated CV (long; supports config subsets)
jupyter nbconvert --to notebook --execute --inplace notebooks/pipeline_execution.ipynb
```

## 12. Limitations

* Only 179 positives; one cluster holds 21.8% of them. PR-AUC is dominated by a few clusters: read intervals, not point values.
* EPV is 1.5-1.9 for the full feature sets and 4.3-5.1 for compact sets, below the usual benchmark of 10.
* DHS GPS is displaced (urban up to 2 km, rural up to 5 km), so cluster climate carries location error.
* Main-model climate values are single-year (2020) rasters; ERA5 temperature and precipitation are still missing.
* Calibration and thresholds describe this survey, not other years or countries. Weights are used in evaluation only.

## 13. Data and license

DHS microdata cannot be redistributed. Code is provided for research and public-health use.
