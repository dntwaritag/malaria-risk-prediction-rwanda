# Malaria Risk Prediction in Rwanda: Climate and Survey Informed Preprocessing Pipeline

A leakage free preprocessing and modelling pipeline that estimates the probability that a
household member tests positive for malaria (rapid diagnostic test), using household,
demographic, residence and satellite climate covariates from the Rwanda 2023 Malaria
Indicator Survey. The goal is to support targeted allocation of bed nets and indoor
residual spraying (SDG 3, target 3.3).

## Status

| Item | Status |
|---|---|
| Data access (DHS Program, Rwanda MIS 2023 and 2017) | Approved, files downloaded locally |
| Data audit | Done, see `reports/audit_mis2023.json` |
| Preprocessing pipeline (`src/`) | In progress |
| Diagnostic figures (`figures/`) | In progress |
| Reflection report | In progress |

## Data

- **Source:** The DHS Program, Rwanda Malaria Indicator Survey 2023 (primary). The 2017 survey is
  kept as a secondary comparison. Access requires a free, approved DHS account:
  https://dhsprogram.com/data/dataset_admin/login_main.cfm
- **Files used (Stata):** Household Member Recode `RWPR8A`, Household Recode `RWHR8A`,
  Children's Recode `RWKR8A`, Geographic Data `RWGE8A` (cluster coordinates, displaced by DHS),
  Geospatial Covariates `RWGC8A` (cluster level climate covariates).
- **Unit of analysis:** one tested household member.
- **Target (Y):** `hml35`, malaria rapid diagnostic test. `1` positive, `0` negative. Codes
  `3` (not present), `4` (refused) and `6` (other) are treated as missing and excluded from supervised
  training. They are never imputed.
- **Measured size (2023, from the audit):** 19,038 rows by 206 columns in the member file. 18,672
  members have a valid test result, of which 179 are positive (0.96%, about 104 to 1), spread over
  170 survey clusters. Among children under 5 only 24 are positive, so all ages are used and age is a feature.
- **Task type:** Binary classification, heavily imbalanced.

## Method summary

- All cleaning, imputation, encoding and scaling run inside a scikit-learn `Pipeline` and
  `ColumnTransformer`. Statistics are learned on training folds only.
- Splitting is grouped by survey cluster (`hv001`) with stratification, so no cluster appears in both
  train and test.
- Survey weights (`hv005`) are used in evaluation, with weighted and unweighted results both reported.
- Probabilities are calibrated (Platt scaling or isotonic regression) on out of fold predictions.
- Collinearity screening (|r| >= 0.85, VIF > 10) is applied to linear models only.
- Climate covariates are cluster level values. Because DHS displaces cluster coordinates, buffer
  averages are used, not exact points.

## Repository layout

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/          # DHS files, never committed
│   └── processed/    # model ready tables, never committed
├── src/              # reusable transformers, pipeline, audit code
├── notebooks/        # executable analysis
├── figures/          # diagnostic plots, 300 DPI
├── reports/          # audit output and written reports
└── docs/             # strategy documents and notes
```

## Reproduce

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# place the DHS files under data/raw/mis2023/ (see "Data" above)
python -m src.audit data/raw/mis2023/RWPR8AFL.DTA --out reports/audit_mis2023.json
jupyter notebook notebooks/pipeline_execution.ipynb
```

## Data use

DHS microdata is released under a data use agreement that forbids redistribution. Raw and
processed data files are excluded by `.gitignore`. Only code and aggregate statistics are versioned.
