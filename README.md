# Climate and Demographic Informed Malaria Risk Prediction in Rwanda

Course: 04-652 Artificial Intelligence System Design
Assignment: Assignment 2, Data Preprocessing (Phase 1 and Phase 2)

## Project Status

This repository is under active development. The table below is kept
current as work proceeds; do not trust numbers elsewhere in this README
over this table if they ever disagree.

| Item | Status |
|---|---|
| GitHub repository structure | Created |
| DHS Program data request submitted | Pending |
| Raw data file obtained | Not yet |
| Phase 1 Strategy Report | Written (PDF in phase1_strategy/); empirical values marked pending until data arrives |
| Audit and collinearity code | `src/audit.py`, `CollinearityScreen` written; tested on synthetic data only |
| Phase 2 pipeline and notebook | Not started, blocked on data |

## 1. Problem Statement

Malaria remains a leading cause of morbidity in Rwanda despite
substantial reductions achieved through vector control and treatment
access programs over the past decade. This project predicts whether an
individual child tested for malaria returns a positive result, using
household, demographic, and residence level covariates collected in a
nationally representative health survey. A model that reliably flags
which household and child level profiles carry elevated malaria risk
can support targeted distribution of bed nets and indoor spraying
resources toward the households most likely to need them, rather than
uniform allocation across an entire district or province.

This task directly supports Sustainable Development Goal 3, Good
Health and Well Being, target 3.3, which calls for ending the
epidemics of malaria by 2030.

Note on scope relative to the original project proposal: the proposal
described a district month level early warning system using climate
time series and aggregated case counts. Obtaining that granular,
un-aggregated monthly district level series requires a formal data
clearance request to the Rwanda Biomedical Centre, whose timeline
exceeds this assignment's deadline. For Assignment 2 specifically, the
operational dataset is therefore the Rwanda Demographic and Health
Survey Program's individual level malaria testing data, which is
obtainable within the assignment's timeframe, keeps the same country,
disease, and SDG framing, and presents the same categories of
preprocessing challenge (mixed variable types, nontrivial missingness,
class imbalance, correlated predictors) that the course rubric
evaluates. This deviation and its justification are documented in full
in the Phase 1 Strategy Report.

## 2. Dataset Overview

- **Source**: The DHS Program, Rwanda [Standard DHS or Malaria
  Indicator Survey, to be finalized once the data request is
  approved]. Access portal: https://dhsprogram.com/data/dataset_admin/login_main.cfm
- **File used**: Person Recode (PR), which contains the malaria rapid
  diagnostic test result (`hml35`) and microscopy result (`hml32`) for
  eligible household members, per DHS's Guide to DHS Statistics
  documentation of the malaria prevalence indicator.
- **Sample size (N)**: Pending. Will be reported as the exact row
  count of the PR file after filtering to the malaria testing eligible
  population (children 6 to 59 months, de facto household members),
  not the full household roster row count.
- **Feature count (P)**: Pending. Will be reported as the number of
  columns retained after an initial audit removes identifier columns,
  columns unrelated to this prediction task, and columns that would
  constitute target leakage (for example any derived malaria morbidity
  or treatment variable computed from the test result itself).
- **Target variable (Y)**: Malaria rapid diagnostic test result
  (`hml35`), recoded to a binary indicator: 1 if positive, 0 if
  negative. Rows with a DHS sentinel missing code for this field are
  excluded from supervised training and reported separately as a
  missingness statistic, not imputed, since a test result cannot be
  meaningfully imputed from other covariates without strong assumptions.

## 3. Task Type

**Binary Classification.**

This is a scope correction from the original project proposal's three
level ordinal classification (normal, elevated, epidemic incidence),
which was defined for district month aggregated case counts. At the
individual test result level, the only outcome DHS records is a
binary positive or negative result, so the task type for this
assignment is binary classification. See the Phase 1 Strategy Report
for the full reasoning behind this change.

## 4. Repository Structure

```
.
├── .gitignore
├── requirements.txt
├── README.md
├── data/
│   ├── raw/            # Original DHS export, never committed (see .gitignore)
│   └── processed/      # Cleaned, encoded, scaled feature table
├── phase1_strategy/
│   └── Phase1_Strategy_Report.pdf
├── phase2_implementation/
│   ├── notebooks/
│   │   └── pipeline_execution.ipynb
│   └── Phase2_Final_Report.pdf
├── src/
│   ├── __init__.py
│   ├── transformers.py      # Custom Scikit Learn transformers
│   └── pipeline_builder.py  # ColumnTransformer and Pipeline composition
└── figures/              # 8 required diagnostic figures, PNG, 300 DPI
```

## 5. Reproduction Steps

1. Create and activate a virtual environment, then install
   dependencies:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Obtain the Rwanda DHS PR (Person Recode) file from
   https://dhsprogram.com/data (requires a free account and an
   approved data request; approval is not guaranteed to be instant).
   Place the downloaded file in `data/raw/`. This file is excluded
   from version control by `.gitignore`, in compliance with the DHS
   Program's data use agreement, which does not permit redistribution
   of microdata.
3. Open `phase2_implementation/notebooks/pipeline_execution.ipynb` and
   run all cells in order. The notebook reads from `data/raw/`, writes
   the final processed feature table to `data/processed/`, and saves
   all 8 diagnostic figures to `figures/`.
4. The preprocessing logic itself lives in `src/transformers.py` and
   `src/pipeline_builder.py`, imported by the notebook rather than
   duplicated inside it, so the same pipeline code can be reused
   outside the notebook (for example in a future scoring script).

## 6. Data Use and Privacy Note

DHS Program microdata is de identified individual level survey data,
released under a data use agreement that permits use for approved
research and coursework but prohibits redistribution of the raw
microdata itself. This repository therefore versions the processing
code and the documentation of what the data contains, but never the
raw or intermediate data files themselves. See `.gitignore`.
