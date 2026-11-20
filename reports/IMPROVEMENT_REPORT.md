# Improvement experiments: combining all downloaded data (20 Nov 2026)

**Question.** Can the model be improved by combining every data source now in `data/` (CHIRPS v3 rainfall, ERA5-Land, MODIS EVI/NDVI, DHS geospatial covariates, interview-month temperature, spatial-neighbour burden)?

**Protocol.** Same as the headline evaluation: logistic regression, 5 repeats x 5-fold StratifiedGroupKFold by cluster (seeds 100-104), leakage-free pipeline. Folds are identical across configurations, so differences are paired. The base row reproduces the headline PR-AUC 0.0799 exactly. Code: `src/improve.py`, `src/improve_summary.py`.

## Results

| Configuration | PR-AUC (mean ± sd) | ROC-AUC | Top-20% lift | Columns | Diff vs current best | Repeats better |
|---|---|---|---|---|---|---|
| + case rate of 5 nearest training clusters (computed inside each fold) | 0.0852 ± 0.0377 | 0.742 | 2.77 | 79 | +0.0054 | 3/5 |
| + same, 10 nearest clusters | 0.0837 ± 0.0327 | 0.742 | 2.78 | 79 | +0.0039 | 4/5 |
| Same, weaker L2 penalty (C=0.3; tuned on the same folds) | 0.0826 ± 0.0323 | 0.744 | 2.84 | 78 | +0.0028 | 5/5 |
| Current best: household features + cluster means | 0.0799 ± 0.0316 | 0.748 | 2.87 | 78 | +0.0000 | 0/5 |
| + spatial k5, C=0.03 | 0.0783 ± 0.0330 | 0.748 | 2.81 | 79 | -0.0015 | 1/5 |
| Same, stronger penalty (C=0.03) | 0.0729 ± 0.0261 | 0.754 | 2.92 | 78 | -0.0070 | 0/5 |
| + spatial k5 + interview-month temperature | 0.0728 ± 0.0324 | 0.729 | 2.76 | 79 | -0.0070 | 0/5 |
| + DHS temperature for the interview month | 0.0658 ± 0.0250 | 0.735 | 2.80 | 78 | -0.0141 | 0/5 |
| Same, C=0.01 | 0.0646 ± 0.0201 | 0.758 | 2.89 | 78 | -0.0153 | 0/5 |
| + spatial k5 + 5 dynamic features | 0.0626 ± 0.0289 | 0.733 | 2.61 | 86 | -0.0173 | 0/5 |
| + 5 interview-month CHIRPS/ERA5/MODIS features | 0.0570 ± 0.0220 | 0.738 | 2.71 | 85 | -0.0229 | 0/5 |
| + all 17 dynamic features (lags 0-3) | 0.0304 ± 0.0078 | 0.732 | 2.61 | 95 | -0.0494 | 0/5 |
| + all DHS geospatial covariates (2020) | 0.0262 ± 0.0056 | 0.676 | 2.42 | 91 | -0.0537 | 0/5 |
| Everything combined (all covariates, all dynamic, temperature, spatial) | 0.0178 ± 0.0043 | 0.634 | 1.90 | 108 | -0.0621 | 0/5 |

Chance level (prevalence) is 0.0096. Standard deviation across repeats is about 0.03 for the best models.

## What this shows

1. **No data source gave a reliable improvement.** The best variants (spatial-neighbour k=5 / k=10, weaker penalty C=0.3) differ from the current best by +0.003 to +0.005 PR-AUC, which is far below the 0.03 repeat-to-repeat spread, and the spatial variants beat the base in only 3 and 4 of 5 repeats. Treat these as ties.
2. **More features made it worse.** Interview-month climate and vegetation features lowered PR-AUC from 0.080 to 0.057-0.066; all 17 dynamic features gave 0.030; all DHS covariates gave 0.026; everything together gave 0.018. With 179 positives (EPV about 1.8) every extra column costs more in variance than it adds in signal.
3. **Why climate adds little here.** Malaria in this survey is rare and concentrated in a few clusters (top 3 hold 35.8% of cases), and the survey covers only 3 interview months (Jul-Dec 2023). Rainfall and vegetation barely vary between clusters within those months, so they cannot separate high-burden clusters.
4. **Regularization.** Stronger penalties (C=0.03, 0.01) lowered PR-AUC but raised ROC-AUC slightly (0.754-0.758). C=0.3 was marginally better; because C was compared on the same folds, this small gain is optimistic.

## Cluster-level view (one repeat, base model)

Averaging scores within each of the 170 clusters, the base model puts **58.7% of all cases in its top 20% of clusters** (34 clusters, 18 of which have cases); cluster-level AUC for "has any case" is 0.65. Adding spatial or climate features did not improve this (52-56%). This is the most decision-relevant number for district targeting and should be presented with the same caveats.

## What is still missing from `data/`

* ERA5-Land files still contain **dewpoint only** (`d2m`); 2 m temperature and total precipitation were not downloaded.
* `data/external/dhs_extra/` is empty: **no DHS 2019-20** data yet. MIS 2017 pooling already hurt (PR-AUC 0.022).
* **No RBC monthly district case series.** This is the data most likely to add information: it adds events and temporal variation.

## Recommended next steps

* Keep the current best configuration (household + cluster means, logistic) as the headline model. Report the spatial-neighbour variant as an exploratory tie, not an improvement.
* Add the RBC district case series and a second DHS survey (harmonized codes) before trying more features.
* Report cluster-level capture alongside PR-AUC.
