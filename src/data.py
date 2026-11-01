"""Load and assemble the Rwanda MIS modelling table.

One row per household member with a valid malaria rapid diagnostic test. Person level columns come
from the Household Member Recode (PR). Cluster level environmental covariates come from the DHS
Geospatial Covariates file (GC), joined on the survey cluster id. The DHS covariates are 5 km raster
values sampled at the (displaced) cluster location, so no exact coordinates are ever used.

Only information available before or independent of the test is admitted as a feature. Anything
that is a second test result, a post diagnosis variable, or a modelled malaria estimate that could
encode the outcome is excluded and listed in EXCLUDED_WITH_REASON.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.transformers import (
    CLUSTER_ID,
    DHS_SENTINEL_CODES,
    SAMPLE_WEIGHT,
    TARGET_COLUMN_RDT,
    DHSSentinelRecoder,
)

# Person and household features from the PR file (names verified to exist in MIS 2023).
CONTINUOUS_PERSON = ["hv105", "hv040", "hv271"]            # age, cluster altitude, wealth score
DISCRETE_PERSON = ["hv009", "hv216", "hml1", "hml4"]       # household size, sleeping rooms, nets, net age (months)
ORDINAL_PERSON = ["hv270"]                                 # wealth quintile, 1 poorest to 5 richest
NOMINAL_PERSON = [
    "hv104", "hv024", "hv025", "hv201", "hv205", "hv213", "hv214", "hv215", "hv226",
    "hv206", "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv221", "hv247",
    "hv227", "hml10", "hml12", "hv219",
]
WEALTH_QUINTILE_ORDER = [1, 2, 3, 4, 5]                    # poorest ... richest -> 0..4

# Cluster covariates kept from the DHS GC file. Latest extracted year (2020) is used because the
# survey year (2023) is not available for these rasters.
GC_YEARLY = [
    "Rainfall", "Precipitation", "Wet_Days", "Mean_Temperature", "Maximum_Temperature",
    "Minimum_Temperature", "Land_Surface_Temperature", "Day_Land_Surface_Temp",
    "Night_Land_Surface_Temp", "Diurnal_Temperature_Range", "Enhanced_Vegetation_Index",
    "Aridity", "PET", "Frost_Days", "UN_Population_Density",
]
GC_STATIC = [
    "Elevation", "Travel_Times", "Nightlights_Composite", "Global_Human_Footprint",
    "Growing_Season_Length", "Irrigation", "Livestock_Cattle", "Livestock_Goats",
]
GC_YEAR = 2020

EXCLUDED_WITH_REASON = {
    "hml32": "microscopy result: a second test of the same outcome (target leakage)",
    "hml35": "target",
    "hml30-hml34, hml36-hml38": "test process and result variables",
    "Malaria_Incidence_*, Malaria_Prevalence_*": "modelled malaria estimates derived from survey data (circular)",
    "ITN_Coverage_*": "modelled intervention estimate that also encodes malaria programme data",
    "hv001, hv002, hv003, hv021, DHSID": "identifiers, used only for grouping",
    "hv005": "design weight, used for evaluation not as a feature",
    "fever treatment, hemoglobin, anemia": "post diagnosis or same visit biomarkers; not in the feature set",
}


def person_feature_columns() -> list[str]:
    return CONTINUOUS_PERSON + DISCRETE_PERSON + ORDINAL_PERSON + NOMINAL_PERSON


def climate_feature_columns() -> list[str]:
    return [f"{c}_{GC_YEAR}" for c in GC_YEARLY] + GC_STATIC


def load_mis2023(raw_dir: str | Path = "data/raw/mis2023") -> dict:
    """Return the assembled table and metadata needed by the pipeline.

    Returns a dict with: ``raw_pr`` (full member file, untouched), ``data`` (valid tests only, with
    features, target ``y``, ``groups`` and ``weight``), ``feature_columns`` and ``dropped_rows`` counts.
    """
    raw_dir = Path(raw_dir)
    pr = pd.read_stata(raw_dir / "RWPR8AFL.DTA", convert_categoricals=False)
    gc = pd.read_csv(raw_dir / "RWGC8AFL.csv")

    pr_clean = DHSSentinelRecoder().transform(pr)
    n_all = len(pr_clean)
    valid = pr_clean[pr_clean[TARGET_COLUMN_RDT].isin([0, 1])].copy()
    dropped = {"rows_total": n_all, "rows_without_valid_test": int(n_all - len(valid))}

    keep_person = [c for c in person_feature_columns() if c in valid.columns]
    missing_person = [c for c in person_feature_columns() if c not in valid.columns]
    gc_cols = [c for c in climate_feature_columns() if c in gc.columns]
    gc_small = gc[["DHSCLUST"] + gc_cols].rename(columns={"DHSCLUST": CLUSTER_ID})

    data = valid[[CLUSTER_ID, SAMPLE_WEIGHT, TARGET_COLUMN_RDT, "hv006", "hv007"] + keep_person].merge(
        gc_small, on=CLUSTER_ID, how="left", validate="m:1"
    )
    data = data.rename(columns={TARGET_COLUMN_RDT: "y", CLUSTER_ID: "cluster", "hv006": "int_month", "hv007": "int_year"})
    data["weight"] = data[SAMPLE_WEIGHT] / 1e6
    data["y"] = data["y"].astype(int)
    feature_columns = keep_person + gc_cols
    return {
        "raw_pr": pr,
        "data": data.drop(columns=[SAMPLE_WEIGHT]),
        "feature_columns": feature_columns,
        "missing_person_columns": missing_person,
        "dropped_rows": dropped,
    }
