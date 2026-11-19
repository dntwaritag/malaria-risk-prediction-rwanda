"""Hold-out stage on the pre-specified 70/30 burden-tier cluster split: diagnostics, figures, models.

The pooled repeated CV (src/repeated.py) is the headline result; this hold-out is reported for completeness.
"""
import json
import warnings

from src import data as D
from src.diagnostics import run_diagnostics, collinearity_report
from src.evaluation import run_evaluation
from src.pipeline_builder import split_burden_tier, make_preprocessor_for


def run():
    warnings.filterwarnings("ignore")
    r = D.load_mis2023(); d = r["data"]; F = r["feature_columns"]
    train, test = split_burden_tier(d)
    info = {"train_rows": int(len(train)), "train_pos": int(train.y.sum()), "test_rows": int(len(test)),
            "test_pos": int(test.y.sum()), "train_clusters": int(train.cluster.nunique()),
            "test_clusters": int(test.cluster.nunique()), "overlap": len(set(train.cluster) & set(test.cluster)),
            "seed": int(train.attrs["split_seed"]), "train_prev": float(train.y.mean()), "test_prev": float(test.y.mean())}
    prep, groups, num = make_preprocessor_for(train[F], F)
    Ztr = prep.fit_transform(train[F]); Zte = prep.transform(test[F])
    info.update({"initial_N": int(len(d)), "initial_P": len(F), "train_final": list(Ztr.shape), "test_final": list(Zte.shape),
                 "nan_train": int(Ztr.isna().sum().sum()), "nan_test": int(Zte.isna().sum().sum())})
    json.dump(info, open("reports/split_info.json", "w"), indent=2)
    print(info, flush=True)
    stats = run_diagnostics(train, F, figures_dir="figures", stats_path="reports/diagnostic_stats.json", test=test)
    print("diagnostics done", flush=True)
    collinearity_report(train, F)
    run_evaluation(train, test, F, n_boot=200)
    print("evaluation done", flush=True)


if __name__ == "__main__":
    run()
