"""Merge improvement experiment outputs and compare each configuration with the base model, repeat by repeat (paired)."""
import glob, json
import numpy as np
res = {}
for f in sorted(glob.glob("reports/improve_*.json")):
    for k, v in json.load(open(f))["configs"].items():
        res.setdefault(k, {}).update(v)
base = np.array(res["base_household_cm"]["logistic"]["pr_auc"]["per_repeat"])
rows = []
for k, v in res.items():
    x = v["logistic"]; p = np.array(x["pr_auc"]["per_repeat"]); dlt = p - base
    rows.append({"config": k, "pr_auc_mean": x["pr_auc"]["mean"], "pr_auc_sd": x["pr_auc"]["sd"], "roc_auc": x["roc_auc"]["mean"],
                 "lift_top20": x["lift_top20"]["mean"], "n_cols": x["n_cols"]["mean"],
                 "diff_vs_base": float(dlt.mean()), "repeats_better": int((dlt > 0).sum()), "per_repeat": [round(float(a), 4) for a in p]})
rows.sort(key=lambda r: -r["pr_auc_mean"])
json.dump(rows, open("reports/improvement_summary.json", "w"), indent=2)
print(f"{'config':28s} {'PR-AUC':>7s} {'sd':>6s} {'ROC':>6s} {'lift':>5s} {'cols':>5s} {'diff':>7s} better")
for r in rows:
    print(f"{r['config']:28s} {r['pr_auc_mean']:7.4f} {r['pr_auc_sd']:6.4f} {r['roc_auc']:6.3f} {r['lift_top20']:5.2f} {r['n_cols']:5.1f} {r['diff_vs_base']:+7.4f} {r['repeats_better']}/5")
