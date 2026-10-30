"""Diagnostic statistics and the eight publication quality figures.

Everything here is computed on the TRAINING split only, so the diagnostics cannot leak test
information into design decisions. Figures are saved at 300 DPI.
"""
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src import data as D
from src.pipeline_builder import make_preprocessor_for, numeric_columns
from src.transformers import IQRWinsorizer

DPI = 300
sns.set_theme(style="whitegrid", context="paper", font_scale=1.0)
PALETTE = {"neg": "#4C72B0", "pos": "#C44E52", "accent": "#DD8452", "grey": "#8C8C8C"}


def _save(fig, figures_dir: Path, name: str):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / name, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ----------------------------------------------------------------------------- statistics
def cramers_v(a: pd.Series, b: pd.Series) -> float:
    ct = pd.crosstab(a, b)
    chi2 = stats.chi2_contingency(ct, correction=False)[0]
    n = ct.values.sum()
    r, k = ct.shape
    return float(np.sqrt(chi2 / (n * (min(r, k) - 1)))) if min(r, k) > 1 else 0.0


def littles_mcar(df: pd.DataFrame) -> dict:
    """Little (1988) MCAR test with EM estimates under a multivariate normal model.

    Returns chi-square, degrees of freedom and p value. Uses pseudo inverses because several
    environmental covariates are almost collinear, so degrees of freedom are an upper bound.
    """
    X = df.values.astype(float)
    n, k = X.shape
    obs = ~np.isnan(X)
    mu = np.nanmean(X, axis=0)
    cov = np.diag(np.nanvar(X, axis=0) + 1e-9)
    for _ in range(30):  # EM
        sum_x = np.zeros(k); sum_xx = np.zeros((k, k))
        for pat in {tuple(r) for r in obs}:
            idx = np.where((obs == np.array(pat)).all(axis=1))[0]
            o = np.array(pat); m = ~o
            Xp = X[idx].copy()
            if m.any():
                coo = cov[np.ix_(o, o)]
                cmo = cov[np.ix_(m, o)]
                B = cmo @ np.linalg.pinv(coo)
                Xp[:, m] = mu[m] + (Xp[:, o] - mu[o]) @ B.T
                cond = cov[np.ix_(m, m)] - B @ cmo.T
                corr = np.zeros((k, k)); corr[np.ix_(m, m)] = cond * len(idx)
            else:
                corr = np.zeros((k, k))
            sum_x += Xp.sum(axis=0); sum_xx += Xp.T @ Xp + corr
        mu_new = sum_x / n
        cov = sum_xx / n - np.outer(mu_new, mu_new) + 1e-9 * np.eye(k)
        mu = mu_new
    d2, dof = 0.0, 0
    for pat in {tuple(r) for r in obs}:
        idx = np.where((obs == np.array(pat)).all(axis=1))[0]
        o = np.array(pat)
        if o.sum() == 0:
            continue
        diff = X[idx][:, o].mean(axis=0) - mu[o]
        d2 += len(idx) * diff @ np.linalg.pinv(cov[np.ix_(o, o)]) @ diff
        dof += int(o.sum())
    dof -= k
    return {"chi2": float(d2), "df": int(dof), "p_value": float(stats.chi2.sf(d2, dof)) if dof > 0 else None,
            "n_patterns": int(len({tuple(r) for r in obs}))}


# ----------------------------------------------------------------------------- main driver
def run_diagnostics(train: pd.DataFrame, feature_columns: list[str], figures_dir="figures",
                    stats_path="reports/diagnostic_stats.json", test: pd.DataFrame | None = None) -> dict:
    figures_dir = Path(figures_dir)
    X = train[feature_columns].copy()
    y = train["y"].values
    num = numeric_columns(feature_columns)
    prep, groups, _ = make_preprocessor_for(X, feature_columns)
    Z = prep.fit_transform(X)                       # fitted on train only
    res: dict = {"n_train": int(len(X)), "p_raw": int(X.shape[1]), "p_after_preprocessing": int(Z.shape[1])}

    # ---------------- Fig 1.1 missingness pattern
    miss = X.isna()
    miss_cols = miss.columns[miss.any()].tolist()
    res["missingness"] = {c: {"count": int(miss[c].sum()), "pct": round(float(miss[c].mean() * 100), 2)} for c in miss_cols}
    order = miss.sort_values(miss_cols, ascending=False).index if miss_cols else miss.index
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5.5), gridspec_kw={"width_ratios": [4, 1.3]})
    sns.heatmap(miss.loc[order].values[::20], cmap=["#E8EEF7", "#C44E52"], cbar=False, ax=ax, yticklabels=False)
    ax.set_xticks(np.arange(len(X.columns)) + 0.5)
    ax.set_xticklabels(X.columns, rotation=90, fontsize=6)
    ax.set_title("Missingness matrix (rows sorted by pattern, every 20th row; red = missing)")
    ax.set_xlabel("Feature"); ax.set_ylabel("Training rows")
    pct = (miss.mean() * 100).sort_values(ascending=False).head(8)
    ax2.barh(pct.index[::-1], pct.values[::-1], color=PALETTE["pos"])
    ax2.set_xlabel("% missing"); ax2.set_title("Top missing columns")
    for i, v in enumerate(pct.values[::-1]):
        ax2.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=8)
    ax2.set_xlim(0, max(60, pct.max() * 1.25))
    fig.tight_layout(); _save(fig, figures_dir, "fig1_1_missingness_heatmap.png")

    # missingness mechanism evidence
    num_block = X[num]
    res["littles_mcar"] = littles_mcar(num_block)
    res["cramers_v_missing_hml4"] = {}
    if "hml4" in X:
        ind = X["hml4"].isna().astype(int)
        for c in ["hml12", "hv024", "hv025", "hv270", "hv227", "hv104"]:
            if c in X:
                res["cramers_v_missing_hml4"][c] = round(cramers_v(ind, X[c].astype("string").fillna("NA")), 3)

    # ---------------- Fig 1.2 target distribution
    yt = pd.Series(y)
    sets = {"Train": yt}
    if test is not None:
        sets["Test"] = test["y"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cnt = pd.DataFrame({k: v.value_counts().reindex([0, 1], fill_value=0) for k, v in sets.items()})
    cnt.index = ["Negative", "Positive"]
    cnt.T.plot(kind="bar", ax=axes[0], color=[PALETTE["neg"], PALETTE["pos"]], rot=0)
    axes[0].set_yscale("log"); axes[0].set_ylabel("Members (log scale)"); axes[0].set_title("Class balance by split")
    for cont in axes[0].containers:
        axes[0].bar_label(cont, fmt="%d", fontsize=8)
    imb = float((yt == 0).sum() / (yt == 1).sum())
    axes[0].text(0.02, 0.96, f"Train imbalance {imb:.0f}:1  ({yt.mean()*100:.2f}% positive)", transform=axes[0].transAxes,
                 va="top", fontsize=9, bbox=dict(fc="white", ec=PALETTE["grey"]))
    age = X["hv105"]
    for lab, col in [(0, PALETTE["neg"]), (1, PALETTE["pos"])]:
        sns.kdeplot(age[yt.values == lab], ax=axes[1], color=col, fill=True, alpha=0.25, common_norm=False,
                    label=("Negative" if lab == 0 else "Positive"))
    axes[1].axvline(age.mean(), color="k", ls="--", lw=1); axes[1].axvline(age.median(), color="k", ls=":", lw=1)
    axes[1].text(age.mean() + 1, axes[1].get_ylim()[1] * 0.9, f"mean {age.mean():.1f}\nmedian {age.median():.1f}", fontsize=8)
    axes[1].set_xlabel("Age (years)"); axes[1].set_title("Age by test result (density, normalized per class)"); axes[1].legend()
    fig.tight_layout(); _save(fig, figures_dir, "fig1_2_target_distribution.png")
    res["target_train"] = {"n": int(len(yt)), "positives": int(yt.sum()), "prevalence_pct": round(float(yt.mean() * 100), 3),
                           "imbalance_ratio": round(imb, 1)}

    # ---------------- Fig 2.1 outliers raw vs winsorized
    skew = pd.Series(groups["skewness"])
    iqr_rows = {}
    for c in num:
        q1, q3 = X[c].quantile([.25, .75]); i = q3 - q1
        lo, hi = q1 - 1.5 * i, q3 + 1.5 * i
        n_out = int(((X[c] < lo) | (X[c] > hi)).sum())
        iqr_rows[c] = {"skew": round(float(skew[c]), 2), "kurtosis": round(float(stats.kurtosis(X[c].dropna())), 2),
                       "q1": round(float(q1), 3), "q3": round(float(q3), 3), "lower": round(float(lo), 3),
                       "upper": round(float(hi), 3), "n_outliers": n_out, "pct_outliers": round(n_out / len(X) * 100, 2)}
    res["outliers_iqr"] = iqr_rows
    top = sorted(iqr_rows, key=lambda c: iqr_rows[c]["n_outliers"], reverse=True)[:6]
    wins = IQRWinsorizer(factor=1.5).fit(X[top]); Xw = wins.transform(X[top])
    fig, axes = plt.subplots(2, 3, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), top):
        d = pd.DataFrame({"Raw": X[c], "Winsorized": Xw[c]}).melt(var_name="version", value_name="value")
        sns.boxplot(data=d, x="version", y="value", hue="version", ax=ax, palette=[PALETTE["grey"], PALETTE["neg"]], fliersize=1.5, legend=False)
        ax.set_title(f"{c}\nskew {iqr_rows[c]['skew']}, {iqr_rows[c]['n_outliers']} flagged ({iqr_rows[c]['pct_outliers']}%)", fontsize=9)
        ax.set_xlabel(""); ax.set_ylabel("")
    fig.suptitle("Outlier box plots: raw vs IQR winsorized (1.5 x IQR fences from training data)", y=1.01)
    fig.tight_layout(); _save(fig, figures_dir, "fig2_1_outlier_boxplots.png")

    # ---------------- transformed numeric block for correlation work
    tnum = [c for c in Z.columns if c in num]
    T = Z[tnum]
    corr = T.corr()
    flagged = [(a, b, float(corr.loc[a, b])) for a, b in combinations(tnum, 2) if abs(corr.loc[a, b]) >= 0.85]
    flagged.sort(key=lambda t: -abs(t[2]))
    res["collinear_pairs_r_ge_0.85"] = [{"a": a, "b": b, "r": round(r, 3)} for a, b, r in flagged]

    # ---------------- Fig 3.1 upper triangle heatmap
    mask = np.tril(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", vmin=-1, vmax=1, center=0, square=True, linewidths=0.2, ax=ax,
                cbar_kws={"label": "Pearson r (transformed training features)", "shrink": 0.7},
                xticklabels=True, yticklabels=True)
    pos_idx = {c: i for i, c in enumerate(tnum)}
    from matplotlib.patches import Rectangle
    for a, b, r in flagged:
        i, j = sorted([pos_idx[a], pos_idx[b]])
        ax.add_patch(Rectangle((j, i), 1, 1, fill=False, ec="black", lw=1.6))
    ax.tick_params(labelsize=6)
    ax.set_title(f"Upper triangle correlation heatmap; black boxes = {len(flagged)} pairs with |r| >= 0.85")
    fig.tight_layout(); _save(fig, figures_dir, "fig3_1_correlation_matrix.png")

    # ---------------- Fig 3.2 collinear pair scatterplots
    pairs = flagged[:4] if flagged else []
    if pairs:
        fig, axes = plt.subplots(1, len(pairs), figsize=(4 * len(pairs), 4))
        axes = np.atleast_1d(axes)
        samp = T.sample(min(3000, len(T)), random_state=42)
        for ax, (a, b, r) in zip(axes, pairs):
            sns.regplot(data=samp, x=a, y=b, ax=ax, scatter_kws={"s": 4, "alpha": 0.25, "color": PALETTE["neg"]},
                        line_kws={"color": PALETTE["pos"], "lw": 1.8})
            ax.set_title(f"r = {r:.2f}", fontsize=10); ax.tick_params(labelsize=7)
            ax.set_xlabel(a, fontsize=7); ax.set_ylabel(b, fontsize=7)
        fig.suptitle("Collinear feature pairs with fitted linear trend (standardized values, 3,000 point sample)", y=1.02)
        fig.tight_layout(); _save(fig, figures_dir, "fig3_2_collinear_scatterplots.png")

    # ---------------- Fig 3.3 feature to target importance
    Zf = Z.copy()
    discrete_mask = np.array([(c not in tnum) for c in Zf.columns])
    mi = mutual_info_classif(Zf.values, y, discrete_features=discrete_mask, random_state=42)
    pb = np.array([abs(stats.pointbiserialr(y, Zf[c].values)[0]) if Zf[c].std() > 0 else 0.0 for c in Zf.columns])
    imp = pd.DataFrame({"feature": Zf.columns, "mi": mi, "abs_r": pb}).sort_values("mi", ascending=False)
    res["target_importance_top15"] = imp.head(15).round(5).to_dict("records")
    top20 = imp.head(20).iloc[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 6), sharey=True)
    axes[0].barh(top20.feature, top20.mi, color=PALETTE["neg"]); axes[0].set_xlabel("Mutual information with test result (nats)")
    axes[1].barh(top20.feature, top20.abs_r, color=PALETTE["accent"]); axes[1].set_xlabel("|point biserial r|")
    axes[0].set_title("Ranked by mutual information"); axes[1].set_title("Linear association (same ranking order)")
    for ax in axes: ax.tick_params(labelsize=8)
    fig.suptitle("Feature to target importance (training data, top 20)", y=1.0)
    fig.tight_layout(); _save(fig, figures_dir, "fig3_3_target_importance.png")

    # ---------------- Fig 4.1 transformation grid on the most skewed right-skewed feature
    cand = groups["right_skewed"] or num
    feat = max(cand, key=lambda c: skew[c])
    raw = X[feat].dropna().values.reshape(-1, 1)
    logged = np.log1p(np.clip(raw, 0, None))
    mm = MinMaxScaler().fit_transform(logged)
    ss = StandardScaler().fit_transform(logged)
    steps = [("Raw", raw), ("log(1 + x)", logged), ("Min-Max after log", mm), ("Standardized after log", ss)]
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    tf = {}
    for ax, (t, arr) in zip(axes.ravel(), steps):
        v = arr.ravel()
        ax.hist(v, bins=50, color=PALETTE["neg"], alpha=0.85)
        ax.set_title(f"{t}: skew {stats.skew(v):.2f}, mean {v.mean():.2f}, sd {v.std():.2f}", fontsize=9)
        tf[t] = {"skew": round(float(stats.skew(v)), 3), "mean": round(float(v.mean()), 3), "std": round(float(v.std()), 3)}
    fig.suptitle(f"Transformation stages for '{feat}' (training data)", y=1.0)
    fig.tight_layout(); _save(fig, figures_dir, "fig4_1_transformation_grid.png")
    res["transformation_feature"] = feat; res["transformation_stats"] = tf

    # ---------------- Fig 5.1 PCA scree
    pca = PCA(random_state=42).fit(T.values)
    ev = pca.explained_variance_; evr = pca.explained_variance_ratio_; cum = np.cumsum(evr)
    k90, k95 = int(np.argmax(cum >= 0.90) + 1), int(np.argmax(cum >= 0.95) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(1, len(evr) + 1), evr * 100, color=PALETTE["neg"], label="Individual variance (%)")
    ax2 = ax.twinx()
    ax2.plot(range(1, len(cum) + 1), cum * 100, color=PALETTE["pos"], marker="o", ms=3, label="Cumulative variance (%)")
    for th, k, ls in [(90, k90, "--"), (95, k95, ":")]:
        ax2.axhline(th, color="k", ls=ls, lw=1); ax2.axvline(k, color="k", ls=ls, lw=1)
        ax2.text(len(cum) * 0.55, th - 4, f"{th}% reached at {k} components", fontsize=8)
    ax.set_xlabel("Principal component"); ax.set_ylabel("Explained variance (%)"); ax2.set_ylabel("Cumulative (%)")
    ax2.set_ylim(0, 102); ax2.grid(False)
    ax.set_title(f"PCA scree and cumulative variance ({len(cum)} scaled numeric features)")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); ax.legend(h1 + h2, l1 + l2, loc="center right")
    fig.tight_layout(); _save(fig, figures_dir, "fig5_1_pca_scree_plot.png")
    res["pca"] = {"n_features": int(len(cum)), "k90": k90, "k95": k95,
                  "first_eigenvalues": [round(float(e), 3) for e in ev[:8]],
                  "first_variance_ratios": [round(float(r), 4) for r in evr[:8]]}

    res["skewness_summary"] = {"right_skewed": groups["right_skewed"], "other_skewed": groups["other_skewed"], "gaussian": groups["gaussian"]}
    Path(stats_path).parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(res, f, indent=2, default=str)
    return res


def collinearity_report(train: pd.DataFrame, feature_columns: list[str], path="reports/collinearity_stats.json") -> dict:
    """VIF before/after the linear-branch screen, dropped features, and MI of each flagged pair member."""
    import warnings
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from src.transformers import CollinearityScreen
    X, y = train[feature_columns], train["y"].values
    prep, _, num = make_preprocessor_for(X, feature_columns)
    Z = prep.fit_transform(X)
    T = Z[[c for c in Z.columns if c in num]]

    def vifs(df):
        A = df.values.astype(float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return {c: float(variance_inflation_factor(A, i)) for i, c in enumerate(df.columns)}
    before = vifs(T)
    scr = CollinearityScreen(corr_threshold=0.85, vif_threshold=10.0, candidate_columns=list(T.columns)).fit(T, y)
    dropped_names = [d[0] for d in scr.dropped_]
    kept = T[[c for c in T.columns if c not in dropped_names]]
    after = vifs(kept)
    mi = dict(zip(T.columns, mutual_info_classif(T.values, y, random_state=42)))
    spear = {c: float(stats.spearmanr(T[c], y)[0]) for c in T.columns}
    pear = {c: float(stats.pearsonr(T[c], y)[0]) for c in T.columns}
    inf = [c for c, v in before.items() if not np.isfinite(v)]
    out = {"n_numeric": int(T.shape[1]), "max_vif_before": float(max(v for v in before.values() if np.isfinite(v))),
           "n_infinite_vif_before": len(inf), "n_vif_gt10_before": int(sum((v > 10) or not np.isfinite(v) for v in before.values())),
           "top_vif_before": sorted(((c, round(v, 1)) for c, v in before.items() if np.isfinite(v)), key=lambda t: -t[1])[:8],
           "dropped": [list(d) for d in scr.dropped_], "n_dropped": len(scr.dropped_), "n_kept": int(kept.shape[1]),
           "max_vif_after": float(max(after.values())), "dropped_mi": {d[0]: round(float(mi[d[0]]), 5) for d in scr.dropped_},
           "spearman_vs_pearson_top": sorted(((c, round(spear[c], 3), round(pear[c], 3)) for c in T.columns), key=lambda t: -abs(t[1]))[:6]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(path, "w"), indent=2, default=str)
    return out
