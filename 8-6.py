#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fig. 8.14–8.16 and Table 8.10–8.11 generator (EN-only figure text, short model names)

Inputs
- model_compare_out/joined_all_models.csv
  * Must include: model, track, problem_id
  * Must include per-problem failure counts/labels for WA/RE/TLE/CE (any of: count_*, label_*, n_*)
  * Optional: n_attempts (otherwise attempts = sum of AC + failures)
  * Optional: w (difficulty weight); if absent, merged from DIFF_WEIGHTS_CSV
- difficulty_weights_A_baseline_only.csv (problem_id, w or w_p)

Outputs
- out/viz_failures/fig_8_14_fail_comp_stacked.png
- out/viz_failures/fig_8_15_fail_diff_vs_ref.png
- out/viz_failures/fig_8_16_fail_comp_by_quartile.png
- out/viz_failures/table_8_10_fail_rates.csv
- out/viz_failures/table_8_11_quartile_fail_rates.csv

Notes
- All figure text is English-only.
- Model labels on Figs 8.14, 8.16 are shortened to "llama", "deepseek" to avoid overlap.
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# Config (edit here)
# =========================
JOINED_CSV = "model_compare_out/joined_all_models.csv"  # make_model_compare.py が出力した統合CSV
DIFF_WEIGHTS_CSV = "./Llama-3.1-8B-Instruct/results/difficulty_weights_A_baseline_only.csv"  # 重み（無いときは w=1）
OUTDIR = "figs_8_6"

# Pair for Fig. 8.15 (keep long names here to match CSV; will shorten only in figure labels)
REF_MODEL = "Llama-3.1-8B-Instruct"
CMP_MODEL = "deepseek-coder-7b-instruct-v1.5"

# Track order (only those present will be used; others appended at end)
TRACKS_ORDER = ["baseline", "CoT", "feedback"]

# Which track to use on Fig. 8.16 (quartile bars)
QUARTILE_TRACK = "feedback"

# Bootstrapping
N_BOOT = 1000
ALPHA = 0.05

# Failure categories to search (columns can be count_*, label_* or n_*)
FAIL_CANDIDATES = ["WA", "RE", "TLE", "CE", "MLE"]  # MLE is optional

# Short display name mapping for figures (fallback to lowercase token if not found)
MODEL_NAME_SHORT = {
    "Llama-3.1-8B-Instruct": "llama",
    "deepseek-coder-7b-instruct-v1.5": "deepseek",
}

# =========================
# Utils
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def bootstrap_mean_ci(x: np.ndarray, n_boot=1000, alpha=0.05, seed=1729):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.mean(x))
    if x.size == 1 or n_boot <= 0:
        return (mean, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(np.mean(x[idx]))
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return (mean, lo, hi)

def bootstrap_weighted_mean_ci(x: np.ndarray, w: np.ndarray, n_boot=1000, alpha=0.05, seed=1729):
    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x = x[m]; w = w[m]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    mean = float(np.sum(w*x) / np.sum(w))
    if x.size == 1 or n_boot <= 0:
        return (mean, np.nan, np.nan)
    rng = np.random.default_rng(seed)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]; wb = w[idx]
        boots[i] = float(np.sum(wb*xb) / np.sum(wb))
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return (mean, lo, hi)

def detect_fail_columns(df: pd.DataFrame):
    def pick_col(cat):
        for prefix in ["count_", "label_", "n_"]:
            col = f"{prefix}{cat}"
            if col in df.columns:
                return col
        return None
    cat_cols = {}
    for cat in FAIL_CANDIDATES + ["AC"]:
        col = pick_col(cat)
        if col is not None:
            cat_cols[cat] = col
    attempts_col = "n_attempts" if "n_attempts" in df.columns else None
    return cat_cols, attempts_col

def compute_per_problem_rates(df: pd.DataFrame, cat_cols: dict, attempts_col: str | None):
    df = df.copy()
    for col in cat_cols.values():
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if attempts_col and attempts_col in df.columns:
        den = pd.to_numeric(df[attempts_col], errors="coerce")
    else:
        cols_sum = [cat_cols[c] for c in cat_cols.keys()]
        den = df[cols_sum].sum(axis=1)
    den = den.replace(0, np.nan)
    df["_den_attempts"] = den
    used_cats = [c for c in FAIL_CANDIDATES if c in cat_cols]
    for cat in used_cats:
        df[f"rate_{cat}"] = df[cat_cols[cat]] / den
    if "AC" in cat_cols:
        df["rate_AC"] = df[cat_cols["AC"]] / den
    return df, used_cats

def load_weights_if_needed(df: pd.DataFrame, weights_csv: str):
    if "w" in df.columns:
        return df
    if not Path(weights_csv).exists():
        df["w"] = 1.0
        return df
    w = pd.read_csv(weights_csv)
    if "w_p" in w.columns and "w" not in w.columns:
        w = w.rename(columns={"w_p": "w"})
    df = df.merge(w[["problem_id", "w"]], on="problem_id", how="left")
    df["w"] = df["w"].fillna(1.0)
    return df

def restrict_tracks_order(df: pd.DataFrame, order: list[str]) -> list[str]:
    exists = [t for t in order if t in df["track"].unique().tolist()]
    for t in df["track"].unique():
        if t not in exists:
            exists.append(t)
    return exists

def to_short_model(name: str) -> str:
    if name in MODEL_NAME_SHORT:
        return MODEL_NAME_SHORT[name]
    # fallback: take first token lowercased
    return str(name).split()[0].lower()

# =========================
# Main
# =========================
def main():
    outdir = Path(OUTDIR); ensure_dir(outdir)
    df = pd.read_csv(JOINED_CSV)
    if "problem_id" in df.columns:
        df["problem_id"] = df["problem_id"].astype(str)

    # weights
    df = load_weights_if_needed(df, DIFF_WEIGHTS_CSV)

    # failure columns
    cat_cols, attempts_col = detect_fail_columns(df)
    if not any(c in cat_cols for c in ["WA", "RE", "TLE", "CE"]):
        raise RuntimeError("No failure columns found. Need count_/label_/n_ for WA/RE/TLE/CE.")
    df_rates, used_cats = compute_per_problem_rates(df, cat_cols, attempts_col)

    # track order
    track_order = restrict_tracks_order(df_rates, TRACKS_ORDER)

    # ========================
    # Table 8.10 (model × track)
    # ========================
    rows = []
    for (model, track), g in df_rates.groupby(["model", "track"], sort=False):
        row = {"model": model, "track": track, "n_problems": int(g["problem_id"].nunique())}
        for cat in used_cats:
            m, lo, hi = bootstrap_mean_ci(g[f"rate_{cat}"].to_numpy(), N_BOOT, ALPHA)
            row[f"{cat}_mean"] = m; row[f"{cat}_ci_lo"] = lo; row[f"{cat}_ci_hi"] = hi
        rows.append(row)
    tab_8_10 = pd.DataFrame(rows).sort_values(["track", "model"])
    tab_8_10_path = outdir / "table_8_10_fail_rates.csv"
    tab_8_10.to_csv(tab_8_10_path, index=False)

    # ========================
    # Fig. 8.14 (stacked; model × track)  [EN-only; short model names]
    # ========================
    plot_df = tab_8_10.copy()
    plot_df["track"] = pd.Categorical(plot_df["track"], categories=track_order, ordered=True)
    plot_df["model_short"] = plot_df["model"].map(to_short_model)
    plot_df = plot_df.sort_values(["track", "model_short"])

    xlabels = [f"{t}\n{ms}" for t, ms in zip(plot_df["track"], plot_df["model_short"])]
    x = np.arange(len(plot_df))
    fig = plt.figure(figsize=(max(9, len(xlabels)*0.9), 5))
    bottom = np.zeros(len(plot_df))
    for cat in used_cats:
        y = plot_df[f"{cat}_mean"].to_numpy()
        plt.bar(x, y, bottom=bottom, label=cat)
        bottom += np.nan_to_num(y)
    plt.xticks(x, xlabels, rotation=0, fontsize=9)
    plt.ylabel("Failure composition (rate)")
    plt.title("Failure composition (stacked; model × track)")
    plt.legend(title="Category")
    fig_814 = outdir / "fig_8_14_fail_comp_stacked.png"
    plt.tight_layout()
    plt.savefig(fig_814, dpi=200)
    plt.close(fig)

    # ========================
    # Fig. 8.15 (difference: cmp - ref) [EN-only; short names in labels]
    # ========================
    models = df_rates["model"].unique().tolist()
    if REF_MODEL not in models or CMP_MODEL not in models:
        if len(models) < 2:
            raise RuntimeError("Need at least two models for difference plot.")
        ref = models[0]; cmpm = models[1]
    else:
        ref = REF_MODEL; cmpm = CMP_MODEL
    ref_short = to_short_model(ref)
    cmp_short = to_short_model(cmpm)

    rows = []
    for track, g in df_rates.groupby("track", sort=False):
        g_ref = g[g["model"]==ref].set_index("problem_id")
        g_cmp = g[g["model"]==cmpm].set_index("problem_id")
        common = g_ref.index.intersection(g_cmp.index)
        if len(common) == 0:
            continue
        gr = g_ref.loc[common]; gc = g_cmp.loc[common]
        for cat in used_cats:
            diff = gc[f"rate_{cat}"].to_numpy() - gr[f"rate_{cat}"].to_numpy()
            m, lo, hi = bootstrap_mean_ci(diff, N_BOOT, ALPHA)
            rows.append({"track": track, "cat": cat, "diff_mean": m, "diff_ci_lo": lo, "diff_ci_hi": hi})
    diff_df = pd.DataFrame(rows)
    diff_df["track"] = pd.Categorical(diff_df["track"], categories=track_order, ordered=True)
    diff_df = diff_df.sort_values(["track", "cat"])

    cats_order = [c for c in FAIL_CANDIDATES if c in used_cats]
    xpos = []; vals = []; ci_lo = []; ci_hi = []; xticklabels = []
    for t in track_order:
        sub = diff_df[diff_df["track"]==t]
        for c in cats_order:
            r = sub[sub["cat"]==c]
            if r.empty:
                continue
            xticklabels.append(f"{t}\n{c}")
            vals.append(float(r["diff_mean"].iloc[0]))
            ci_lo.append(float(r["diff_ci_lo"].iloc[0]))
            ci_hi.append(float(r["diff_ci_hi"].iloc[0]))
    x = np.arange(len(vals))
    fig = plt.figure(figsize=(max(9, len(xticklabels)*0.8), 4.8))
    plt.bar(x, vals)
    # CI whiskers
    for i, (m, lo, hi) in enumerate(zip(vals, ci_lo, ci_hi)):
        if not (math.isnan(lo) or math.isnan(hi)):
            plt.vlines(i, lo, hi)
            plt.hlines([lo, hi], i-0.15, i+0.15)
    plt.axhline(0, linestyle="--")
    plt.xticks(x, xticklabels, rotation=0, fontsize=9)
    plt.ylabel(f"{cmp_short} − {ref_short} (rate)")
    plt.title(f"Failure rate difference ({cmp_short} − {ref_short})")
    fig_815 = outdir / "fig_8_15_fail_diff_vs_ref.png"
    plt.tight_layout()
    plt.savefig(fig_815, dpi=200)
    plt.close(fig)

    # ========================
    # Quartiles (w) and Table 8.11
    # ========================
    w_uni = df_rates[["problem_id", "w"]].drop_duplicates("problem_id").set_index("problem_id")["w"]
    qs = np.quantile(w_uni.values, [0.25, 0.5, 0.75])
    def to_bin(v):
        if v <= qs[0]: return "Q1 (easy)"
        elif v <= qs[1]: return "Q2"
        elif v <= qs[2]: return "Q3"
        else: return "Q4 (hard)"
    wbin = w_uni.apply(to_bin)
    df_rates = df_rates.merge(wbin.rename("w_bin").reset_index(), on="problem_id", how="left")

    rows = []
    for (model, track, w_b), g in df_rates.groupby(["model", "track", "w_bin"], sort=False):
        gw = g["w"].to_numpy()
        row = {"model": model, "track": track, "w_bin": w_b, "n_problems": int(g["problem_id"].nunique())}
        for cat in used_cats:
            x = g[f"rate_{cat}"].to_numpy()
            m, lo, hi = bootstrap_mean_ci(x, N_BOOT, ALPHA)
            dm, dlo, dhi = bootstrap_weighted_mean_ci(x, gw, N_BOOT, ALPHA)
            row[f"{cat}_mean"] = m; row[f"{cat}_ci_lo"] = lo; row[f"{cat}_ci_hi"] = hi
            row[f"{cat}_Dmean"] = dm; row[f"{cat}_Dci_lo"] = dlo; row[f"{cat}_Dci_hi"] = dhi
        rows.append(row)
    tab_8_11 = pd.DataFrame(rows).sort_values(["track", "w_bin", "model"])
    tab_8_11_path = outdir / "table_8_11_quartile_fail_rates.csv"
    tab_8_11.to_csv(tab_8_11_path, index=False)

    # ========================
    # Fig. 8.16 (quartile stacked for a chosen track) [EN-only; short model names]
    # ========================
    qdf = tab_8_11[tab_8_11["track"] == QUARTILE_TRACK].copy()
    qdf["model_short"] = qdf["model"].map(to_short_model)
    q_order = ["Q1 (easy)", "Q2", "Q3", "Q4 (hard)"]
    qdf["w_bin"] = pd.Categorical(qdf["w_bin"], categories=[q for q in q_order if q in qdf["w_bin"].unique()], ordered=True)
    qdf = qdf.sort_values(["w_bin", "model_short"])

    xlabels = [f"{wb}\n{ms}" for wb, ms in zip(qdf["w_bin"], qdf["model_short"])]
    x = np.arange(len(qdf))
    fig = plt.figure(figsize=(max(10, len(xlabels)*0.9), 5))
    bottom = np.zeros(len(qdf))
    for cat in used_cats:
        y = qdf[f"{cat}_mean"].to_numpy()
        plt.bar(x, y, bottom=bottom, label=cat)
        bottom += np.nan_to_num(y)
    plt.xticks(x, xlabels, rotation=0, fontsize=9)
    plt.ylabel(f"Failure composition (rate)  [{QUARTILE_TRACK}]")
    plt.title("Failure composition by difficulty quartile")
    plt.legend(title="Category")
    fig_816 = outdir / "fig_8_16_fail_comp_by_quartile.png"
    plt.tight_layout()
    plt.savefig(fig_816, dpi=200)
    plt.close(fig)

    # Print summary
    print("=== DONE ===")
    print("Tables:")
    print("-", str(tab_8_10_path))
    print("-", str(tab_8_11_path))
    print("Figures:")
    print("-", str(fig_814))
    print("-", str(fig_815))
    print("-", str(fig_816))

if __name__ == "__main__":
    main()
