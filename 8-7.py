# -*- coding: utf-8 -*-
"""
Figure 8.17 (Coefficient forest), Figure 8.18 (Added-variable plots), Table 8.11 (WLS coefficients)
— generated in one shot.

Assumptions (edit CONFIG below):
- Per-problem results for each model are stored at:
    <MODEL_DIR>/<RESULTS_CSV>  with columns:
      ['problem_id','track','pass_at_1','pass_at_10', ...]  (k=1,10 used here)
- Difficulty weights and axes are available as either:
    (A) difficulty_axes.csv  with ['problem_id','z_B','z_T','z_H','z_WA']
        (already standardized; z-score)
    or (B) raw sources to build axes:
        - problem_metrics_bytes_time.csv  with ['problem_id','bytes','time_ms' (or 'time_sec')]
        - human_rates.csv                 with ['problem_id','human_ac_rate','human_wa_rate']
      -> this script standardizes to z_B,z_T,z_H,z_WA
- Common weights file:
    difficulty_weights_A_baseline_only.csv with ['problem_id','w'] or ['problem_id','w_p']

Outputs:
- fig_out/fig_8_17_coefficients.png
- fig_out/fig_8_18_added_variable.png   (added-variable plots for baseline @1; 4 predictors)
- fig_out/table_8_11_coefficients.csv   (WLS coef, HC3 SE/CI, Holm-adjusted p)
"""

import os
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# ============ CONFIG (edit here) ============
MODEL_FILES = {
    "llama":     {"dir": "./Llama-3.1-8B-Instruct",         "file": "results/per_problem_pass_at_k.csv"},
    "deepseek":  {"dir": "./deepseek-coder-7b-instruct-v1.5","file": "results/per_problem_pass_at_k.csv"},
}
TRACKS = ["baseline","CoT","feedback"]
K_LIST = [1, 10]  # we report @1 / @10 in this section

# Difficulty axes sources
AXES_FILE = "./Llama-3.1-8B-Instruct/results/difficulty_metrics_all_axes_baseline_only.csv"  # preferred (z_B, z_T, z_H, z_WA)
BYTES_TIME_FILE = "./problem_metrics_bytes_time.csv"  # fallback to build axes
HUMAN_FILE = "./human_rates.csv"                     # fallback to build axes

# Common weights (w)
WEIGHTS_FILE = "./Llama-3.1-8B-Instruct/results/difficulty_weights_A_baseline_only.csv"

# Output dir
OUTDIR = Path("./fig_8_7")

# Matplotlib settings (no custom colors; English labels)
plt.rcParams.update({"figure.dpi": 140})

# ===========================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_per_problem(model_key: str) -> pd.DataFrame:
    info = MODEL_FILES[model_key]
    path = Path(info["dir"]) / info["file"]
    if not path.is_file():
        raise FileNotFoundError(f"Per-problem file not found for {model_key}: {path}")
    df = pd.read_csv(path)
    # normalize column names
    if "problem" in df.columns and "problem_id" not in df.columns:
        df = df.rename(columns={"problem":"problem_id"})
    return df

def read_weights() -> pd.DataFrame:
    if not Path(WEIGHTS_FILE).is_file():
        print(f"[WARN] weights file missing: {WEIGHTS_FILE}; using w=1")
        return pd.DataFrame({"problem_id": [], "w": []})
    w = pd.read_csv(WEIGHTS_FILE)
    if "w" not in w.columns and "w_p" in w.columns:
        w = w.rename(columns={"w_p":"w"})
    w = w[["problem_id","w"]].copy()
    return w

def zscore(x: pd.Series) -> pd.Series:
    x = x.astype(float)
    mu = x.mean()
    sd = x.std(ddof=0)
    if sd == 0 or np.isclose(sd, 0.0):
        return pd.Series(np.zeros(len(x)), index=x.index)
    return (x - mu) / sd

def build_axes_from_raw() -> pd.DataFrame:
    if not (Path(BYTES_TIME_FILE).is_file() and Path(HUMAN_FILE).is_file()):
        raise FileNotFoundError("Axes file not found and raw sources missing. Provide difficulty_axes.csv or both raw files.")
    bt = pd.read_csv(BYTES_TIME_FILE)     # expect: problem_id, bytes, time_ms (or time_sec)
    hr = pd.read_csv(HUMAN_FILE)          # expect: problem_id, human_ac_rate, human_wa_rate
    for col in ["problem_id"]:
        if col not in bt.columns or col not in hr.columns:
            raise ValueError("problem_id missing in raw axes sources.")
    # unify time column
    if "time_ms" in bt.columns:
        t = bt["time_ms"].astype(float)
    elif "time_sec" in bt.columns:
        t = bt["time_sec"].astype(float) * 1000.0
    else:
        raise ValueError("time_ms or time_sec is required in problem_metrics_bytes_time.csv")

    # log-transform for scale robustness
    bytes_log = np.log1p(bt["bytes"].astype(float))
    time_log  = np.log1p(t)

    # define human difficulty proxies (higher = harder)
    # z_H: human hard rate = 1 - AC rate
    if "human_ac_rate" not in hr.columns:
        raise ValueError("human_ac_rate is required in human_rates.csv")
    human_hard = 1.0 - hr["human_ac_rate"].astype(float)
    # z_WA: human wrong answer tendency (if not present, fall back to human_hard)
    if "human_wa_rate" in hr.columns:
        human_wa = hr["human_wa_rate"].astype(float)
    else:
        human_wa = human_hard.copy()

    axes = (
        bt[["problem_id"]].merge(hr[["problem_id"]], on="problem_id", how="outer")
          .merge(bt[["problem_id"]], on="problem_id", how="outer")
    ).drop_duplicates("problem_id")
    axes = axes.merge(bt[["problem_id"]], on="problem_id", how="left")  # anchor all ids from bt

    # Now assemble and z-score
    tmp = pd.DataFrame({
        "problem_id": bt["problem_id"],
        "z_B": zscore(bytes_log),
        "z_T": zscore(time_log),
    })
    tmp2 = pd.DataFrame({
        "problem_id": hr["problem_id"],
        "z_H": zscore(human_hard),
        "z_WA": zscore(human_wa),
    })
    axes = tmp.merge(tmp2, on="problem_id", how="inner")
    return axes

def read_axes() -> pd.DataFrame:
    if Path(AXES_FILE).is_file():
        df = pd.read_csv(AXES_FILE)
        needed = {"problem_id","z_B","z_T","z_H","z_WA"}
        if not needed.issubset(df.columns):
            raise ValueError(f"{AXES_FILE} must contain columns: {needed}")
        return df[["problem_id","z_B","z_T","z_H","z_WA"]].copy()
    # Build from raw
    print("[INFO] difficulty_axes.csv not found; building axes from raw sources...")
    return build_axes_from_raw()

def assemble_delta(axes: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    llama = read_per_problem("llama")
    deep  = read_per_problem("deepseek")

    # Keep only requested tracks
    llama = llama[llama["track"].isin(TRACKS)].copy()
    deep  = deep [deep ["track"].isin(TRACKS)].copy()

    # Choose needed k columns
    need_cols = ["problem_id","track"] + [f"pass_at_{k}" for k in K_LIST]
    llama = llama[[c for c in need_cols if c in llama.columns]].copy()
    deep  = deep [[c for c in need_cols if c in deep .columns]].copy()

    # Inner join on problem_id,track to ensure paired comparison
    L = llama.set_index(["problem_id","track"])
    D = deep .set_index(["problem_id","track"])
    common = L.index.intersection(D.index)
    L = L.loc[common].reset_index()
    D = D.loc[common].reset_index()

    # Merge into one table with both models' pass@k
    merged = L.merge(D, on=["problem_id","track"], suffixes=("_llama","_deep"))
    for k in K_LIST:
        merged[f"delta_at_{k}"] = merged[f"pass_at_{k}_deep"] - merged[f"pass_at_{k}_llama"]

    # Attach axes & weights
    merged = merged.merge(axes, on="problem_id", how="left")
    if weights.empty:
        merged["w"] = 1.0
    else:
        merged = merged.merge(weights, on="problem_id", how="left")
        merged["w"] = merged["w"].fillna(1.0)
    # drop rows with any NaN in predictors or deltas used
    for k in K_LIST:
        merged = merged[merged[f"delta_at_{k}"].notna()]
    merged = merged.dropna(subset=["z_B","z_T","z_H","z_WA","w"])
    return merged

def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0]*m
    running_max = 0.0
    for rank, idx in enumerate(order, start=1):
        adj_p = (m - rank + 1) * pvals[idx]
        adj_p = min(1.0, adj_p)
        running_max = max(running_max, adj_p)
        adj[idx] = running_max
    return adj

def fit_wls_block(df: pd.DataFrame, track: str, k: int) -> pd.DataFrame:
    sub = df[df["track"]==track].copy()
    y = sub[f"delta_at_{k}"].to_numpy()
    X = sub[["z_B","z_T","z_H","z_WA"]].to_numpy()
    X = sm.add_constant(X)  # [const, z_B, z_T, z_H, z_WA]
    w = sub["w"].to_numpy()
    if X.shape[0] < X.shape[1]:
        raise ValueError(f"Insufficient rows for WLS in track={track}, k={k}: n={X.shape[0]}")
    model = sm.WLS(y, X, weights=w)
    res = model.fit(cov_type="HC3")  # robust SE
    # Extract
    names = ["const","z_B","z_T","z_H","z_WA"]
    coefs = res.params
    ses   = res.bse
    pvals = res.pvalues
    ci    = res.conf_int(alpha=0.05)
    out_rows = []
    for i, nm in enumerate(names):
        row = {
            "track": track,
            "k": k,
            "term": nm,
            "coef": float(coefs[i]),
            "se_HC3": float(ses[i]),
            "ci_lo": float(ci[i,0]),
            "ci_hi": float(ci[i,1]),
            "p_raw": float(pvals[i]),
            "n_obs": int(len(sub)),
            "w_sum": float(sub["w"].sum()),
        }
        out_rows.append(row)
    # Holm per (track,k) over predictors only (exclude const)
    pred_mask = [r["term"]!="const" for r in out_rows]
    p_pred = [r["p_raw"] for r in out_rows if r["term"]!="const"]
    if len(p_pred) > 0:
        p_adj = holm_bonferroni(p_pred)
        j = 0
        for r in out_rows:
            if r["term"]=="const":
                r["p_holm"] = np.nan
            else:
                r["p_holm"] = float(p_adj[j]); j+=1
    else:
        for r in out_rows:
            r["p_holm"] = np.nan
    return pd.DataFrame(out_rows)

def build_table_coefficients(delta_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for tr in TRACKS:
        for k in K_LIST:
            df = fit_wls_block(delta_df, tr, k)
            rows.append(df)
    tbl = pd.concat(rows, ignore_index=True)
    return tbl

def plot_forest(tbl: pd.DataFrame, out_path: Path):
    # Forest across track×k in a grid (3 tracks × 2 ks)
    tracks = TRACKS
    ks = K_LIST
    nrows = len(tracks)
    ncols = len(ks)
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10), sharex=False)
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    for i,tr in enumerate(tracks):
        for j,k in enumerate(ks):
            ax = axes[i,j]
            sub = tbl[(tbl["track"]==tr) & (tbl["k"]==k) & (tbl["term"]!="const")].copy()
            # order as z_B, z_T, z_H, z_WA
            order = ["z_B","z_T","z_H","z_WA"]
            sub["term"] = pd.Categorical(sub["term"], order)
            sub = sub.sort_values("term")
            y_pos = np.arange(len(sub))[::-1]
            ax.errorbar(sub["coef"], y_pos, xerr=[sub["coef"]-sub["ci_lo"], sub["ci_hi"]-sub["coef"]],
                        fmt='o', capsize=3)
            ax.axvline(0.0, linewidth=1, linestyle="--")
            ax.set_yticks(y_pos)
            ax.set_yticklabels(sub["term"])
            ax.set_xlabel("Coefficient (β)")
            ax.set_title(f"Forest: {tr} @k={k}")
    fig.suptitle("Coefficient Forest (WLS, HC3)", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.97])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def added_variable_data(df: pd.DataFrame, track: str, k: int) -> Dict[str, Tuple[np.ndarray,np.ndarray,float]]:
    """
    Returns residual pairs (x_resid, y_resid, slope) for each predictor in ['z_B','z_T','z_H','z_WA'].
    slope is from regressing y_resid ~ x_resid (OLS).
    """
    sub = df[df["track"]==track].copy()
    y = sub[f"delta_at_{k}"].to_numpy()
    X_full = sub[["z_B","z_T","z_H","z_WA"]].copy()
    results = {}
    for col in ["z_B","z_T","z_H","z_WA"]:
        other = [c for c in X_full.columns if c != col]
        # y_resid: regress y on other
        Xy = sm.add_constant(X_full[other].to_numpy())
        ry = sm.OLS(y, Xy).fit()
        y_resid = ry.resid
        # x_resid: regress col on other
        Xx = sm.add_constant(X_full[other].to_numpy())
        rx = sm.OLS(X_full[col].to_numpy(), Xx).fit()
        x_resid = rx.resid
        # slope on residuals
        rz = sm.OLS(y_resid, sm.add_constant(x_resid)).fit()
        results[col] = (x_resid, y_resid, float(rz.params[1]))
    return results

def plot_added_variable(df: pd.DataFrame, out_path: Path, track="baseline", k=1):
    """
    2x2 subplots for z_B, z_T, z_H, z_WA on (track,k).
    """
    av = added_variable_data(df, track, k)
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    names = [("z_B","Implementation (z_B)"),
             ("z_T","Runtime (z_T)"),
             ("z_H","Human diff. (z_H)"),
             ("z_WA","Human WA (z_WA)")]
    for ax,(key,title) in zip(axes.ravel(), names):
        x_resid, y_resid, slope = av[key]
        ax.scatter(x_resid, y_resid, s=20)
        # simple OLS line on residuals
        X = sm.add_constant(x_resid)
        reg = sm.OLS(y_resid, X).fit()
        x_line = np.linspace(x_resid.min(), x_resid.max(), 100)
        y_line = reg.params[0] + reg.params[1]*x_line
        ax.plot(x_line, y_line)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_title(title)
        ax.set_xlabel("Predictor residual")
        ax.set_ylabel("Δ residual")
    fig.suptitle(f"Added-variable plots ({track} @k={k})", fontsize=12)
    fig.tight_layout(rect=[0,0,1,0.95])
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)

def main():
    ensure_dir(OUTDIR)
    # 1) Read axes & weights
    axes = read_axes()
    weights = read_weights()
    # 2) Assemble per-problem deltas for k in K_LIST
    delta_df = assemble_delta(axes, weights)
    # 3) Build coefficient table (WLS, HC3; Holm per (track,k))
    tbl = build_table_coefficients(delta_df)
    # 4) Save Table 8.11
    tbl_path = OUTDIR / "table_8_11_coefficients.csv"
    tbl.to_csv(tbl_path, index=False)
    print(f"[OK] wrote {tbl_path}  rows={len(tbl)}")
    # 5) Fig.8.17 (forest)
    fig17 = OUTDIR / "fig_8_17_coefficients.png"
    plot_forest(tbl, fig17)
    print(f"[OK] wrote {fig17}")
    # 6) Fig.8.18 (added-variable plots) — baseline @1 (most informative)
    fig18 = OUTDIR / "fig_8_18_added_variable.png"
    plot_added_variable(delta_df, fig18, track="baseline", k=1)
    print(f"[OK] wrote {fig18}")

if __name__ == "__main__":
    main()
