# -*- coding: utf-8 -*-
"""
Fig. 8.10 / Fig. 8.11 / Table 8.7 generator

- 図8.10: Δpass@k 曲線（DeepSeek − Llama、k=1..100、track別）
- 図8.11: ΔD-pass@k 曲線（同上、難易度加重版）
- 表8.7  : 代表点（k=1/10/100）の Δ平均・95%CI・勝率（win/tie/loss）・Holm補正後p

入力:
  - overall_curves_mean_and_D.csv   （必須: make_model_compare.py の出力）
  - pairwise_vs_ref_sign_test.csv   （必須: 同上の出力）
  - joined_all_models.csv           （任意: あれば "Δの95%CI" を "問題ペアのブートストラップ" で算出）
                                    （ない場合は per-model CI から保守的に差のCIを構成）

出力:
  - fig8_outputs/fig8_10_delta_pass_<track>.png     （track ∈ {baseline, CoT, feedback}、各1枚）
  - fig8_outputs/fig8_11_delta_Dpass_<track>.png    （同上）
  - fig8_outputs/table8_7_delta_summary.csv

注意:
  - seabornは使いません。各図は単独プロット（サブプロットなし）。色指定はしません。
  - k=1/10/100 が存在しない場合は、存在するkのみ表に出します。
  - "Holm補正後p" は pairwise_vs_ref_sign_test.csv 内の値を利用（既に補正済みが入っている想定）。
"""

import os
import re
import math
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG（必要に応じて編集）
# =========================
CURVES_CSV   = "./model_compare_out/overall_curves_mean_and_D.csv"
PAIRWISE_CSV = "./model_compare_out/pairwise_vs_ref_sign_test.csv"
JOINED_CSV   = "./model_compare_out/joined_all_models.csv"   # 任意。無ければ "" のままでOK
OUTDIR       = "./figs_8_4"

# モデル・トラック
REF_MODEL = "Llama-3.1-8B-Instruct"
CMP_MODEL = "deepseek-coder-7b-instruct-v1.5"
TRACKS    = ["baseline", "CoT", "feedback"]

# 統計
ALPHA  = 0.05
N_BOOT = 5000  # JOINEDがある場合のペア・ブートストラップ反復回数

# =========================
# UTILS
# =========================
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def load_csv_must(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)

def paired_boot_ci(x: np.ndarray, w: np.ndarray = None, n_boot: int = 2000, alpha: float = 0.05) -> Tuple[float,float,float]:
    """
    x: per-problem paired differences (DeepSeek - Llama), shape (n,)
    w: optional weights per problem, same shape
    Return: (est_mean, lo95, hi95) via bootstrap
    """
    m = np.isfinite(x)
    x = x[m]
    if w is not None:
        w = w[m]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    # point estimate
    if w is None:
        est = float(np.mean(x))
    else:
        est = float(np.sum(w*x)/np.sum(w))
    if x.size == 1:
        return (est, np.nan, np.nan)
    rng = np.random.default_rng(1729)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        if w is None:
            boots[i] = float(np.mean(xb))
        else:
            wb = w[idx]
            boots[i] = float(np.sum(wb*xb)/np.sum(wb))
    lo = float(np.percentile(boots, 100*alpha/2))
    hi = float(np.percentile(boots, 100*(1-alpha/2)))
    return (est, lo, hi)

def save_delta_curve(track: str, ks: List[int], d_mean: List[float], d_lo: List[float], d_hi: List[float],
                     ylabel: str, outfile: str) -> None:
    plt.figure()
    plt.plot(ks, d_mean)
    # CIバンド（利用可能な場合のみ）
    valid_band = all([not (np.isnan(a) or np.isnan(b)) for a,b in zip(d_lo, d_hi)])
    if valid_band:
        plt.fill_between(ks, d_lo, d_hi, alpha=0.3)
    plt.title(f"Δ curve vs Llama ({track})")
    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(outfile, dpi=160)
    plt.close()

# =========================
# LOAD
# =========================
ensure_dir(OUTDIR)

curves   = load_csv_must(CURVES_CSV)
pairwise = load_csv_must(PAIRWISE_CSV)

has_joined = False
dfj = None
w_series = None
if JOINED_CSV and os.path.isfile(JOINED_CSV):
    dfj = pd.read_csv(JOINED_CSV)
    # difficulty weight per problem
    w_series = (dfj[["problem_id","w"]]
                .drop_duplicates("problem_id")
                .set_index("problem_id")["w"])
    has_joined = True

# =========================
# PREP
# =========================
# 使えるトラック／k を決定
tracks_avail = [t for t in TRACKS if t in curves["track"].unique().tolist()]
ks_all = sorted(curves["k"].unique().tolist())
REP_KS = [k for k in [1,10,100] if k in ks_all]

# 入力バリデーション
for tr in tracks_avail:
    assert ((curves["track"]==tr) & (curves["model"]==REF_MODEL)).any(), f"curves: {tr}/{REF_MODEL} not found"
    assert ((curves["track"]==tr) & (curves["model"]==CMP_MODEL)).any(), f"curves: {tr}/{CMP_MODEL} not found"

# =========================
# BUILD Δ CURVES + TABLE
# =========================
table_rows = []

for tr in tracks_avail:
    cL = curves[(curves["track"]==tr) & (curves["model"]==REF_MODEL)].set_index("k")
    cD = curves[(curves["track"]==tr) & (curves["model"]==CMP_MODEL)].set_index("k")
    ks = sorted(set(cL.index).intersection(set(cD.index)))

    # ---- Δpass@k ----
    d_mean, d_lo, d_hi = [], [], []
    for k in ks:
        mpL = float(cL.loc[k, "mean_pass"])
        mpD = float(cD.loc[k, "mean_pass"])
        d_mean.append(mpD - mpL)

        if has_joined:
            # paired bootstrap over problems_llama
            dLj = dfj[(dfj["track"]==tr) & (dfj["model"]==REF_MODEL)].set_index("problem_id")
            dDj = dfj[(dfj["track"]==tr) & (dfj["model"]==CMP_MODEL)].set_index("problem_id")
            common = dLj.index.intersection(dDj.index)
            col = f"pass_at_{k}"
            if (col in dLj.columns) and (col in dDj.columns) and (len(common)>0):
                x = (dDj.loc[common, col] - dLj.loc[common, col]).to_numpy(float)
                est, lo, hi = paired_boot_ci(x, None, n_boot=N_BOOT, alpha=ALPHA)
                d_lo.append(lo); d_hi.append(hi)
            else:
                d_lo.append(np.nan); d_hi.append(np.nan)
        else:
            # 保守的CI（モデル別CIの差から構成）
            if {"mean_ci_lo","mean_ci_hi"}.issubset(set(cL.columns)) and {"mean_ci_lo","mean_ci_hi"}.issubset(set(cD.columns)):
                lo = float(cD.loc[k,"mean_ci_lo"] - cL.loc[k,"mean_ci_hi"])
                hi = float(cD.loc[k,"mean_ci_hi"] - cL.loc[k,"mean_ci_lo"])
                d_lo.append(lo); d_hi.append(hi)
            else:
                d_lo.append(np.nan); d_hi.append(np.nan)

    save_delta_curve(tr, ks, d_mean, d_lo, d_hi, ylabel="Δpass@k",
                     outfile=os.path.join(OUTDIR, f"fig8_10_delta_pass_{tr}.png"))

    # ---- ΔD-pass@k ----
    dD_mean, dD_lo, dD_hi = [], [], []
    for k in ks:
        dpL = float(cL.loc[k, "D_pass"])
        dpD = float(cD.loc[k, "D_pass"])
        dD_mean.append(dpD - dpL)

        if has_joined:
            dLj = dfj[(dfj["track"]==tr) & (dfj["model"]==REF_MODEL)].set_index("problem_id")
            dDj = dfj[(dfj["track"]==tr) & (dfj["model"]==CMP_MODEL)].set_index("problem_id")
            common = dLj.index.intersection(dDj.index)
            col = f"pass_at_{k}"
            if (col in dLj.columns) and (col in dDj.columns) and (len(common)>0):
                x = (dDj.loc[common, col] - dLj.loc[common, col]).to_numpy(float)
                w = w_series.loc[common].to_numpy(float) if w_series is not None else None
                est, lo, hi = paired_boot_ci(x, w, n_boot=N_BOOT, alpha=ALPHA)
                dD_lo.append(lo); dD_hi.append(hi)
            else:
                dD_lo.append(np.nan); dD_hi.append(np.nan)
        else:
            if {"D_ci_lo","D_ci_hi"}.issubset(set(cL.columns)) and {"D_ci_lo","D_ci_hi"}.issubset(set(cD.columns)):
                lo = float(cD.loc[k,"D_ci_lo"] - cL.loc[k,"D_ci_hi"])
                hi = float(cD.loc[k,"D_ci_hi"] - cL.loc[k,"D_ci_lo"])
                dD_lo.append(lo); dD_hi.append(hi)
            else:
                dD_lo.append(np.nan); dD_hi.append(np.nan)

    save_delta_curve(tr, ks, dD_mean, dD_lo, dD_hi, ylabel="ΔD-pass@k",
                     outfile=os.path.join(OUTDIR, f"fig8_11_delta_Dpass_{tr}.png"))

    # ---- Table 8.7 行（k=1/10/100 から存在するもの） ----
    for k in [kk for kk in [1,10,100] if kk in ks]:
        idx = ks.index(k)

        # 勝率・Holm補正p は pairwise CSV から（既に計算済み）
        metric = f"pass_at_{k}"
        rowp = pairwise[(pairwise["track"]==tr) & (pairwise["metric"]==metric) & (pairwise["cmp_model"]==CMP_MODEL)]
        wins   = int(rowp["wins"].iloc[0])   if len(rowp)>0 and "wins" in rowp.columns else np.nan
        ties   = int(rowp["ties"].iloc[0])   if len(rowp)>0 and "ties" in rowp.columns else np.nan
        losses = int(rowp["losses"].iloc[0]) if len(rowp)>0 and "losses" in rowp.columns else np.nan
        p_holm = float(rowp["sign_p_holm"].iloc[0]) if len(rowp)>0 and "sign_p_holm" in rowp.columns else np.nan

        table_rows.append({
            "track": tr, "k": k,
            "Δpass@k_mean": round(d_mean[idx], 4),
            "Δpass@k_CI_lo": (round(d_lo[idx], 4) if not np.isnan(d_lo[idx]) else np.nan),
            "Δpass@k_CI_hi": (round(d_hi[idx], 4) if not np.isnan(d_hi[idx]) else np.nan),
            "ΔD-pass@k_mean": round(dD_mean[idx], 4),
            "ΔD-pass@k_CI_lo": (round(dD_lo[idx], 4) if not np.isnan(dD_lo[idx]) else np.nan),
            "ΔD-pass@k_CI_hi": (round(dD_hi[idx], 4) if not np.isnan(dD_hi[idx]) else np.nan),
            "wins": wins, "ties": ties, "losses": losses, "sign_test_p_holm": p_holm
        })

# =========================
# SAVE TABLE
# =========================
table = pd.DataFrame(table_rows).sort_values(["track","k"])
table_path = os.path.join(OUTDIR, "table8_7_delta_summary.csv")
table.to_csv(table_path, index=False)

# =========================
# PRINT OUTPUT PATHS
# =========================
print("[OK] Generated files under:", OUTDIR)
for tr in tracks_avail:
    p1 = os.path.join(OUTDIR, f"fig8_10_delta_pass_{tr}.png")
    p2 = os.path.join(OUTDIR, f"fig8_11_delta_Dpass_{tr}.png")
    if os.path.isfile(p1): print(" -", p1)
    if os.path.isfile(p2): print(" -", p2)
print(" -", table_path)
