#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
図8.22〜図8.24／表8.12〜表8.14 を一括生成するスクリプト

【前提】
- 本スクリプトは “コード内設定を編集” して使う（引数なし）。
- 既存の集計CSV（model_compare_out/*.csv）を優先して利用。
- LOPOや代替重み比較は “問題別の per_problem CSV” から再集計する。

【出力】
- out_figs/fig8_22_weight_scatter.png
- out_figs/fig8_23_curves_mean_and_D.png
- out_figs/fig8_24_lopo_deltas.png
- out_tables/table8_12_rank_corr_weights.csv
- out_tables/table8_13_rep_points_bootstrap_CI.csv
- out_tables/table8_14_lopo_summary.csv

【想定ディレクトリ構造（例）】
./model_compare_out/
  ├─ leaderboard_by_model_track.csv
  ├─ overall_curves_mean_and_D.csv
  ├─ pairwise_vs_ref_sign_test.csv
  ├─ stratified_by_difficulty_quartiles.csv
  └─ (任意) joined_all_models.csv   ← あれば高速（なくてもOK）

＜問題別CSV（per_problem）＞
./Llama-3.1-8B-Instruct/results/per_problem_pass_at_k.csv
./deepseek-coder-7b-instruct-v1.5/results/per_problem_pass_at_k.csv
# さらに追加モデルがあれば MODEL_DIRS に追加

＜重みCSV（標準・代替）＞
# 標準重み（既存を想定。問題IDと w 列）
./difficulty_weights_A_baseline_only.csv
# 代替重み（任意。存在しない場合はスキップ）
./alt_weights_human_only.csv     （列: problem_id, w_human）
./alt_weights_avg_median.csv     （列: problem_id, w_avg）

【注意】
- matplotlib のみ使用（seaborn不使用）。
- 代替重みCSVが無い場合でも、“標準 vs フラット（=1.0）”の比較で図8.22・表8.12を生成。
- LOPOは Llama と DeepSeek の共通問題のみで実施。track ごとに k∈{1,10} で分布を出力。
"""

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional

# ========== 設定（必要に応じて編集） ==========
MODEL_DIRS = [
    "./Llama-3.1-8B-Instruct",
    "./deepseek-coder-7b-instruct-v1.5",
    # "./Qwen2.5-Coder-7B-Instruct",
]
PER_PROBLEM_FILENAME = "results/per_problem_pass_at_k.csv"
TRACKS = ["baseline", "CoT", "feedback"]
K_FOR_SCATTER = [1, 10, 100]          # 図8.22は k=1 を中心に。必要なら [1,10] などに変更可
K_FOR_LOPO = [1, 10]         # 図8.24・表8.14は @1/@10
REP_KS = [1, 10, 100]        # 表8.13の代表点（曲線CSVに存在するkに合わせる）
REF_MODEL = "Llama-3.1-8B-Instruct"
TARGET_MODEL = "deepseek-coder-7b-instruct-v1.5"

# 既存集計CSV（あれば図8.23・表8.13の作図・作表に使用）
MODEL_COMPARE_DIR = Path("./model_compare_out")

# 重み（標準＆代替）。存在しない場合は自動スキップ
STANDARD_WEIGHT_CSV = Path("./Llama-3.1-8B-Instruct/results/difficulty_weights_A_baseline_only.csv")  # 列: problem_id, w または w_p
ALT_WEIGHT_FILES = [
    # (path, weight_column_name)
    (Path("./alt_weights_human_only.csv"), "w_human"),
    (Path("./alt_weights_avg_median.csv"), "w_avg"),
]
# “フラット重み（=1.0）”は常に比較に含める（コード内で自動付与）

# 出力先
OUT_FIG_DIR = Path("./figs_8_9")
OUT_TBL_DIR = Path("./figs_8_9")

# ブートストラップ回数（図8.23は既存のCIを再描画するのみ）
N_BOOT = 1000

# 乱数
RNG = np.random.default_rng(1729)

# ========== ユーティリティ ==========
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in s)

def load_overall_curves(model_compare_dir: Path) -> Optional[pd.DataFrame]:
    p = model_compare_dir / "overall_curves_mean_and_D.csv"
    if p.is_file():
        return pd.read_csv(p)
    return None

def load_leaderboard(model_compare_dir: Path) -> Optional[pd.DataFrame]:
    p = model_compare_dir / "leaderboard_by_model_track.csv"
    if p.is_file():
        return pd.read_csv(p)
    return None

def load_joined_all(model_compare_dir: Path) -> Optional[pd.DataFrame]:
    p = model_compare_dir / "joined_all_models.csv"
    if p.is_file():
        return pd.read_csv(p)
    return None

def load_per_problem_from_models(model_dirs: List[str], per_problem_filename: str) -> pd.DataFrame:
    rows = []
    for d in model_dirs:
        csvp = Path(d) / per_problem_filename
        if not csvp.is_file():
            print(f"[WARN] per_problem missing: {csvp}")
            continue
        df = pd.read_csv(csvp)
        # 標準化：列名の揺れを吸収
        if "problem" in df.columns and "problem_id" not in df.columns:
            df = df.rename(columns={"problem":"problem_id"})
        df["model"] = Path(d).name
        rows.append(df)
    if not rows:
        raise FileNotFoundError("No per_problem CSVs found. Check MODEL_DIRS and PER_PROBLEM_FILENAME.")
    df = pd.concat(rows, ignore_index=True)
    return df

def detect_k_columns(df: pd.DataFrame) -> List[int]:
    ks = []
    for c in df.columns:
        if c.startswith("pass_at_"):
            tail = c.split("_")[-1]
            if tail.isdigit():
                ks.append(int(tail))
    ks = sorted(set(ks))
    return ks

def merge_weight(per_problem: pd.DataFrame, weight_csv: Optional[Path], out_col: str) -> pd.DataFrame:
    df = per_problem.copy()
    if weight_csv is None:
        # 何も与えられない場合 → out_col を 1.0 で作る
        df[out_col] = 1.0
        return df
    if not weight_csv.is_file():
        print(f"[WARN] weight not found: {weight_csv}. Skip this weight.")
        df[out_col] = np.nan  # 後でドロップ
        return df
    wdf = pd.read_csv(weight_csv)
    # 重み列名の吸収
    cand = [c for c in wdf.columns if c.lower() in ["w","w_p", out_col.lower()]]
    if not cand:
        raise ValueError(f"Weight CSV {weight_csv} must contain a weight column (w / w_p / {out_col}).")
    wcol = cand[0]
    wdf = wdf.rename(columns={wcol: out_col})
    m = df.merge(wdf[["problem_id", out_col]], on="problem_id", how="left")
    # フォールバック：欠損を 1.0 に
    m[out_col] = m[out_col].fillna(1.0)
    return m

def d_mean(values: np.ndarray, weights: np.ndarray) -> float:
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x = values[m]; w = weights[m]
    if x.size == 0:
        return np.nan
    return float(np.sum(w*x) / np.sum(w))

def kendall_tau(a: List[float], b: List[float]) -> float:
    # 簡易 Kendall τ（ties は近似処理）
    from math import comb
    n = len(a)
    if n < 2:
        return np.nan
    # 順位
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    conc, disc = 0, 0
    for i in range(n):
        for j in range(i+1, n):
            s1 = np.sign(ra[i]-ra[j])
            s2 = np.sign(rb[i]-rb[j])
            if s1*s2 > 0: conc += 1
            elif s1*s2 < 0: disc += 1
    denom = comb(n,2)
    if denom == 0:
        return np.nan
    return (conc - disc) / denom

def spearman_rho(a: List[float], b: List[float]) -> float:
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if len(ra) < 2:
        return np.nan
    return float(np.corrcoef(ra, rb)[0,1])

def common_problem_set(df: pd.DataFrame, model_a: str, model_b: str, track: str) -> pd.DataFrame:
    da = df[(df["model"]==model_a) & (df["track"]==track)]
    db = df[(df["model"]==model_b) & (df["track"]==track)]
    common = np.intersect1d(da["problem_id"].unique(), db["problem_id"].unique())
    m = df[df["problem_id"].isin(common)].copy()
    return m

# ========== D-pass 再集計（任意の重み列名で） ==========
def summarize_Dpass_by_weight(per_problem: pd.DataFrame, weight_col: str, k_list: List[int]) -> pd.DataFrame:
    rows = []
    for (model, track), g in per_problem.groupby(["model","track"], sort=False):
        w = g[weight_col].to_numpy()
        for k in k_list:
            col = f"pass_at_{k}"
            if col not in g.columns:
                continue
            x = g[col].to_numpy()
            rows.append({
                "model": model, "track": track, "k": k,
                "D_pass": d_mean(x, w)
            })
    return pd.DataFrame(rows)

# ========== 図8.22：重み差し替えの散布図と順位相関 ==========
def make_fig8_22_scatter_and_table(per_problem_base: pd.DataFrame,
                                   standard_weight_csv: Optional[Path],
                                   alt_weight_files: List[Tuple[Path,str]],
                                   out_fig: Path, out_tbl: Path):
    print("[INFO] Figure 8.22 / Table 8.12 ...")
    ensure_dir(out_fig.parent); ensure_dir(out_tbl.parent)

    # 標準重み
    df_std = merge_weight(per_problem_base, standard_weight_csv, out_col="w_std")
    # フラット重み
    df_flat = merge_weight(per_problem_base, None, out_col="w_flat")

    # 代替重み群（存在するもののみ）
    alt_defs = [("flat", "w_flat")]
    for p, col in alt_weight_files:
        tag = Path(p).stem
        try:
            df_alt = merge_weight(per_problem_base, p, out_col=col)
            alt_defs.append((tag, col))
        except Exception as e:
            print(f"[WARN] skip alt weight {p}: {e}")

    # k=1 を中心に集計（必要なら K_FOR_SCATTER を増やす）
    recs = []
    # 散布図は track ごとに 1 つの Figure（k=K_FOR_SCATTER の点をすべて打つ）
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(6*len(TRACKS), 6), squeeze=False)
    for ti, track in enumerate(TRACKS):
        ax = axes[0, ti]
        # D_pass (標準)
        d_std = summarize_Dpass_by_weight(df_std[df_std["track"]==track], "w_std", K_FOR_SCATTER)
        # 各代替に対して散布
        colors = ["C1","C2","C3","C4","C5"]  # matplotlib デフォルト。色指定はせず使用（可読性用）
        for ai, (atag, acol) in enumerate(alt_defs):
            d_alt = summarize_Dpass_by_weight(df_alt if atag!="flat" else df_flat, acol, K_FOR_SCATTER)
            merged = d_std.merge(d_alt, on=["model","track","k"], suffixes=("_std","_alt"))
            # 順位相関（モデル×k）
            for k in K_FOR_SCATTER:
                sub = merged[merged["k"]==k]
                # 2モデル以上ないと相関不可
                if sub["model"].nunique() >= 2:
                    tau = kendall_tau(sub["D_pass_std"].tolist(), sub["D_pass_alt"].tolist())
                    rho = spearman_rho(sub["D_pass_std"].tolist(), sub["D_pass_alt"].tolist())
                else:
                    tau = np.nan; rho = np.nan
                recs.append({
                    "track": track, "k": k, "alt_weight": atag,
                    "kendall_tau": tau, "spearman_rho": rho,
                    "n_models": int(sub["model"].nunique())
                })
            # 散布打ち
            ax.scatter(merged["D_pass_std"], merged["D_pass_alt"], label=f"{atag}", s=50)
        # y=x 線
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi])
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"Track={track} (D-pass, k={K_FOR_SCATTER})")
        ax.set_xlabel("Standard weight D-pass")
        ax.set_ylabel("Alt weight D-pass")
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)

    # 表8.12：順位相関（track×k×代替重み）
    pd.DataFrame(recs).to_csv(out_tbl, index=False)
    print(f"[OK] {out_fig}, {out_tbl}")

# ========== 図8.23：Mean/D-Mean 曲線（既存CSVの再描画） ==========
def make_fig8_23_curves(curves_csv: Path, out_fig: Path):
    print("[INFO] Figure 8.23 (curves) ...")
    ensure_dir(out_fig.parent)
    curves = pd.read_csv(curves_csv)
    # k 軸は curves の k（既存集計）
    # 行：track、列：2（上: mean, 下: D）
    tracks_present = [t for t in TRACKS if t in curves["track"].unique()]
    nrow = 2
    ncol = max(1, len(tracks_present))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol, 5*nrow), squeeze=False)
    for ci, track in enumerate(tracks_present):
        sub = curves[curves["track"]==track]
        for ri, (ykey, lo, hi, title) in enumerate([
            ("mean_pass", "mean_ci_lo", "mean_ci_hi", f"{track}: Mean pass@k"),
            ("D_pass", "D_ci_lo", "D_ci_hi", f"{track}: D-Mean pass@k"),
        ]):
            ax = axes[ri, ci]
            for model in sub["model"].unique():
                dfm = sub[sub["model"]==model].sort_values("k")
                ax.plot(dfm["k"], dfm[ykey], label=model)
                # CI帯（塗り）
                ax.fill_between(dfm["k"], dfm[lo], dfm[hi], alpha=0.2)
            ax.set_title(title); ax.set_xlabel("k"); ax.set_ylabel(ykey)
            ax.grid(True, linestyle="--", alpha=0.3)
            if ri==0:
                ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    print(f"[OK] {out_fig}")

# ========== 表8.13：代表点（k=1/10/100）の 95%CI（既存CSVから） ==========
def make_table8_13_rep_points(curves_csv: Path, rep_ks: List[int], out_tbl: Path):
    print("[INFO] Table 8.13 (rep points CI) ...")
    ensure_dir(out_tbl.parent)
    curves = pd.read_csv(curves_csv)
    rows = []
    for (track, model), g in curves.groupby(["track","model"], sort=False):
        for k in rep_ks:
            sub = g[g["k"]==k]
            if sub.empty:
                continue
            r = sub.iloc[0]
            rows.append({
                "track": track, "model": model, "k": k,
                "mean": r.get("mean_pass", np.nan),
                "mean_ci_lo": r.get("mean_ci_lo", np.nan),
                "mean_ci_hi": r.get("mean_ci_hi", np.nan),
                "D_mean": r.get("D_pass", np.nan),
                "D_ci_lo": r.get("D_ci_lo", np.nan),
                "D_ci_hi": r.get("D_ci_hi", np.nan),
                "n_problems": r.get("n_problems", np.nan),
            })
    pd.DataFrame(rows).sort_values(["track","model","k"]).to_csv(out_tbl, index=False)
    print(f"[OK] {out_tbl}")

# ========== 図8.24・表8.14：LOPO 解析 ==========
def make_fig8_24_and_table8_14_lopo(per_problem: pd.DataFrame, out_fig: Path, out_tbl: Path):
    print("[INFO] Figure 8.24 / Table 8.14 (LOPO) ...")
    ensure_dir(out_fig.parent); ensure_dir(out_tbl.parent)

    # DeepSeek と Llama の共通問題のみ対象
    lopo_summary = []
    # 3(track)×2(k) の箱ひげ（それぞれに LOPO の Δ分布）
    fig, axes = plt.subplots(len(TRACKS), len(K_FOR_LOPO), figsize=(5*len(K_FOR_LOPO), 4*len(TRACKS)), squeeze=False)

    for ti, track in enumerate(TRACKS):
        df_t = common_problem_set(per_problem, REF_MODEL, TARGET_MODEL, track)
        if df_t.empty:
            # 空なら空プロット（スキップ）
            for ki, k in enumerate(K_FOR_LOPO):
                ax = axes[ti, ki]; ax.set_axis_off()
            continue

        # 問題集合
        problems = sorted(df_t["problem_id"].unique())
        # 参照：全問題での Δ（確認用）
        def full_delta(k):
            col = f"pass_at_{k}"
            a = df_t[(df_t["model"]==TARGET_MODEL)].set_index("problem_id")[col]
            b = df_t[(df_t["model"]==REF_MODEL)].set_index("problem_id")[col]
            common = a.index.intersection(b.index)
            return float((a.loc[common] - b.loc[common]).mean()), int((a.loc[common] > b.loc[common]).sum()), int((a.loc[common] < b.loc[common]).sum())

        for ki, k in enumerate(K_FOR_LOPO):
            col = f"pass_at_{k}"
            deltas = []
            wins_in_lopo = []
            for leave_out in problems:
                keep = [p for p in problems if p != leave_out]
                a = df_t[(df_t["model"]==TARGET_MODEL) & (df_t["problem_id"].isin(keep))].set_index("problem_id")[col]
                b = df_t[(df_t["model"]==REF_MODEL) & (df_t["problem_id"].isin(keep))].set_index("problem_id")[col]
                common = a.index.intersection(b.index)
                if len(common)==0:
                    continue
                d = float((a.loc[common] - b.loc[common]).mean())
                deltas.append(d)
                wins_in_lopo.append(int((a.loc[common] > b.loc[common]).sum()))
            if len(deltas)==0:
                # 空プロット
                ax = axes[ti, ki]
                ax.set_axis_off()
                continue

            # 箱ひげ
            ax = axes[ti, ki]
            ax.boxplot(deltas, vert=True, labels=[f"{track}@{k}"])
            ax.axhline(0.0)
            ax.set_ylabel("LOPO mean Δ (DeepSeek−Llama)")
            ax.set_title(f"Track={track}, k={k} (LOPO Δ)")

            # 表要約
            full_d, full_w, full_l = full_delta(k)
            dir_retention = float((np.array(deltas) > 0).mean())  # Δ>0 の割合
            lopo_summary.append({
                "track": track, "k": k,
                "full_mean_delta": full_d,
                "full_wins": full_w,
                "dir_retention_rate": dir_retention,         # 効果方向（正）の維持率
                "lopo_mean_delta_min": float(np.min(deltas)),
                "lopo_mean_delta_max": float(np.max(deltas)),
                "wins_min": int(np.min(wins_in_lopo)),
                "wins_max": int(np.max(wins_in_lopo)),
                "n_lopo": len(deltas),
                "n_common_problems": len(problems)
            })

    fig.tight_layout()
    fig.savefig(out_fig, dpi=150)
    plt.close(fig)
    pd.DataFrame(lopo_summary).sort_values(["track","k"]).to_csv(out_tbl, index=False)
    print(f"[OK] {out_fig}, {out_tbl}")

# ========== メイン ==========
def main():
    ensure_dir(OUT_FIG_DIR); ensure_dir(OUT_TBL_DIR)

    # 1) 既存の curves を読み込んで 図8.23・表8.13 を作成
    curves_df = load_overall_curves(MODEL_COMPARE_DIR)
    if curves_df is None:
        raise FileNotFoundError("model_compare_out/overall_curves_mean_and_D.csv が見つかりません。先に集計を作成してください。")
    make_fig8_23_curves(MODEL_COMPARE_DIR/"overall_curves_mean_and_D.csv", OUT_FIG_DIR/"fig8_23_curves_mean_and_D.png")
    make_table8_13_rep_points(MODEL_COMPARE_DIR/"overall_curves_mean_and_D.csv", REP_KS, OUT_TBL_DIR/"table8_13_rep_points_bootstrap_CI.csv")

    # 2) 問題別 per_problem をロード（joined_all があればそれを優先）
    joined = load_joined_all(MODEL_COMPARE_DIR)
    if joined is not None and all(c in joined.columns for c in ["model","track","problem_id"]):
        per_problem = joined.copy()
    else:
        per_problem = load_per_problem_from_models(MODEL_DIRS, PER_PROBLEM_FILENAME)

    # track フィルタ
    per_problem = per_problem[per_problem["track"].isin(TRACKS)].copy()

    # 3) 図8.22（重み差し替え散布図）＋ 表8.12（順位相関）
    make_fig8_22_scatter_and_table(
        per_problem_base=per_problem,
        standard_weight_csv=STANDARD_WEIGHT_CSV if STANDARD_WEIGHT_CSV.is_file() else None,
        alt_weight_files=[x for x in ALT_WEIGHT_FILES if x[0].is_file()],
        out_fig=OUT_FIG_DIR/"fig8_22_weight_scatter.png",
        out_tbl=OUT_TBL_DIR/"table8_12_rank_corr_weights.csv"
    )

    # 4) 図8.24（LOPO）＋ 表8.14（要約）
    make_fig8_24_and_table8_14_lopo(
        per_problem=per_problem,
        out_fig=OUT_FIG_DIR/"fig8_24_lopo_deltas.png",
        out_tbl=OUT_TBL_DIR/"table8_14_lopo_summary.csv"
    )

    print("\n[Done] 図8.22〜8.24・表8.12〜8.14 を出力しました。")
    print(f"  Figures: {OUT_FIG_DIR}")
    print(f"  Tables : {OUT_TBL_DIR}")

if __name__ == "__main__":
    main()
