#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
図8.12–8.13（Δ分布のヒストグラム：非加重 / D加重）と
表8.8–8.9（問題単位の勝率・有意差：pass@k / D-pass@k）を一括で作成するスクリプト。

前提（make_model_compare.py の出力）:
- model_compare_out/joined_all_models.csv
    columns例: problem_id, track, model, pass_at_1..pass_at_100, w, auc_logk, k50, k90, ...
- model_compare_out/pairwise_vs_ref_sign_test.csv
    columns: track, metric, ref_model, cmp_model, mean_diff, D_mean_diff, wins, ties, losses, sign_p, sign_p_holm, n_common

要点:
- 図8.12: Δp = pass@k(DeepSeek) - pass@k(Llama) の問題別分布（非加重ヒストグラム）
- 図8.13: Δp の D加重ヒストグラム（weights=w）
- 表8.8: pass@k の問題単位 Sign test（Holm補正）+勝率（pairwise CSV から整形 or 自動計算）
- 表8.9: D-pass@k 用（D_mean_diff を併記）。Sign test は問題単位Δの符号に基づく（重みは検定に使わない）

※可視化は matplotlib を使用（サブプロットは作らず、1図=1プロットで複数ファイル保存）。
"""

# ====== CONFIG ======
BASE_DIR = "./model_compare_out"      # make_model_compare.py の出力ディレクトリ
OUT_DIR  = "./figs_8_5"      # 図表の出力先
REF_MODEL = "Llama-3.1-8B-Instruct"   # 基準モデル
TRACKS = ["baseline", "CoT", "feedback"]
K_LIST_FOR_HIST = [1, 10]             # 図8.12–8.13で出す k
K_LIST_FOR_TABLES = [1, 10, 100]      # 表8.8–8.9で出す k
HIST_BINS = 21                        # Δのヒストグラムbin数（-1.0..+1.0想定）
RANDOM_SEED = 1729                    # ブートストラップ等で使う場合に備えて
# =====================

import os
import math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

np.random.seed(RANDOM_SEED)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def sign_test_p(a: np.ndarray, b: np.ndarray) -> float:
    """
    Two-sided exact sign test (binomial). Tie（等しい）は除外。
    戻り値: p-value（[0,1]）
    """
    x = a - b
    x = x[np.isfinite(x)]
    pos = int((x > 0).sum())
    neg = int((x < 0).sum())
    n = pos + neg
    if n == 0:
        return np.nan
    from math import comb
    tail = min(pos, neg)
    p = sum(comb(n, k) for k in range(0, tail+1)) / (2**n)
    p = 2*p
    return float(min(1.0, p))

def holm_bonferroni(pvals: List[float]) -> List[float]:
    """
    Holm (1979) step-down method for multiple testing correction.
    """
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

def load_joined(base_dir: str) -> pd.DataFrame:
    path = Path(base_dir) / "joined_all_models.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing: {path}")
    df = pd.read_csv(path)
    # safety
    if "problem_id" not in df.columns and "problem" in df.columns:
        df = df.rename(columns={"problem":"problem_id"})
    if "w" not in df.columns:
        df["w"] = 1.0
    return df

def load_pairwise_or_none(base_dir: str) -> pd.DataFrame:
    path = Path(base_dir) / "pairwise_vs_ref_sign_test.csv"
    if not path.is_file():
        return None
    return pd.read_csv(path)

def list_cmp_models(df: pd.DataFrame, ref_model: str) -> List[str]:
    models = sorted(df["model"].unique().tolist())
    return [m for m in models if m != ref_model]

def per_problem_delta(df_joined: pd.DataFrame, track: str, k: int, ref_model: str, cmp_model: str) -> pd.DataFrame:
    """
    戻り: DataFrame[problem_id, delta, w]  （delta = cmp - ref ）
    """
    col = f"pass_at_{k}"
    sub = df_joined[df_joined["track"]==track]
    a = sub[sub["model"]==ref_model][["problem_id", col, "w"]].rename(columns={col:"ref"}).set_index("problem_id")
    b = sub[sub["model"]==cmp_model][["problem_id", col, "w"]].rename(columns={col:"cmp"}).set_index("problem_id")

    common = a.index.intersection(b.index)
    if len(common)==0:
        return pd.DataFrame(columns=["problem_id","delta","w"])

    a = a.loc[common]
    b = b.loc[common]
    # ここでの w は joined 側の w を採用（同一の共通重みであるはず）
    # 念のため平均を取り整合
    w = pd.DataFrame({"w": (a["w"].values + b["w"].values)/2.0}, index=common)

    delta = b["cmp"] - a["ref"]
    out = pd.DataFrame({"problem_id": common, "delta": delta.values, "w": w["w"].values})
    out = out.reset_index(drop=True)
    return out

def draw_hist_unweighted(deltas: np.ndarray, title: str, save_path: Path, bins: int = HIST_BINS):
    """
    図8.12 用：非加重ヒストグラム（Δpの分布）。1図=1プロットで保存。
    """
    fig = plt.figure(figsize=(6,4), dpi=150)
    plt.hist(deltas, bins=bins, range=(-1.0, 1.0))
    plt.axvline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Δ (cmp − ref)")
    plt.ylabel("Count (problems_llama)")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def draw_hist_weighted(deltas: np.ndarray, weights: np.ndarray, title: str, save_path: Path, bins: int = HIST_BINS):
    """
    図8.13 用：D加重ヒストグラム（weights=w）。1図=1プロットで保存。
    """
    fig = plt.figure(figsize=(6,4), dpi=150)
    plt.hist(deltas, bins=bins, range=(-1.0, 1.0), weights=weights)
    plt.axvline(0.0, linestyle="--")
    plt.title(title)
    plt.xlabel("Δ (cmp − ref)")
    plt.ylabel("Weighted count (by w)")
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

def make_tables_from_pairwise(df_pairwise: pd.DataFrame,
                              k_list: List[int],
                              tracks: List[str],
                              ref_model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    pairwise_vs_ref_sign_test.csv を整形して、
    表8.8（pass@k）と表8.9（D-pass@k）を返す。
    """
    rows_pass = []
    rows_dpass = []
    for tr in tracks:
        for k in k_list:
            metric = f"pass_at_{k}"
            sub = df_pairwise[(df_pairwise["track"]==tr) & (df_pairwise["metric"]==metric)]
            if sub.empty:
                continue
            for _, r in sub.iterrows():
                rows_pass.append({
                    "track": tr,
                    "k": k,
                    "ref_model": r["ref_model"],
                    "cmp_model": r["cmp_model"],
                    "n_common": int(r["n_common"]),
                    "wins": int(r["wins"]),
                    "ties": int(r["ties"]),
                    "losses": int(r["losses"]),
                    "mean_diff": float(r["mean_diff"]),
                    "p_sign_holm": float(r.get("sign_p_holm", np.nan))
                })
                rows_dpass.append({
                    "track": tr,
                    "k": k,
                    "ref_model": r["ref_model"],
                    "cmp_model": r["cmp_model"],
                    "n_common": int(r["n_common"]),
                    "wins": int(r["wins"]),
                    "ties": int(r["ties"]),
                    "losses": int(r["losses"]),
                    "D_mean_diff": float(r["D_mean_diff"]),
                    "p_sign_holm": float(r.get("sign_p_holm", np.nan))
                })
    table_pass = pd.DataFrame(rows_pass).sort_values(["track","k","cmp_model"])
    table_dpass = pd.DataFrame(rows_dpass).sort_values(["track","k","cmp_model"])
    return table_pass, table_dpass

def compute_pairwise_from_joined(df_joined: pd.DataFrame,
                                 k_list: List[int],
                                 tracks: List[str],
                                 ref_model: str) -> pd.DataFrame:
    """
    pairwise_vs_ref_sign_test.csv が無い場合のフォールバック:
    joined_all_models.csv からペアワイズ統計（wins/ties/losses, mean_diff, D_mean_diff, sign_p, Holm）を作る。
    """
    rows = []
    models = sorted(df_joined["model"].unique())
    cmp_models = [m for m in models if m != ref_model]
    for tr in tracks:
        for k in k_list:
            for m in cmp_models:
                df_delta = per_problem_delta(df_joined, tr, k, ref_model, m)
                if df_delta.empty:
                    continue
                d = df_delta["delta"].to_numpy()
                w = df_delta["w"].to_numpy()
                wins = int((d>0).sum())
                ties = int((d==0).sum())
                losses = int((d<0).sum())
                n_common = int((d!=0).sum() + ties)
                mean_diff = float(np.nanmean(d)) if len(d)>0 else np.nan
                D_mean_diff = float(np.nansum(w*d)/np.nansum(w)) if np.nansum(w)>0 else np.nan
                # sign test p
                # 注意: sign test は重みを使わず、Δの符号のみ
                p = sign_test_p(d, np.zeros_like(d))
                rows.append({
                    "track": tr, "metric": f"pass_at_{k}",
                    "ref_model": ref_model, "cmp_model": m,
                    "n_common": n_common, "wins": wins, "ties": ties, "losses": losses,
                    "mean_diff": mean_diff, "D_mean_diff": D_mean_diff, "sign_p": p
                })
    pairwise = pd.DataFrame(rows)
    # Holm 補正（track×metric ごとに）
    if not pairwise.empty:
        def apply_holm(g):
            adj = holm_bonferroni(g["sign_p"].fillna(1.0).tolist())
            g = g.copy()
            g["sign_p_holm"] = adj
            return g
        pairwise = pairwise.groupby(["track","metric"], group_keys=False).apply(apply_holm)
    return pairwise

def main():
    outdir = Path(OUT_DIR); ensure_dir(outdir)
    joined = load_joined(BASE_DIR)
    pairwise = load_pairwise_or_none(BASE_DIR)
    if pairwise is None:
        pairwise = compute_pairwise_from_joined(joined, K_LIST_FOR_TABLES, TRACKS, REF_MODEL)
        # 参考として保存
        pairwise_out = outdir / "pairwise_vs_ref_sign_test__recomputed.csv"
        pairwise.to_csv(pairwise_out, index=False)
        print(f"[INFO] pairwise (recomputed) -> {pairwise_out}")
    else:
        print("[INFO] loaded pairwise_vs_ref_sign_test.csv")

    # --- 表8.8（pass@k）・表8.9（D-pass@k） ---
    table_pass, table_dpass = make_tables_from_pairwise(pairwise, K_LIST_FOR_TABLES, TRACKS, REF_MODEL)
    t88 = outdir / "table_8_8__pass_at_k__sign_test.csv"
    t89 = outdir / "table_8_9__D_pass_at_k__sign_test.csv"
    table_pass.to_csv(t88, index=False)
    table_dpass.to_csv(t89, index=False)
    print(f"[OK] wrote {t88}")
    print(f"[OK] wrote {t89}")

    # --- 図8.12（非加重Δヒスト）・図8.13（D加重Δヒスト） ---
    # cmp モデル列（REF 以外すべて）
    cmp_models = list_cmp_models(joined, REF_MODEL)
    if len(cmp_models)==0:
        print("[WARN] no cmp_models found.")
    for tr in TRACKS:
        for k in K_LIST_FOR_HIST:
            for cmp_model in cmp_models:
                df_delta = per_problem_delta(joined, tr, k, REF_MODEL, cmp_model)
                if df_delta.empty:
                    print(f"[WARN] no common problems_llama for ({tr}, k={k}, cmp={cmp_model})")
                    continue
                d = df_delta["delta"].to_numpy()
                w = df_delta["w"].to_numpy()

                # 非加重ヒスト（図8.12）
                title = f"Δ distribution (unweighted):{tr},k={k},deepseek vs Llama"
                save = outdir / f"fig_8_12__delta_hist__{tr}__k{k}__{cmp_model.replace('/','-')}.png"
                draw_hist_unweighted(d, title, save)

                # D加重ヒスト（図8.13）
                title_w = f"Δ distribution (weighted by w):{tr},k={k},deepseek vs Llama"
                save_w = outdir / f"fig_8_13__delta_hist_weighted__{tr}__k{k}__{cmp_model.replace('/','-')}.png"
                draw_hist_weighted(d, w, title_w, save_w)

    # 簡単な README を出力
    readme = outdir / "README_fig8_12-13_table8_8-9.txt"
    readme.write_text(
        "生成物:\n"
        f"- {t88.name}: 表8.8（pass@k、勝率とHolm補正p）\n"
        f"- {t89.name}: 表8.9（D-pass@k、D_mean_diffとHolm補正p）\n"
        "- fig_8_12__*.png: 図8.12（Δの非加重ヒスト、track×k×cmp）\n"
        "- fig_8_13__*.png: 図8.13（ΔのD加重ヒスト、track×k×cmp）\n"
        "\n注意:\n"
        "- Sign test は問題単位Δの符号のみを用いており、D-pass の表でも検定自体は非加重です（一般的な手法）。\n"
        "- D_mean_diff は重み付き平均差で、実質的効果量の参考値として併記します。\n",
        encoding="utf-8"
    )
    print(f"[OK] wrote {readme}")

if __name__ == "__main__":
    main()
