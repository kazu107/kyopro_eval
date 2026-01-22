#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
図8.6〜図8.9 と 表8.6 を一括生成するスクリプト
- 入力: make_model_compare.py が出力した joined_all_models.csv
- 前提: 列に ['model','track','problem_id','w','pass_at_1','pass_at_10'] 等があること
- 出力: ./figs/ 以下に PNG と CSV

可視化ポリシー:
- seaborn不使用（matplotlibのみ）
- 1図＝1ファイル（サブプロット禁止）
- 明示的な色指定は行わない（デフォルト配色）

図表:
- 図8.6: pass@1（問題別）のヒストグラム（trackごと、モデル2種を重ね）
- 図8.7: Δ分布（DeepSeek−Llama）のヒストグラム（k=1/10 × track）
- 図8.8: 散布図（pass@1: Llama vs DeepSeek、点系列= w 四分位）
- 図8.9: ヒートマップ（pass@10、行=問題(重み降順)、列=(track,model)）
- 表8.6: 勝率・引分・敗北と符号検定（Holm補正付き）
"""

from pathlib import Path
from typing import List
import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG（ここだけ編集）
# =========================
# make_model_compare.py の出力ディレクトリ（相対 or 絶対）
INPUT_DIR = "./model_compare_out"
JOINED_FILENAME = "joined_all_models.csv"

# 比較する k
K_HIST = 1                # 図8.6は pass@1
K_DELTA_LIST = [1, 10]    # 図8.7は @1 と @10
K_HEAT = 10               # 図8.9は pass@10

# 出力先
OUTDIR = "./figs_8_3"

# =========================
# ユーティリティ
# =========================
def sign_test_p(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sided exact sign test on paired differences (ignoring ties)."""
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

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# =========================
# メイン
# =========================
def main():
    inpath = Path(INPUT_DIR) / JOINED_FILENAME
    if not inpath.is_file():
        raise FileNotFoundError(f"Not found: {inpath}")

    outdir = Path(OUTDIR); ensure_dir(outdir)

    df = pd.read_csv(inpath)

    # 必須列チェック
    need = {"model","track","problem_id","w"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {inpath}: {missing}")

    # モデル検出（2モデル比較前提）
    models = sorted(df["model"].unique().tolist())
    if len(models) < 2:
        raise ValueError("At least two models are required in joined_all_models.csv")
    # Llama と DeepSeek の順に揃えられる場合はそれに合わせる
    preferred_order = []
    for cand in ["Llama","deepseek"]:
        if cand in models:
            preferred_order.append(cand)
    others = [m for m in models if m not in preferred_order]
    order = preferred_order + others
    model_a, model_b = order[0], order[1]

    # 対象トラック
    tracks = [t for t in ["baseline","CoT","feedback"] if t in df["track"].unique()]

    # 重みと四分位
    w_uni = (df[["problem_id","w"]].drop_duplicates("problem_id")
             .set_index("problem_id")["w"])
    qs = np.quantile(w_uni.values, [0.25,0.5,0.75])
    def w_bin(val: float) -> str:
        if val <= qs[0]: return "Q1 (easy)"
        elif val <= qs[1]: return "Q2"
        elif val <= qs[2]: return "Q3"
        else: return "Q4 (hard)"
    w_bins = w_uni.apply(w_bin)

    # ==================== 図8.6（pass@1, ヒスト） ====================
    col_hist = f"pass_at_{K_HIST}"
    if col_hist in df.columns:
        for tr in tracks:
            sub = df[df["track"]==tr]
            vals = {}
            for m in [model_a, model_b]:
                v = (sub[sub["model"]==m]
                     .drop_duplicates("problem_id")[col_hist]
                     .dropna().to_numpy())
                if v.size:
                    vals[m] = v
            if len(vals) < 2:
                continue

            plt.figure()
            bins = np.linspace(0.0, 1.0, 11)
            for m, v in vals.items():
                plt.hist(v, bins=bins, alpha=0.5, label=m, density=False)
            plt.title(f"({tr})  Histogram of per-problem pass@{K_HIST}")
            plt.xlabel(f"pass@{K_HIST} (per problem)")
            plt.ylabel("Count of problems_llama")
            plt.legend()
            plt.savefig(outdir / f"fig_8_6_hist_pass{K_HIST}_{tr}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # ==================== 図8.7（Δヒスト：@1,@10 × track） ====================
    for k in K_DELTA_LIST:
        col = f"pass_at_{k}"
        if col not in df.columns:
            continue
        for tr in tracks:
            sub = df[df["track"]==tr]
            piv = (sub.pivot_table(index="problem_id", columns="model", values=col, aggfunc="mean")
                     .dropna(subset=[model_a, model_b]))
            if piv.empty:
                continue
            delta = piv[model_b] - piv[model_a]
            wins = int((delta>0).sum()); ties = int((delta==0).sum()); losses = int((delta<0).sum())
            p = sign_test_p(piv[model_b].to_numpy(), piv[model_a].to_numpy())

            plt.figure()
            plt.hist(delta.to_numpy(), bins=21, alpha=0.9)
            plt.title(f"(k={k}, {tr})  Δ = Llama - deepseek | W/T/L={wins}/{ties}/{losses}, p={p:.3f}")
            plt.xlabel(f"Δ pass@{k} per problem")
            plt.ylabel("Count of problems_llama")
            plt.savefig(outdir / f"fig_8_7_delta_pass{k}_{tr}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # ==================== 図8.8（散布図：pass@1, 色=四分位） ====================
    col = "pass_at_1"
    if col in df.columns:
        for tr in tracks:
            sub = df[df["track"]==tr]
            piv = (sub.pivot_table(index="problem_id", columns="model", values=col, aggfunc="mean")
                     .join(w_bins.rename("w_bin"), how="left")
                     .dropna(subset=[model_a, model_b]))
            if piv.empty:
                continue

            plt.figure()
            for qb in ["Q1 (easy)","Q2","Q3","Q4 (hard)"]:
                ss = piv[piv["w_bin"]==qb]
                if ss.empty:
                    continue
                plt.scatter(ss[model_a].to_numpy(), ss[model_b].to_numpy(), alpha=0.85, label=qb)
            mn = float(min(piv[model_a].min(), piv[model_b].min(), 0.0))
            mx = float(max(piv[model_a].max(), piv[model_b].max(), 1.0))
            plt.plot([mn, mx], [mn, mx])
            plt.xlim(mn, mx); plt.ylim(mn, mx)
            plt.xlabel(f"{model_a}  pass@1")
            plt.ylabel(f"{model_b}  pass@1")
            plt.title(f"({tr})  Per-problem scatter @1 by w quartile")
            plt.legend()
            plt.savefig(outdir / f"fig_8_8_scatter_pass1_{tr}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # ==================== 図8.9（ヒートマップ：pass@10） ====================
    colh = f"pass_at_{K_HEAT}"
    if colh in df.columns:
        order = w_uni.sort_values(ascending=False).index.tolist()
        mat = (df.pivot_table(index="problem_id", columns=["track","model"], values=colh, aggfunc="mean")
                 .reindex(order))
        mat = mat.dropna(axis=1, how="all")
        if mat.shape[0] > 0 and mat.shape[1] > 0:
            plt.figure(figsize=(8,10))
            plt.imshow(mat.to_numpy(), aspect="auto", interpolation="nearest")
            plt.title(f"Heatmap of pass@{K_HEAT} (rows: problems_llama by descending w)")
            plt.xlabel("Columns = (track, model)")
            plt.ylabel("Problems (descending w)")
            nrows = mat.shape[0]
            yticks_idx = list(range(0, nrows, max(1, nrows//20)))
            plt.yticks(yticks_idx, [mat.index[i] for i in yticks_idx])
            xticks_idx = list(range(mat.shape[1]))
            plt.xticks(xticks_idx, [f"{a}\n{b}" for a,b in mat.columns], rotation=90)
            plt.tight_layout()
            plt.savefig(outdir / f"fig_8_9_heatmap_pass{K_HEAT}.png", dpi=200, bbox_inches="tight")
            plt.close()

    # ==================== 表8.6（W/T/L と符号検定 + Holm） ====================
    rows = []
    for tr in tracks:
        sub = df[df["track"]==tr]
        for k in [1, 10]:
            colk = f"pass_at_{k}"
            if colk not in sub.columns:
                continue
            piv = (sub.pivot_table(index="problem_id", columns="model", values=colk, aggfunc="mean")
                     .dropna(subset=[model_a, model_b]))
            if piv.empty:
                continue
            a = piv[model_a].to_numpy()
            b = piv[model_b].to_numpy()
            d = b - a
            wins = int((d>0).sum()); ties = int((d==0).sum()); losses = int((d<0).sum())
            p = sign_test_p(b, a)
            rows.append({
                "track": tr, "metric": f"pass@{k}",
                "model_ref": model_a, "model_cmp": model_b,
                "n": int(len(d)),
                "wins": wins, "ties": ties, "losses": losses,
                "mean_diff": float(np.nanmean(d)),
                "p_sign": p
            })
    tbl = pd.DataFrame(rows)
    if not tbl.empty:
        tbl["p_holm"] = np.nan
        for (tr, met), sub in tbl.groupby(["track","metric"]):
            idx = sub.index.tolist()
            adj = holm_bonferroni(sub["p_sign"].fillna(1.0).tolist())
            tbl.loc[idx, "p_holm"] = adj
        tbl.to_csv(outdir / "table_8_6.csv", index=False)

    print("Done. Outputs ->", outdir.resolve())

if __name__ == "__main__":
    main()
