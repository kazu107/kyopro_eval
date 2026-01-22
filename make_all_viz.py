#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# CONFIG (edit here)
# =========================
# Model result folders (each must contain results/per_problem_pass_at_k.csv)
MODEL_DIRS: List[str] = [
    "./Llama-3.1-8B-Instruct",   # ← ここを実際のフォルダに合わせて編集
    # "./Qwen2.5-Coder-7B-Instruct",
    # "./deepseek-coder-7b-instruct-v1.5",
]
PER_PROBLEM_FILENAME = "results/per_problem_pass_at_k.csv"  # 相対パス
TRACKS = ["baseline", "CoT", "feedback"]                    # 存在しない track は自動スキップ

# Difficulty files（すでに作成済みのものを指定）
DIFF_WEIGHTS_CSV = r"C:\Users\kazuu\PycharmProjects\atc\Llama-3.1-8B-Instruct\results\difficulty_weights_A_baseline_only.csv"      # columns: problem_id,w_p
DIFF_AXES_CSV    = r"C:\Users\kazuu\PycharmProjects\atc\Llama-3.1-8B-Instruct\results\difficulty_metrics_all_axes_baseline_only.csv"  # columns: problem_id,z_B,z_T,z_H,z_WA

# Outputs
OUTPUT_DIR = "viz_out"   # ここにPNG/SVGと中間CSVを保存します
SAVE_SVG = True          # TrueならSVGも保存（ベクタ）

# Radar metrics（性能レーダーに載せる列名）
RADAR_METRICS = ["pass_at_1","pass_at_10","pass_at_100","auc_logk","k50","k90"]
# Heatmapで使うk
HEATMAP_K = 10
# Per-problem pass@k heatmap を書く対象モデル（リスト先頭を既定）
HEATMAP_DETAILED_MODEL = None  # None => MODEL_DIRS[0]の名前に自動設定

# Random seed for any randomized step
RANDOM_SEED = 1729
np.random.seed(RANDOM_SEED)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in s)

def minmax_df(df: pd.DataFrame, invert_cols=None) -> pd.DataFrame:
    X = df.copy()
    invert_cols = set(invert_cols or [])
    for c in X.columns:
        if c in invert_cols:
            X[c] = 1.0 / (1.0 + X[c])  # e.g., smaller-is-better -> larger-is-better
    for c in X.columns:
        col = X[c].astype(float)
        lo, hi = col.min(), col.max()
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            X[c] = (col - lo) / (hi - lo)
        else:
            X[c] = 0.5  # constant => neutral
    return X

def radar_plot(values: pd.Series, title: str, outprefix: Path):
    labels = values.index.tolist()
    vals = values.values.astype(float).tolist()
    vals += vals[:1]  # close the loop
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_title(title)
    plt.tight_layout()

    png = outprefix.with_suffix(".png")
    fig.savefig(png, dpi=200)
    if SAVE_SVG:
        svg = outprefix.with_suffix(".svg")
        fig.savefig(svg)
    plt.close(fig)

def pca_2d(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2, :]            # (2,d)
    coords = Xc @ comps.T        # (n,2)
    ev = (S**2) / (X.shape[0]-1)
    evr = ev / ev.sum()
    return coords, comps, evr[:2]

def try_tsne_2d(X: np.ndarray, perplexity=6, n_iter=2000, random_state=0):
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", random_state=random_state, n_iter=n_iter)
        Y = tsne.fit_transform(X)
        return Y
    except Exception as e:
        print(f"[INFO] t-SNE skipped (scikit-learn not available or error: {e})")
        return None

def save_scatter_with_labels(df2: pd.DataFrame, title: str, outprefix: Path):
    fig, ax = plt.subplots()
    ax.scatter(df2.iloc[:,0], df2.iloc[:,1])
    for pid, (x,y) in df2.iterrows():
        ax.text(x, y, pid, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(df2.columns[0]); ax.set_ylabel(df2.columns[1])
    plt.tight_layout()
    png = outprefix.with_suffix(".png")
    fig.savefig(png, dpi=200)
    if SAVE_SVG:
        svg = outprefix.with_suffix(".svg")
        fig.savefig(svg)
    plt.close(fig)

def heatmap(values: np.ndarray, row_labels, col_labels, title: str, outprefix: Path, xticks=None, xtick_labels=None):
    fig, ax = plt.subplots()
    im = ax.imshow(values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    if xticks is not None and xtick_labels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    fig.colorbar(im, ax=ax, label="value")
    ax.set_title(title)
    plt.tight_layout()
    png = outprefix.with_suffix(".png")
    fig.savefig(png, dpi=200)
    if SAVE_SVG:
        svg = outprefix.with_suffix(".svg")
        fig.savefig(svg)
    plt.close(fig)

def load_all_per_problem(model_dirs: List[str], per_problem_filename: str) -> pd.DataFrame:
    rows = []
    for mdir in model_dirs:
        mdir = Path(mdir)
        csvp = mdir / per_problem_filename
        if not csvp.is_file():
            print(f"[WARN] missing: {csvp}")
            continue
        df = pd.read_csv(csvp)
        df["model"] = mdir.name
        rows.append(df)
    if not rows:
        raise RuntimeError("No per_problem CSVs found. Check MODEL_DIRS and PER_PROBLEM_FILENAME.")
    df = pd.concat(rows, ignore_index=True)
    if "problem_id" not in df.columns and "problem" in df.columns:
        df = df.rename(columns={"problem":"problem_id"})
    return df

def main():
    outdir = Path(OUTPUT_DIR)
    ensure_dir(outdir)

    per_problem = load_all_per_problem(MODEL_DIRS, PER_PROBLEM_FILENAME)
    per_problem = per_problem[per_problem["track"].isin(TRACKS)].copy()

    if os.path.isfile(DIFF_WEIGHTS_CSV):
        weights = pd.read_csv(DIFF_WEIGHTS_CSV)
    else:
        print(f"[WARN] weights CSV not found: {DIFF_WEIGHTS_CSV}. Using w_p=1.")
        weights = pd.DataFrame({"problem_id": per_problem["problem_id"].unique(), "w_p": 1.0})
    if os.path.isfile(DIFF_AXES_CSV):
        axes = pd.read_csv(DIFF_AXES_CSV)
    else:
        print(f"[WARN] axes CSV not found: {DIFF_AXES_CSV}. z_* columns unavailable.")
        axes = pd.DataFrame({"problem_id": per_problem["problem_id"].unique()})
    per_problem = per_problem.merge(weights, on="problem_id", how="left").merge(
        axes[["problem_id","z_B","z_T","z_H","z_WA"]] if set(["z_B","z_T","z_H","z_WA"]).issubset(axes.columns) else axes[["problem_id"]],
        on="problem_id", how="left"
    )

    # Save joined per-problem
    joined_csv = outdir / "joined_per_problem.csv"
    per_problem.to_csv(joined_csv, index=False)
    print(f"[OK] wrote {joined_csv}")

    # 1) Performance radar
    agg = per_problem.groupby(["model","track"]).agg({
        "pass_at_1":"mean","pass_at_10":"mean","pass_at_100":"mean","auc_logk":"mean",
        "k50":"median","k90":"median",
        "label_WA":"sum","label_TLE":"sum","label_RE":"sum","label_CE":"sum","n_attempts":"sum"
    }).reset_index()
    topline_csv = outdir / "topline_summary.csv"
    agg.to_csv(topline_csv, index=False)
    print(f"[OK] wrote {topline_csv}")

    for model, dfm in agg.groupby("model"):
        X = dfm.set_index("track")[RADAR_METRICS].copy()
        Xn = minmax_df(X, invert_cols={"k50","k90"})
        for track, row in Xn.iterrows():
            title = f"Performance radar – {model} [{track}]"
            outprefix = outdir / f"radar_perf__{safe_name(model)}__{safe_name(track)}"
            radar_plot(row, title, outprefix)

    # 2) Failure-composition radar
    for model, dfm in agg.groupby("model"):
        dfm = dfm.set_index("track")
        fail_cols = ["label_WA","label_TLE","label_RE","label_CE"]
        for c in fail_cols:
            dfm[c.replace("label_","rate_")] = dfm[c] / dfm["n_attempts"].replace(0, np.nan)
        rate_cols = [c for c in dfm.columns if c.startswith("rate_")]
        Xn = minmax_df(dfm[rate_cols])
        for track, row in Xn.iterrows():
            title = f"Failure composition radar – {model} [{track}]"
            outprefix = outdir / f"radar_fail__{safe_name(model)}__{safe_name(track)}"
            radar_plot(row, title, outprefix)

    # 3) PCA / t-SNE on difficulty axes
    if set(["z_B","z_T","z_H","z_WA"]).issubset(per_problem.columns):
        D = per_problem[["problem_id","z_B","z_T","z_H","z_WA"]].drop_duplicates("problem_id").set_index("problem_id")
        X = D.values
        coords, comps, evr = pca_2d(X)
        pca_df = pd.DataFrame(coords, index=D.index, columns=["PC1","PC2"])
        pca_csv = outdir / "pca_coords.csv"
        pca_df.to_csv(pca_csv)
        print(f"[OK] wrote {pca_csv}")
        save_scatter_with_labels(pca_df, "PCA map (difficulty axes)", outdir / "pca_map")

        # optional t-SNE
        try:
            from sklearn.manifold import TSNE  # noqa: F401
            tsne_coords = try_tsne_2d(X, perplexity=6, n_iter=2000, random_state=RANDOM_SEED)
            if tsne_coords is not None:
                tsne_df = pd.DataFrame(tsne_coords, index=D.index, columns=["TSNE1","TSNE2"])
                tsne_csv = outdir / "tsne_coords.csv"
                tsne_df.to_csv(tsne_csv)
                print(f"[OK] wrote {tsne_csv}")
                save_scatter_with_labels(tsne_df, "t-SNE map (difficulty axes)", outdir / "tsne_map")
        except Exception as e:
            print(f"[INFO] t-SNE skipped (not installed): {e}")

    else:
        print("[INFO] z_B/z_T/z_H/z_WA not available; PCA/t-SNE skipped.")

    # 4) Heatmap: per-problem pass@K
    per_problem["model_track"] = per_problem["model"] + "_" + per_problem["track"]
    val_col = f"pass_at_{HEATMAP_K}"
    mat = per_problem.pivot_table(index="problem_id", columns="model_track", values=val_col, aggfunc="mean")

    wp = per_problem[["problem_id","w"]].drop_duplicates("problem_id").set_index("problem_id")["w"].fillna(1.0)
    order = wp.sort_values(ascending=False).index
    mat = mat.reindex(order)

    mat_csv = outdir / f"heatmap_pass_at_{HEATMAP_K}.csv"
    mat.to_csv(mat_csv)
    print(f"[OK] wrote {mat_csv}")
    heatmap(mat.values, mat.index.tolist(), mat.columns.tolist(),
            title=f"Per-problem pass@{HEATMAP_K} (rows sorted by w_p)",
            outprefix=outdir / f"heatmap_pass_at_{HEATMAP_K}")

    # 5) Heatmap: per-problem pass@k (k=1..100) for one model
    models_present = per_problem["model"].unique().tolist()
    target_model = HEATMAP_DETAILED_MODEL or (models_present[0] if models_present else None)
    if target_model is not None:
        cols = [c for c in per_problem.columns if c.startswith("pass_at_")]
        ks = sorted([int(c.split("_")[-1]) for c in cols if c.split("_")[-1].isdigit()])
        cols_sorted = [f"pass_at_{k}" for k in ks]

        dfm = per_problem[per_problem["model"]==target_model].copy()
        if "baseline" in dfm["track"].unique():
            dfm = dfm[dfm["track"]=="baseline"]
        else:
            dfm = dfm[dfm["track"]==dfm["track"].unique()[0]]

        M = dfm.set_index("problem_id")[cols_sorted].reindex(order)
        M_csv = outdir / f"heatmap_pass_at_k__{safe_name(target_model)}.csv"
        M.to_csv(M_csv)
        print(f"[OK] wrote {M_csv}")

        xticks = []; xticklabels = []
        for t in [1,10,50,100]:
            if t in ks:
                xticks.append(ks.index(t))
                xticklabels.append(str(t))

        heatmap(M.values, M.index.tolist(), [str(k) for k in ks],
                title=f"Per-problem pass@k – {target_model}",
                outprefix=outdir / f"heatmap_pass_at_k__{safe_name(target_model)}",
                xticks=xticks if xticks else None,
                xtick_labels=xticklabels if xticklabels else None)
    else:
        print("[INFO] No models available for detailed pass@k heatmap.")

    print("\\n[Done] All figures and CSVs saved to:", outdir)

if __name__ == "__main__":
    main()
