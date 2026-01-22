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
MODEL_DIRS: List[str] = [
    #"./Llama-3.1-8B-Instruct",
    "./deepseek-coder-7b-instruct-v1.5"
]
PER_PROBLEM_FILENAME = "results/per_problem_pass_at_k.csv"
TRACKS = ["baseline", "CoT", "feedback"]

DIFF_WEIGHTS_CSV = r"C:\Users\kazuu\PycharmProjects\atc\Llama-3.1-8B-Instruct\results\difficulty_weights_A_baseline_only.csv"      # columns: problem_id,w_p
DIFF_AXES_CSV    = r"C:\Users\kazuu\PycharmProjects\atc\Llama-3.1-8B-Instruct\results\difficulty_metrics_all_axes_baseline_only.csv"  # columns: problem_id,z_B,z_T,z_H,z_WA

OUTPUT_DIR = "viz_out_deepseek"
SAVE_SVG = True

RADAR_MODE = "absolute"  # 'absolute' or 'minmax'
K_SCORE_MODE = "lin"     # 'lin' or 'inv1p'
K_MAX_FOR_RADAR = 100

HEATMAP_K = 10
HEATMAP_DETAILED_MODEL = None

RANDOM_SEED = 1729
np.random.seed(RANDOM_SEED)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in s)

def minmax_df(df: pd.DataFrame, invert_cols=None) -> pd.DataFrame:
    X = df.copy()
    invert_cols = set(invert_cols or [])
    for c in X.columns:
        if c in invert_cols:
            X[c] = 1.0 / (1.0 + X[c])
    for c in X.columns:
        col = X[c].astype(float)
        lo, hi = col.min(), col.max()
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            X[c] = (col - lo) / (hi - lo)
        else:
            X[c] = 0.5
    return X

def radar_plot(values: pd.Series, title: str, outprefix: Path, rlabel=None):
    labels = values.index.tolist()
    vals = values.values.astype(float).tolist()
    vals += vals[:1]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig = plt.figure()
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, vals, linewidth=2)
    ax.fill(angles, vals, alpha=0.15)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    if rlabel:
        ax.set_rlabel_position(90)
    ax.set_title(title)
    plt.tight_layout()

    fig.savefig(str(outprefix)+".png", dpi=220)
    if SAVE_SVG:
        fig.savefig(str(outprefix)+".svg")
    plt.close(fig)

def k_to_score(k: float) -> float:
    if k is None or not np.isfinite(k):
        return 0.0
    k = float(k)
    k = max(1.0, min(K_MAX_FOR_RADAR, k))
    if K_SCORE_MODE == "inv1p":
        return 1.0 / (1.0 + k)
    else:
        return (K_MAX_FOR_RADAR + 1.0 - k) / K_MAX_FOR_RADAR

def pca_2d(X: np.ndarray):
    X = np.asarray(X, dtype=float)
    Xc = X - X.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    comps = Vt[:2, :]
    coords = Xc @ comps.T
    ev = (S**2) / (X.shape[0]-1)
    evr = ev / ev.sum()
    return coords, comps, evr[:2]

def try_tsne_2d(X: np.ndarray, perplexity=6, n_iter=2000, random_state=0):
    try:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, init='pca', random_state=random_state, n_iter=n_iter)
        Y = tsne.fit_transform(X)
        return Y
    except Exception as e:
        print(f"[INFO] t-SNE skipped: {e}")
        return None

def save_scatter_with_labels(df2: pd.DataFrame, title: str, outprefix: Path):
    fig, ax = plt.subplots()
    ax.scatter(df2.iloc[:,0], df2.iloc[:,1])
    for pid, (x,y) in df2.iterrows():
        ax.text(x, y, pid, fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(df2.columns[0]); ax.set_ylabel(df2.columns[1])
    plt.tight_layout()
    fig.savefig(str(outprefix)+".png", dpi=220)
    if SAVE_SVG:
        fig.savefig(str(outprefix)+".svg")
    plt.close(fig)

def heatmap(values: np.ndarray, row_labels, col_labels, title: str, outprefix: Path, xticks=None, xtick_labels=None):
    fig, ax = plt.subplots()
    im = ax.imshow(values, aspect='auto', interpolation='nearest')
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=7)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha='right', fontsize=8)
    if xticks is not None and xtick_labels is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)
    fig.colorbar(im, ax=ax, label='value')
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(str(outprefix)+'.png', dpi=220)
    if SAVE_SVG:
        fig.savefig(str(outprefix)+'.svg')
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
        df['model'] = mdir.name
        rows.append(df)
    if not rows:
        raise RuntimeError("No per_problem CSVs found.")
    df = pd.concat(rows, ignore_index=True)
    if 'problem_id' not in df.columns and 'problem' in df.columns:
        df = df.rename(columns={'problem':'problem_id'})
    return df

def main():
    outdir = Path(OUTPUT_DIR); ensure_dir(outdir)
    per_problem = load_all_per_problem(MODEL_DIRS, PER_PROBLEM_FILENAME)
    per_problem = per_problem[per_problem['track'].isin(TRACKS)].copy()

    if os.path.isfile(DIFF_WEIGHTS_CSV):
        weights = pd.read_csv(DIFF_WEIGHTS_CSV)
    else:
        print(f"[WARN] weights CSV not found: {DIFF_WEIGHTS_CSV}. Using w_p=1.")
        weights = pd.DataFrame({'problem_id': per_problem['problem_id'].unique(), 'w_p': 1.0})
    if os.path.isfile(DIFF_AXES_CSV):
        axes = pd.read_csv(DIFF_AXES_CSV)
    else:
        print(f"[WARN] axes CSV not found: {DIFF_AXES_CSV}.")
        axes = pd.DataFrame({'problem_id': per_problem['problem_id'].unique()})
    per_problem = per_problem.merge(weights, on='problem_id', how='left').merge(
        axes[['problem_id','z_B','z_T','z_H','z_WA']] if set(['z_B','z_T','z_H','z_WA']).issubset(axes.columns) else axes[['problem_id']],
        on='problem_id', how='left'
    )

    (outdir / 'joined_per_problem.csv').write_text(per_problem.to_csv(index=False))

    # Radar
    agg = per_problem.groupby(['model','track']).agg({
        'pass_at_1':'mean','pass_at_10':'mean','pass_at_100':'mean','auc_logk':'mean',
        'k50':'median','k90':'median',
        'label_WA':'sum','label_TLE':'sum','label_RE':'sum','label_CE':'sum','n_attempts':'sum'
    }).reset_index()
    (outdir / 'topline_summary.csv').write_text(agg.to_csv(index=False))

    for model, dfm in agg.groupby('model'):
        dfm = dfm.set_index('track')

        if RADAR_MODE == 'minmax':
            X = dfm[['pass_at_1','pass_at_10','pass_at_100','auc_logk','k50','k90']].copy()
            Xn = minmax_df(X, invert_cols={'k50','k90'})
            for track, row in Xn.iterrows():
                radar_plot(row, f"Performance radar – {model} [{track}] (min-max)",
                           outdir / f"radar_perf__{safe_name(model)}__{safe_name(track)}", rlabel=True)
        else:
            rows = {}
            for track, row in dfm.iterrows():
                auc_norm = (row['auc_logk'] / math.log(K_MAX_FOR_RADAR)) if np.isfinite(row['auc_logk']) else 0.0
                k50_s = k_to_score(row['k50'] if np.isfinite(row['k50']) else K_MAX_FOR_RADAR)
                k90_s = k_to_score(row['k90'] if np.isfinite(row['k90']) else K_MAX_FOR_RADAR)
                rows[track] = pd.Series({
                    'pass_at_1':  row['pass_at_1'],
                    'pass_at_10': row['pass_at_10'],
                    'pass_at_100':row['pass_at_100'],
                    'auc_logk':   auc_norm,
                    'k50':        k50_s,
                    'k90':        k90_s,
                })
            Xabs = pd.DataFrame(rows).T
            for track, row in Xabs.iterrows():
                radar_plot(row, f"Performance radar – {model} [{track}] (absolute)",
                           outdir / f"radar_perf__{safe_name(model)}__{safe_name(track)}", rlabel=True)

        # failure
        fail = dfm[['label_WA','label_TLE','label_RE','label_CE','n_attempts']].copy()
        for c in ['label_WA','label_TLE','label_RE','label_CE']:
            fail[c.replace('label_','rate_')] = fail[c] / fail['n_attempts'].replace(0, np.nan)
        rate_cols = [c for c in fail.columns if c.startswith('rate_')]
        Xf = fail[rate_cols].clip(lower=0.0, upper=1.0) if RADAR_MODE=='absolute' else minmax_df(fail[rate_cols])
        for track, row in Xf.iterrows():
            radar_plot(row, f"Failure composition radar – {model} [{track}] ({RADAR_MODE})",
                       outdir / f"radar_fail__{safe_name(model)}__{safe_name(track)}", rlabel=True)

    # PCA / t-SNE
    if set(['z_B','z_T','z_H','z_WA']).issubset(per_problem.columns):
        D = per_problem[['problem_id','z_B','z_T','z_H','z_WA']].drop_duplicates('problem_id').set_index('problem_id')
        X = D.values
        coords, comps, evr = pca_2d(X)
        pca_df = pd.DataFrame(coords, index=D.index, columns=['PC1','PC2'])
        pca_df.to_csv(outdir / 'pca_coords.csv', index=True)
        fig, ax = plt.subplots()
        ax.scatter(pca_df['PC1'], pca_df['PC2'])
        for pid, (x,y) in pca_df.iterrows():
            ax.text(x, y, pid, fontsize=8)
        ax.set_title(f"PCA map (difficulty axes) – EVR: PC1={evr[0]:.2f}, PC2={evr[1]:.2f}")
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); plt.tight_layout()
        plt.savefig(str(outdir / 'pca_map.png'), dpi=220); 
        if SAVE_SVG: plt.savefig(str(outdir / 'pca_map.svg')); 
        plt.close()

    # Heatmaps
    per_problem['model_track'] = per_problem['model'] + '_' + per_problem['track']
    val_col = f"pass_at_{HEATMAP_K}"
    mat = per_problem.pivot_table(index='problem_id', columns='model_track', values=val_col, aggfunc='mean')
    wp = per_problem[['problem_id','w']].drop_duplicates('problem_id').set_index('problem_id')['w'].fillna(1.0)
    order = wp.sort_values(ascending=False).index
    mat = mat.reindex(order)
    mat.to_csv(outdir / f"heatmap_pass_at_{HEATMAP_K}.csv", index=True)
    heatmap(mat.values, mat.index.tolist(), mat.columns.tolist(),
            f"Per-problem pass@{HEATMAP_K} (rows sorted by w_p)", outdir / f"heatmap_pass_at_{HEATMAP_K}")

    models_present = per_problem['model'].unique().tolist()
    target_model = HEATMAP_DETAILED_MODEL or (models_present[0] if models_present else None)
    if target_model is not None:
        cols = [c for c in per_problem.columns if c.startswith('pass_at_')]
        ks = sorted([int(c.split('_')[-1]) for c in cols if c.split('_')[-1].isdigit()])
        cols_sorted = [f"pass_at_{k}" for k in ks]
        dfm = per_problem[per_problem['model']==target_model].copy()
        if 'baseline' in dfm['track'].unique():
            dfm = dfm[dfm['track']=='baseline']
        else:
            dfm = dfm[dfm['track']==dfm['track'].unique()[0]]
        M = dfm.set_index('problem_id')[cols_sorted].reindex(order)
        M.to_csv(outdir / f"heatmap_pass_at_k__{safe_name(target_model)}.csv", index=True)
        xticks=[]; labels=[]
        for t in [1,10,50,100]:
            if t in ks: xticks.append(ks.index(t)); labels.append(str(t))
        heatmap(M.values, M.index.tolist(), [str(k) for k in ks],
                f"Per-problem pass@k – {target_model}", outdir / f"heatmap_pass_at_k__{safe_name(target_model)}",
                xticks=xticks if xticks else None, xtick_labels=labels if labels else None)

if __name__ == "__main__":
    main()
