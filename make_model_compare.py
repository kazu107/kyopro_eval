#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# =========================
# CONFIG (edit here)
# =========================
MODEL_DIRS: List[str] = [
    "./Llama-3.1-8B-Instruct",
    "./deepseek-coder-7b-instruct-v1.5",
    # "./Qwen2.5-Coder-7B-Instruct",
]
PER_PROBLEM_FILENAME = "results/per_problem_pass_at_k.csv"
TRACKS = ["baseline", "CoT", "feedback"]

DIFF_WEIGHTS_CSV = r"C:\Users\kazuu\PycharmProjects\atc\Llama-3.1-8B-Instruct\results\difficulty_weights_A_baseline_only.csv"
WEIGHT_COL_IN_OUTPUT = "w"

K_LIST = [1, 10, 100]
MAKE_CURVES = True
REF_MODEL = "Llama-3.1-8B-Instruct"
N_BOOT = 1000

OUTDIR = "model_compare_out"

# =========================
# utils
# =========================
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def safe_name(s: str) -> str:
    return ''.join(ch if ch.isalnum() or ch in '-_.' else '_' for ch in s)

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
        raise RuntimeError('No per_problem CSVs found.')
    df = pd.concat(rows, ignore_index=True)
    if 'problem_id' not in df.columns and 'problem' in df.columns:
        df = df.rename(columns={'problem':'problem_id'})
    return df

def merge_weights(per_problem: pd.DataFrame, weights_csv: str, weight_col_out: str) -> pd.DataFrame:
    if os.path.isfile(weights_csv):
        w = pd.read_csv(weights_csv)
        if 'w_p' in w.columns and weight_col_out != 'w_p':
            w = w.rename(columns={'w_p': weight_col_out})
        per_problem = per_problem.merge(w[['problem_id', weight_col_out]], on='problem_id', how='left')
    else:
        print(f"[WARN] weights not found: {weights_csv}. Using {weight_col_out}=1.")
        per_problem[weight_col_out] = 1.0
    return per_problem

def get_k_columns(df: pd.DataFrame) -> Tuple[List[int], List[str]]:
    ks = []
    for c in df.columns:
        if c.startswith('pass_at_'):
            tail = c.split('_')[-1]
            if tail.isdigit():
                ks.append(int(tail))
    ks = sorted(set(ks))
    cols = [f'pass_at_{k}' for k in ks]
    return ks, cols

def mean_and_ci(values: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    x = values[np.isfinite(values)]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    m = x.mean()
    if x.size == 1 or n_boot <= 0:
        return (float(m), np.nan, np.nan)
    rng = np.random.default_rng(1729)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = x[idx].mean()
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return (float(m), lo, hi)

def d_mean_and_ci(values: np.ndarray, weights: np.ndarray, n_boot: int = 1000, alpha: float = 0.05):
    m = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    x = values[m]; w = weights[m]
    if x.size == 0:
        return (np.nan, np.nan, np.nan)
    m0 = float(np.sum(w*x) / np.sum(w))
    if x.size == 1 or n_boot <= 0:
        return (m0, np.nan, np.nan)
    rng = np.random.default_rng(1729)
    n = x.size
    boots = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]; wb = w[idx]
        boots[i] = float(np.sum(wb*xb) / np.sum(wb))
    lo = float(np.percentile(boots, 2.5))
    hi = float(np.percentile(boots, 97.5))
    return (m0, lo, hi)

def sign_test_p(a: np.ndarray, b: np.ndarray) -> float:
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

def main():
    outdir = Path(OUTDIR); ensure_dir(outdir)
    per_problem = load_all_per_problem(MODEL_DIRS, PER_PROBLEM_FILENAME)
    per_problem = per_problem[per_problem['track'].isin(TRACKS)].copy()
    per_problem = merge_weights(per_problem, DIFF_WEIGHTS_CSV, WEIGHT_COL_IN_OUTPUT)
    per_problem.rename(columns={WEIGHT_COL_IN_OUTPUT: 'w'}, inplace=True)

    (outdir / 'joined_all_models.csv').write_text(per_problem.to_csv(index=False))

    # k columns
    all_ks, pass_cols_all = get_k_columns(per_problem)

    # Leaderboard
    def summarize_group(df: pd.DataFrame) -> dict:
        row = {}
        for k in K_LIST:
            col = f'pass_at_{k}'
            if col in df.columns:
                m, lo, hi = mean_and_ci(df[col].to_numpy(), N_BOOT)
                dm, dlo, dhi = d_mean_and_ci(df[col].to_numpy(), df['w'].to_numpy(), N_BOOT)
                row[f'pass_at_{k}'] = m; row[f'pass_at_{k}_ci_lo'] = lo; row[f'pass_at_{k}_ci_hi'] = hi
                row[f'D_pass_at_{k}'] = dm; row[f'D_pass_at_{k}_ci_lo'] = dlo; row[f'D_pass_at_{k}_ci_hi'] = dhi
        for c in ['auc_logk','k50','k90']:
            if c in df.columns:
                if c == 'auc_logk':
                    m, lo, hi = mean_and_ci(df[c].to_numpy(), N_BOOT)
                else:
                    x = df[c].to_numpy(); med = np.nanmedian(x)
                    if N_BOOT>0:
                        rng = np.random.default_rng(1729)
                        x_clean = x[np.isfinite(x)]; n = len(x_clean)
                        boots = [np.nanmedian(x_clean[rng.integers(0,n,size=n)]) for _ in range(N_BOOT)]
                        lo = float(np.percentile(boots, 2.5)); hi = float(np.percentile(boots, 97.5))
                    else:
                        lo=hi=np.nan
                    m, lo, hi = med, lo, hi
                row[c] = float(m); row[f'{c}_ci_lo'] = float(lo); row[f'{c}_ci_hi'] = float(hi)
        row['n_problems'] = int(df['problem_id'].nunique())
        return row

    gb = per_problem.groupby(['model','track'], sort=False)
    leaderboard = gb.apply(summarize_group).apply(pd.Series).reset_index()
    (outdir / 'leaderboard_by_model_track.csv').write_text(leaderboard.to_csv(index=False))

    # Curves
    if MAKE_CURVES:
        rows = []
        for (model, track), df in gb:
            w = df['w'].to_numpy()
            for k in all_ks:
                col = f'pass_at_{k}'
                x = df[col].to_numpy() if col in df.columns else None
                if x is None: continue
                m, lo, hi = mean_and_ci(x, N_BOOT)
                dm, dlo, dhi = d_mean_and_ci(x, w, N_BOOT)
                rows.append({'model':model,'track':track,'k':k,
                             'mean_pass':m,'mean_ci_lo':lo,'mean_ci_hi':hi,
                             'D_pass':dm,'D_ci_lo':dlo,'D_ci_hi':dhi,
                             'n_problems': int(df['problem_id'].nunique())})
        curves = pd.DataFrame(rows).sort_values(['track','model','k'])
        (outdir / 'overall_curves_mean_and_D.csv').write_text(curves.to_csv(index=False))

    # Pairwise vs REF (same track, K_LIST)
    rows = []
    for tr in TRACKS:
        df_t = per_problem[per_problem['track']==tr]
        if df_t.empty or REF_MODEL not in df_t['model'].unique():
            continue
        ref = df_t[df_t['model']==REF_MODEL][['problem_id','w'] + [f'pass_at_{k}' for k in K_LIST if f'pass_at_{k}' in df_t.columns]].set_index('problem_id')
        for m in df_t['model'].unique():
            if m == REF_MODEL: continue
            cmp = df_t[df_t['model']==m][['problem_id','w'] + [f'pass_at_{k}' for k in K_LIST if f'pass_at_{k}' in df_t.columns]].set_index('problem_id')
            common = ref.index.intersection(cmp.index)
            if len(common)==0: continue
            w = ref.loc[common,'w'].to_numpy()
            for k in K_LIST:
                col = f'pass_at_{k}'
                if col not in ref.columns or col not in cmp.columns: continue
                a = ref.loc[common,col].to_numpy()
                b = cmp.loc[common,col].to_numpy()
                diff = b - a
                mean_diff = float(np.nanmean(diff))
                d_mean_diff = float(np.nansum(w*diff)/np.nansum(w))
                wins = int((diff>0).sum()); ties=int((diff==0).sum()); losses=int((diff<0).sum())
                p = sign_test_p(b, a)
                rows.append({'track':tr,'metric':col,'ref_model':REF_MODEL,'cmp_model':m,'n_common':int(len(common)),
                             'mean_diff':mean_diff,'D_mean_diff':d_mean_diff,
                             'wins':wins,'ties':ties,'losses':losses,'sign_p':p})
    pairwise = pd.DataFrame(rows)
    if not pairwise.empty:
        for (tr,met), sub in pairwise.groupby(['track','metric']):
            idx = sub.index.tolist()
            adj = holm_bonferroni(sub['sign_p'].fillna(1.0).tolist())
            pairwise.loc[idx, 'sign_p_holm'] = adj
    (outdir / 'pairwise_vs_ref_sign_test.csv').write_text(pairwise.to_csv(index=False))

    # Difficulty quartile stratification
    w_uni = per_problem[['problem_id','w']].drop_duplicates('problem_id').set_index('problem_id')['w']
    qs = np.quantile(w_uni.values, [0.25,0.5,0.75])
    def w_bin(val):
        if val <= qs[0]: return 'Q1 (easy)'
        elif val <= qs[1]: return 'Q2'
        elif val <= qs[2]: return 'Q3'
        else: return 'Q4 (hard)'
    wbin = w_uni.apply(w_bin).rename('w_bin')
    df_w = per_problem.merge(wbin.reset_index(), on='problem_id', how='left')

    rows = []
    for (model, track, wb), df_sub in df_w.groupby(['model','track','w_bin']):
        w = df_sub['w'].to_numpy()
        for k in K_LIST:
            col = f'pass_at_{k}'
            if col not in df_sub.columns: continue
            x = df_sub[col].to_numpy()
            m, lo, hi = mean_and_ci(x, N_BOOT)
            dm, dlo, dhi = d_mean_and_ci(x, w, N_BOOT)
            rows.append({'model':model,'track':track,'w_bin':wb,'k':k,
                         'mean_pass':m,'mean_ci_lo':lo,'mean_ci_hi':hi,
                         'D_pass':dm,'D_ci_lo':dlo,'D_ci_hi':dhi,
                         'n_problems': int(df_sub['problem_id'].nunique())})
    strata = pd.DataFrame(rows).sort_values(['w_bin','track','model','k'])
    (outdir / 'stratified_by_difficulty_quartiles.csv').write_text(strata.to_csv(index=False))

    # Heatmap matrices
    for K in K_LIST:
        col = f'pass_at_{K}'
        if col not in per_problem.columns: continue
        mat = per_problem.pivot_table(index='problem_id', columns=['track','model'], values=col, aggfunc='mean')
        order = w_uni.sort_values(ascending=False).index
        mat = mat.reindex(order)
        (outdir / f'heatmap_matrix_pass_at_{K}.csv').write_text(mat.to_csv())

    print('\n[Done] Wrote CSVs to', OUTDIR)

if __name__ == '__main__':
    main()
