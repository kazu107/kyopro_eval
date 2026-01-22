# analyze_difficulty_bins_v1.py
# -*- coding: utf-8 -*-
from pathlib import Path
import numpy as np
import pandas as pd
import math
from collections import defaultdict

# ===== 設定 =====
JOINED_CSV = Path("model_compare_out/joined_all_models.csv")   # 必要
WEIGHT_CSV = Path("C:/Users/kazuu/PycharmProjects/atc/Llama-3.1-8B-Instruct/results/difficulty_weights_A_baseline_only.csv")    # 必要
OUT_DIR    = Path("./difficulty_checks_out")
TARGET_MODELS = None  # 例: ["Llama-3.1-8B-Instruct","deepseek-coder-7b-instruct-v1.5"]。Noneなら全て
TARGET_TRACKS = None  # 例: ["baseline","CoT","feedback"]。Noneなら全て
K_LIST = [1, 10, 100]
N_BOOT = 5000
RANDOM_SEED = 42
# ===============

rng = np.random.default_rng(RANDOM_SEED)
OUT_DIR.mkdir(parents=True, exist_ok=True)

def need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"not found: {p.resolve()}")
    return p

def std_problem_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "problem_id" not in df.columns:
        for alt in ["problem", "task_id", "id"]:
            if alt in df.columns:
                df = df.rename(columns={alt:"problem_id"})
                break
    if "problem_id" not in df.columns:
        raise KeyError("problem_id key not found")
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    return df

def load_joined(p: Path) -> pd.DataFrame:
    df = pd.read_csv(need(p))
    df = std_problem_id(df)
    for c in ["model","track"]:
        if c not in df.columns:
            raise KeyError(f"{c} missing in joined")
        df[c] = df[c].astype(str).str.strip()
    return df

def load_weights(p: Path) -> pd.DataFrame:
    w = pd.read_csv(need(p))
    w = std_problem_id(w)
    # 重み列の検出
    wcol = None
    for cand in ["w","w_p","weight","weight_p"]:
        if cand in w.columns:
            wcol = cand; break
    if wcol is None:
        raise KeyError(f"weight col not found in {list(w.columns)}")
    if wcol != "w":
        w = w.rename(columns={wcol:"w"})
    w["w"] = pd.to_numeric(w["w"], errors="coerce")
    return w[["problem_id","w"]]

def merge_weights(joined: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()
    # 既存wは退避
    if "w" in df.columns:
        df = df.rename(columns={"w":"w_joined"})
    df = df.merge(weights.rename(columns={"w":"w_weights"}), on="problem_id", how="left", validate="m:1")
    # 最終wを決定
    df["w"] = df["w_weights"]
    if "w_joined" in df.columns:
        df["w"] = df["w"].fillna(df["w_joined"])
    df["w"] = df["w"].fillna(1.0)
    # 不要列は掃除
    for c in ["w_joined","w_weights"]:
        if c in df.columns: df = df.drop(columns=[c])
    return df

def qcut_equalize(w: pd.Series, jitter_eps=1e-9):
    """等頻度でQ1..Q4を作る。tiesが多い場合は微小ジッターでブレークして強制4分位化。"""
    s = w.astype(float).copy()
    # まず素直な qcut を試す
    try:
        bins = pd.qcut(s, q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
        if len(bins.cat.categories) == 4:
            return bins
    except Exception:
        pass
    # ジッターを入れて再試行
    span = np.nanmax(s) - np.nanmin(s)
    jitter = (rng.normal(loc=0.0, scale=1.0, size=len(s))) * (span + 1.0) * jitter_eps
    s2 = s + jitter
    bins = pd.qcut(s2, q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
    if hasattr(bins, "cat") and len(bins.cat.categories)==4:
        return bins
    # それでもダメなら rank を使う
    ranks = s.rank(method="average")
    bins = pd.qcut(ranks, q=4, labels=["Q1","Q2","Q3","Q4"], duplicates="drop")
    return bins

def bootstrap_ci_mean(x: np.ndarray, n_boot=N_BOOT, alpha=0.05):
    x = x[~np.isnan(x)]
    if len(x)==0:
        return (np.nan, np.nan, np.nan)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = np.quantile(means, alpha/2)
    hi = np.quantile(means, 1-alpha/2)
    return (float(np.mean(x)), float(lo), float(hi))

def safe_corr(a: pd.Series, b: pd.Series, method="pearson"):
    df = pd.DataFrame({"a":a, "b":b}).replace([np.inf,-np.inf], np.nan).dropna()
    if len(df)<3: return np.nan
    if method=="spearman":
        return df["a"].rank().corr(df["b"].rank())
    return df["a"].corr(df["b"])

def main():
    joined = load_joined(JOINED_CSV)
    weights = load_weights(WEIGHT_CSV)
    df = merge_weights(joined, weights)

    models = TARGET_MODELS or sorted(df["model"].unique())
    tracks = TARGET_TRACKS or sorted(df["track"].unique())

    # 出力テーブル
    rows_corr = []
    rows_bins = []

    for m in models:
        for tr in tracks:
            sub = df[(df["model"]==m) & (df["track"]==tr)].copy()
            if sub.empty: continue

            # 相関
            for k in K_LIST:
                col = f"pass_at_{k}"
                if col not in sub.columns: continue
                r_p = safe_corr(sub["w"], sub[col], method="pearson")
                r_s = safe_corr(sub["w"], sub[col], method="spearman")
                rows_corr.append({
                    "model": m, "track": tr, "k": k,
                    "pearson_corr_w_vs_pass": r_p,
                    "spearman_corr_w_vs_pass": r_s,
                    "n": int(sub[[col,"w"]].replace([np.inf,-np.inf],np.nan).dropna().shape[0]),
                })

            # 等頻度四分位（Q1..Q4）
            bins = qcut_equalize(sub["w"])
            sub["w_bin"] = bins
            # 件数確認
            counts = sub["w_bin"].value_counts().reindex(["Q1","Q2","Q3","Q4"]).fillna(0).astype(int).to_dict()

            for k in K_LIST:
                col = f"pass_at_{k}"
                if col not in sub.columns: continue
                for q in ["Q1","Q2","Q3","Q4"]:
                    x = pd.to_numeric(sub.loc[sub["w_bin"]==q, col], errors="coerce").values
                    mean, lo, hi = bootstrap_ci_mean(x, n_boot=N_BOOT, alpha=0.05)
                    rows_bins.append({
                        "model": m, "track": tr, "k": k, "w_bin": q,
                        "n_in_bin": int(len(x)),
                        "mean": mean, "ci_lo": lo, "ci_hi": hi
                    })

            # 進行ログ
            print(f"[INFO] {m} / {tr}  bin counts:", counts)

    corr_df = pd.DataFrame(rows_corr).sort_values(["model","track","k"])
    bins_df = pd.DataFrame(rows_bins).sort_values(["model","track","k","w_bin"])

    corr_df.to_csv(OUT_DIR / "corr_w_vs_pass_by_model_track.csv", index=False)
    bins_df.to_csv(OUT_DIR / "quartile_means_with_CI_by_model_track.csv", index=False)

    # 画面にも要点を出す
    print("\n=== Correlations (pearson/spearman) ===")
    print(corr_df.to_string(index=False, max_rows=200))

    print("\n=== Quartile means (95% CI) ===")
    def fmt(r):
        return f"{r['mean']:.3f} [{r['ci_lo']:.3f},{r['ci_hi']:.3f}]"
    for (m,tr,k), g in bins_df.groupby(["model","track","k"]):
        g = g.set_index("w_bin").reindex(["Q1","Q2","Q3","Q4"])
        s = " | ".join([f"{q}: {fmt(g.loc[q])} (n={int(g.loc[q,'n_in_bin'])})" for q in ["Q1","Q2","Q3","Q4"]])
        print(f"{m} / {tr} / pass@{k}:  {s}")

if __name__ == "__main__":
    main()
