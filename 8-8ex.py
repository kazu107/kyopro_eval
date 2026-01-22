# make_fig8_19_21.py
# -*- coding: utf-8 -*-
"""
図8.19（PCA Δ@1）、図8.20（t-SNE Δ@1/Δ@10）、図8.21（PCA 四分位ハル＋平均Δ注記）、
表8.12（PC1/PC2 × Δ のSpearman相関）を一括生成します。

【前提ファイル（列名は最低限下記を想定）】
- pca_coords.csv:      problem_id, PC1, PC2, [任意: w or w_p, z_B, z_T, z_H, z_WA]
- tsne_coords.csv:     problem_id, tSNE1, tSNE2  ※列名が異なる場合は下の CONFIG で列名を変更
- per-problem CSV（候補のいずれかで自動探索）
  1) joined_per_problem.csv
  2) model_compare_out/joined_all_models.csv
  必須列: problem_id, track, model, pass_at_1, pass_at_10（必要に応じて pass_at_k）

【モデル名の正規化】
- Llama-3.1-8B-Instruct（基準）
- deepseek-coder-7b-instruct-v1.5

【出力】
- out/fig_8_19_pca_delta_at1.png
- out/fig_8_20_tsne_delta_at1_at10.png
- out/fig_8_21_pca_quartile_hulls.png
- out/table_8_12_pc_corr.csv

【注意】
- 依存: pandas, numpy, matplotlib, scipy
- seabornは使用しません。
- 列名が異なる場合は CONFIG を調整してください。
"""

import os
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import spearmanr
from scipy.spatial import ConvexHull

# =========================
# CONFIG（必要に応じて編集）
# =========================
PCA_CSV = "./viz_out/pca_coords.csv"
TSNE_CSV = "./viz_out/tsne_coords.csv"

# per-problem の自動探索候補（見つかった最初の1つを使用）
PER_PROBLEM_CSV_CANDIDATES = [
    "joined_per_problem.csv",
    os.path.join("model_compare_out", "joined_all_models.csv"),
]

# 列名マッピング（データの列名が異なる場合は変更）
COL_PROB = "problem_id"
COL_PC1, COL_PC2 = "PC1", "PC2"
COL_TSNE1, COL_TSNE2 = "tSNE1", "tSNE2"    # tsne_coordsの列
COL_TRACK, COL_MODEL = "track", "model"
COL_PASS1, COL_PASS10 = "pass_at_1", "pass_at_10"
COL_W = "w"     # 難易度重み列（pcaやweightsに含まれる場合）
ALT_COL_W = "w_p"  # 代替候補

# 使用するトラックとモデル（部分一致で正規化）
TRACKS = ["baseline", "CoT", "feedback"]
LLAMA_ALIASES = ["Llama-3.1-8B-Instruct", "Llama", "llama"]
DEEPSEEK_ALIASES = ["deepseek-coder-7b-instruct-v1.5", "DeepSeek", "deepseek"]

# 出力先
OUTDIR = "figs_8_8ex"
DPI = 180

# 図の描画パラメータ
POINT_SIZE = 50
ALPHA_POINTS = 0.9
CMAP = "coolwarm"  # 中央0の両極性カラーマップ
CBAR_LABEL_AT1 = "Δpass@1 (DeepSeek − Llama)"
CBAR_LABEL_AT10 = "Δpass@10 (DeepSeek − Llama)"

# =========================
# ユーティリティ
# =========================
def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)

def read_csv_must(path: str | Path, what: str) -> pd.DataFrame:
    if not Path(path).is_file():
        raise FileNotFoundError(f"[ERROR] {what} not found: {path}")
    return pd.read_csv(path)

def find_per_problem_csv() -> str:
    for p in PER_PROBLEM_CSV_CANDIDATES:
        if Path(p).is_file():
            return p
    raise FileNotFoundError(
        "[ERROR] per-problem CSV not found. "
        f"Tried: {PER_PROBLEM_CSV_CANDIDATES}"
    )

def normalize_model_name(s: str) -> str:
    sl = s.lower()
    for a in LLAMA_ALIASES:
        if a.lower() in sl:
            return "Llama-3.1-8B-Instruct"
    for a in DEEPSEEK_ALIASES:
        if a.lower() in sl:
            return "deepseek-coder-7b-instruct-v1.5"
    return s  # その他はそのまま（後で除外）

def to_track_keep(s: str) -> bool:
    return s in TRACKS

def symmetric_vmin_vmax(x: np.ndarray, clip_pct: float = 1.0) -> tuple[float, float]:
    """
    Δの分布に合わせて、色の振幅を対称にする。
    clip_pct: 0<clip_pct<=100 のパーセンタイルで外れ値を軽く抑える（デフォルト=100:無裁断）。
    """
    x = x[np.isfinite(x)]
    if x.size == 0:
        return -1, 1
    if clip_pct < 100.0:
        lo = np.percentile(x, (100-clip_pct)/2)
        hi = np.percentile(x, 100-(100-clip_pct)/2)
        m = max(abs(lo), abs(hi))
    else:
        m = np.max(np.abs(x))
    if m == 0:
        m = 1e-6
    return -m, +m

def compute_delta(df_per: pd.DataFrame, k_col: str) -> pd.DataFrame:
    """
    per-problem DF（問題×track×model×pass_at_k）から
    Δ = DeepSeek − Llama を問題×trackで計算して返す。
    """
    df = df_per.copy()
    # 正規化
    df[COL_MODEL] = df[COL_MODEL].astype(str).map(normalize_model_name)
    df = df[df[COL_TRACK].map(to_track_keep)]
    # 必要列チェック
    need_cols = [COL_PROB, COL_TRACK, COL_MODEL, k_col]
    for c in need_cols:
        if c not in df.columns:
            raise KeyError(f"[ERROR] per-problem is missing column: {c}")

    # ピボットして (problem, track) 行、列=モデル
    piv = df.pivot_table(index=[COL_PROB, COL_TRACK], columns=COL_MODEL, values=k_col, aggfunc="mean")
    # モデル列が無ければエラー
    if "Llama-3.1-8B-Instruct" not in piv.columns or "deepseek-coder-7b-instruct-v1.5" not in piv.columns:
        raise KeyError("[ERROR] required models not found after normalization. "
                       f"Existing columns: {list(piv.columns)}")

    piv = piv.reset_index()
    piv["delta"] = piv["deepseek-coder-7b-instruct-v1.5"] - piv["Llama-3.1-8B-Instruct"]
    out = piv[[COL_PROB, COL_TRACK, "delta"]].rename(columns={"delta": f"delta_{k_col}"})
    return out

def join_weight_base(pca_df: pd.DataFrame) -> pd.Series:
    """
    四分位ハル用の重み列 w を返す。
    優先: 'w' → 'w_p' → 代替（z_B,z_T,z_H,z_WA の平均） → 最後は PC1の|z| など簡易代理。
    """
    if COL_W in pca_df.columns:
        w = pca_df[COL_W].astype(float)
    elif ALT_COL_W in pca_df.columns:
        w = pca_df[ALT_COL_W].astype(float)
    else:
        # 代替: 難易度素性があれば正規化して平均
        cand = [c for c in ["z_B", "z_T", "z_H", "z_WA"] if c in pca_df.columns]
        if cand:
            z = pca_df[cand].copy()
            for c in cand:
                z[c] = (z[c] - z[c].mean()) / (z[c].std(ddof=0) + 1e-12)
            w = z.mean(axis=1)
        else:
            # 最後の手段: PC1をZ化して|PC1|を代理（極端な易/難の代表度合い）
            zpc1 = (pca_df[COL_PC1] - pca_df[COL_PC1].mean()) / (pca_df[COL_PC1].std(ddof=0) + 1e-12)
            w = zpc1.abs()
    return w

def quartile_bins(w: pd.Series) -> pd.Series:
    qs = np.quantile(w, [0.25, 0.5, 0.75])
    def _bin(val):
        if val <= qs[0]:
            return "Q1 (easy)"
        elif val <= qs[1]:
            return "Q2"
        elif val <= qs[2]:
            return "Q3"
        else:
            return "Q4 (hard)"
    return w.map(_bin)

# =========================
# メイン処理
# =========================
def main():
    ensure_dir(OUTDIR)

    # --- 入力読み込み ---
    pca = read_csv_must(PCA_CSV, "PCA coords")
    for c in [COL_PROB, COL_PC1, COL_PC2]:
        if c not in pca.columns:
            raise KeyError(f"[ERROR] PCA CSV missing column: {c}")

    tsne = read_csv_must(TSNE_CSV, "t-SNE coords")
    for c in [COL_PROB, COL_TSNE1, COL_TSNE2]:
        if c not in tsne.columns:
            raise KeyError(f"[ERROR] t-SNE CSV missing column: {c}")

    per_path = find_per_problem_csv()
    per = read_csv_must(per_path, "per-problem")
    # 代表カラムがなければ列名の再指定を促す
    for c in [COL_PROB, COL_TRACK, COL_MODEL, COL_PASS1, COL_PASS10]:
        if c not in per.columns:
            raise KeyError(f"[ERROR] per-problem CSV missing column: {c} in {per_path}")

    # --- Δ計算（@1/@10） ---
    d1 = compute_delta(per, COL_PASS1)   # 列名: delta_pass_at_1
    d10 = compute_delta(per, COL_PASS10) # 列名: delta_pass_at_10
    deltas = d1.merge(d10, on=[COL_PROB, COL_TRACK], how="outer")

    # --- PCA / tSNE と結合 ---
    base = deltas.merge(pca[[COL_PROB, COL_PC1, COL_PC2]], on=COL_PROB, how="left")
    base = base.merge(tsne[[COL_PROB, COL_TSNE1, COL_TSNE2]], on=COL_PROB, how="left")

    # --- 図8.19: PCA（Δ@1） ---
    fig, axes = plt.subplots(1, len(TRACKS), figsize=(5.0*len(TRACKS), 4.8), dpi=DPI, constrained_layout=True)
    if len(TRACKS) == 1:
        axes = [axes]
    vmin, vmax = symmetric_vmin_vmax(base["delta_"+COL_PASS1].values, clip_pct=98.0)

    for ax, tr in zip(axes, TRACKS):
        sub = base[base[COL_TRACK] == tr].copy()
        sc = ax.scatter(sub[COL_PC1], sub[COL_PC2],
                        c=sub["delta_"+COL_PASS1],
                        s=POINT_SIZE, alpha=ALPHA_POINTS,
                        cmap=CMAP, norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax),
                        edgecolor="none")
        ax.set_title(f"PCA • Δ@1 • {tr}", fontsize=12)
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(True, linewidth=0.3, alpha=0.5)
    cbar = fig.colorbar(sc, ax=axes, shrink=0.9)
    cbar.set_label(CBAR_LABEL_AT1)
    p_fig_819 = os.path.join(OUTDIR, "fig_8_19_pca_delta_at1.png")
    fig.savefig(p_fig_819)
    plt.close(fig)

    # --- 図8.20: t-SNE（Δ@1/Δ@10、行=track, 列=2）---
    nrow, ncol = len(TRACKS), 2
    # ★ここだけ constrained_layout=False にして、右側に余白を明示的に確保
    fig, axes = plt.subplots(nrow, ncol, figsize=(10.0, 4.5*nrow), dpi=DPI, constrained_layout=False)
    if nrow == 1:
        axes = np.array([axes])

    # サブプロット群のレイアウトを調整して、右側にカラーバー用スペースを空ける
    fig.subplots_adjust(left=0.08, right=0.80, top=0.95, bottom=0.07,
                        wspace=0.35, hspace=0.35)

    vmin1, vmax1 = symmetric_vmin_vmax(base["delta_"+COL_PASS1].values, clip_pct=98.0)
    vmin10, vmax10 = symmetric_vmin_vmax(base["delta_"+COL_PASS10].values, clip_pct=98.0)

    for i, tr in enumerate(TRACKS):
        sub = base[base[COL_TRACK] == tr].copy()
        # Δ@1
        ax = axes[i, 0]
        sc1 = ax.scatter(sub[COL_TSNE1], sub[COL_TSNE2],
                         c=sub["delta_"+COL_PASS1],
                         s=POINT_SIZE, alpha=ALPHA_POINTS,
                         cmap=CMAP, norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin1, vmax=vmax1),
                         edgecolor="none")
        ax.set_title(f"t-SNE • Δ@1 • {tr}", fontsize=12)
        ax.set_xlabel("t-SNE1"); ax.set_ylabel("t-SNE2")
        ax.grid(True, linewidth=0.3, alpha=0.5)
        # Δ@10
        ax = axes[i, 1]
        sc10 = ax.scatter(sub[COL_TSNE1], sub[COL_TSNE2],
                          c=sub["delta_"+COL_PASS10],
                          s=POINT_SIZE, alpha=ALPHA_POINTS,
                          cmap=CMAP, norm=TwoSlopeNorm(vcenter=0.0, vmin=vmin10, vmax=vmax10),
                          edgecolor="none")
        ax.set_title(f"t-SNE • Δ@10 • {tr}", fontsize=12)
        ax.set_xlabel("t-SNE1"); ax.set_ylabel("t-SNE2")
        ax.grid(True, linewidth=0.3, alpha=0.5)

    # 共通カラーバーを2本（Δ@1, Δ@10）: 右側の余白に縦長で配置
    # 必要に応じて 0.83, 0.02, 0.35 といった値を微調整してください
    cax1 = fig.add_axes([0.88, 0.55, 0.02, 0.35])
    cax2 = fig.add_axes([0.88, 0.10, 0.02, 0.35])
    cb1 = fig.colorbar(sc1, cax=cax1)
    cb2 = fig.colorbar(sc10, cax=cax2)
    cb1.set_label(CBAR_LABEL_AT1)
    cb2.set_label(CBAR_LABEL_AT10)
    p_fig_820 = os.path.join(OUTDIR, "fig_8_20_tsne_delta_at1_at10.png")
    fig.savefig(p_fig_820, bbox_inches="tight")
    plt.close(fig)

    # --- 図8.21: PCA 上に四分位ハル（Q1〜Q4）＋平均Δ注記（@1） ---
    # 重み w を用意し、四分位ビンを算出
    pca_ex = pca.copy()
    pca_ex["__w__"] = join_weight_base(pca_ex)
    pca_ex["w_bin"] = quartile_bins(pca_ex["__w__"])

    # Δ@1 を問題単位に集約（track横断の平均、注記用）
    d1_avg = d1.groupby(COL_PROB, as_index=False)[f"delta_{COL_PASS1}"].mean().rename(
        columns={f"delta_{COL_PASS1}": "delta_at1_mean"}
    )
    pca_h = pca_ex.merge(d1_avg, on=COL_PROB, how="left")

    # 描画
    fig, ax = plt.subplots(figsize=(6.4, 5.6), dpi=DPI, constrained_layout=True)

    # まず点（Δ@1）を背景に描く
    vminh, vmaxh = symmetric_vmin_vmax(pca_h["delta_at1_mean"].values, clip_pct=98.0)
    sc = ax.scatter(pca_h[COL_PC1], pca_h[COL_PC2],
                    c=pca_h["delta_at1_mean"],
                    s=35, alpha=0.6, cmap=CMAP,
                    norm=TwoSlopeNorm(vcenter=0.0, vmin=vminh, vmax=vmaxh),
                    edgecolor="none")

    # 各四分位で凸包を描く
    for wb, sub in pca_h.groupby("w_bin"):
        pts = sub[[COL_PC1, COL_PC2]].to_numpy()
        if pts.shape[0] >= 3:
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                ax.fill(poly[:, 0], poly[:, 1], alpha=0.15, label=wb)
            except Exception:
                # 退避：うまくハルが引けない場合はスキップ
                pass
        # 四分位ごとの平均Δ@1を注記
        mu = np.nanmean(sub["delta_at1_mean"].values)
        ax.text(np.nanmedian(sub[COL_PC1].values),
                np.nanmedian(sub[COL_PC2].values),
                f"{wb}\nΔ@1={mu:+.3f}",
                fontsize=5, ha="center", va="center")

    ax.set_title("PCA with quartile hulls & mean Δ@1")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    ax.legend(frameon=False, loc="best")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.85)
    cbar.set_label(CBAR_LABEL_AT1)
    p_fig_821 = os.path.join(OUTDIR, "fig_8_21_pca_quartile_hulls.png")
    fig.savefig(p_fig_821)
    plt.close(fig)

    # --- 表8.12: Spearman（PC×Δ@1/Δ@10、track別） ---
    rows = []
    for tr in TRACKS:
        sub = base[base[COL_TRACK] == tr].copy()
        # 欠損除去
        m1 = np.isfinite(sub[COL_PC1]) & np.isfinite(sub["delta_"+COL_PASS1])
        m2 = np.isfinite(sub[COL_PC2]) & np.isfinite(sub["delta_"+COL_PASS1])
        m3 = np.isfinite(sub[COL_PC1]) & np.isfinite(sub["delta_"+COL_PASS10])
        m4 = np.isfinite(sub[COL_PC2]) & np.isfinite(sub["delta_"+COL_PASS10])
        # ρとp
        def safe_spear(x, y):
            if np.sum(np.isfinite(x) & np.isfinite(y)) < 3:
                return (np.nan, np.nan)
            r, p = spearmanr(x, y, nan_policy="omit")
            return (float(r), float(p))
        r11, p11 = safe_spear(sub.loc[m1, COL_PC1].values,  sub.loc[m1, "delta_"+COL_PASS1].values)
        r12, p12 = safe_spear(sub.loc[m2, COL_PC2].values,  sub.loc[m2, "delta_"+COL_PASS1].values)
        r21, p21 = safe_spear(sub.loc[m3, COL_PC1].values,  sub.loc[m3, "delta_"+COL_PASS10].values)
        r22, p22 = safe_spear(sub.loc[m4, COL_PC2].values,  sub.loc[m4, "delta_"+COL_PASS10].values)
        rows.append({
            "track": tr,
            "Spearman(PC1, Δ@1)": r11, "p(PC1, Δ@1)": p11,
            "Spearman(PC2, Δ@1)": r12, "p(PC2, Δ@1)": p12,
            "Spearman(PC1, Δ@10)": r21, "p(PC1, Δ@10)": p21,
            "Spearman(PC2, Δ@10)": r22, "p(PC2, Δ@10)": p22,
            "n(Δ@1)": int(np.sum(m1)), "n(Δ@10)": int(np.sum(m3)),
        })
    corr_df = pd.DataFrame(rows)
    p_tbl_812 = os.path.join(OUTDIR, "table_8_12_pc_corr.csv")
    corr_df.to_csv(p_tbl_812, index=False)

    # --- 完了ログ ---
    print("[OK] Wrote:")
    print(" -", p_fig_819)
    print(" -", p_fig_820)
    print(" -", p_fig_821)
    print(" -", p_tbl_812)


if __name__ == "__main__":
    main()
