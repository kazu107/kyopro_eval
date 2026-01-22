# make_fig_8_2_6boxes.py
# -*- coding: utf-8 -*-
"""
Fig 8.2 (k50 / k90): 6箱（モデル×手法= 2×3）を1枚にまとめて描画するスクリプト。
要件:
- 箱の下にモデル名は表示しない（x軸ラベル空）。
- 色: Llama=赤, DeepSeek=青。
- 右上に凡例（モデル名）を表示。
- baseline / CoT / feedback の各グループ見出しのみ x 軸下部に表示。
- k50, k90 をそれぞれ別PNGに保存。

入力優先:
1) model_compare_out/joined_all_models.csv（per-problem, 各行=1問題×1モデル×1トラック）
   - 存在しない場合はその場のディレクトリから再帰探索。
2) joined に k50/k90 列が無い場合、pass_at_1..100 から再計算（しきい値 0.5 / 0.9）。

出力:
./figs_8_2_sixboxes/
  - fig_8_2_k50_six_boxes.png
  - fig_8_2_k90_six_boxes.png
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ========= 設定（必要なら編集） =========
ROOT_DIRS = [
    Path("./model_compare_out"),
    Path("model_compare_out"),
    Path("."),
]
OUT_DIR = Path("./figs_8_1_k50_k90")
TRACK_ORDER = ["baseline", "CoT", "feedback"]   # 手法の順序
MODEL_ORDER = ["Llama-3.1-8B-Instruct", "deepseek-coder-7b-instruct-v1.5"]
MODEL_COLORS = {
    "Llama-3.1-8B-Instruct": "red",
    "deepseek-coder-7b-instruct-v1.5": "blue",
}
# =====================================

OUT_DIR.mkdir(parents=True, exist_ok=True)

def find_csv(root_dirs, name):
    for rd in root_dirs:
        if (rd / name).exists():
            return rd / name
        # 再帰探索
        for p in rd.rglob(name):
            return p
    return None

def load_joined():
    p = find_csv(ROOT_DIRS, "joined_all_models.csv")
    if p is None:
        raise FileNotFoundError("joined_all_models.csv が見つかりません。make_model_compare.py を先に実行してください。")
    df = pd.read_csv(p)
    # 最低限の必須列
    need = {"problem_id","track","model"}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"joined_all_models.csv に必須列が不足: {miss}")
    return df

def k_from_pass_row(row, thr: float) -> float:
    # pass_at_1..100 から「初めて >=thr となる k」を返す
    ks = []
    for c in row.index:
        if isinstance(c, str) and c.startswith("pass_at_"):
            tail = c.split("_")[-1]
            if tail.isdigit():
                ks.append(int(tail))
    if not ks:
        return np.nan
    ks = sorted(set(ks))
    for k in ks:
        if row.get(f"pass_at_{k}", np.nan) >= thr:
            return float(k)
    return np.nan

def ensure_kcols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    has_k50 = "k50" in df.columns
    has_k90 = "k90" in df.columns
    # pass_at_* があるかチェック（再計算のため）
    pass_cols = [c for c in df.columns if isinstance(c,str) and c.startswith("pass_at_") and c.split("_")[-1].isdigit()]
    if (not has_k50 or not has_k90) and pass_cols:
        # 問題×モデル×トラック単位で計算
        if not has_k50:
            df["k50"] = df.apply(lambda r: k_from_pass_row(r, 0.5), axis=1)
        if not has_k90:
            df["k90"] = df.apply(lambda r: k_from_pass_row(r, 0.9), axis=1)
    elif not has_k50 or not has_k90:
        raise ValueError("k50/k90 列が無く、pass_at_1..100 も見つからないため再計算できません。")
    return df

def build_six_boxes_arrays(df: pd.DataFrame, kcol: str):
    """6箱分のデータ配列と、描画位置・色・グループ境界情報を返す。"""
    arrays = []   # 各箱の per-problem 値
    colors = []   # 各箱の色（モデルに応じて）
    # 配置: [baseline-Llama, baseline-DeepSeek, CoT-Llama, CoT-DeepSeek, feedback-Llama, feedback-DeepSeek]
    for tr in TRACK_ORDER:
        for m in MODEL_ORDER:
            sub = df[(df["track"]==tr) & (df["model"]==m)]
            x = pd.to_numeric(sub[kcol], errors="coerce").dropna().values
            arrays.append(x)
            colors.append(MODEL_COLORS.get(m, "gray"))
    return arrays, colors

def plot_six_boxes(kcol: str, title_prefix: str, outname: str, df_joined: pd.DataFrame):
    arrays, colors = build_six_boxes_arrays(df_joined, kcol)

    # 少なくともどれかにデータが必要
    if all((len(a)==0 for a in arrays)):
        raise ValueError(f"{kcol}: 箱ひげに使える per-problem データが見つかりません。")

    # プロット
    fig = plt.figure()
    bp = plt.boxplot(arrays, patch_artist=True, showfliers=False)

    # 色・線
    for box, c in zip(bp["boxes"], colors):
        box.set(facecolor=c, edgecolor="black", linewidth=1.0)
    for whisk in bp["whiskers"]:
        whisk.set(color="black", linewidth=1.0)
    for cap in bp["caps"]:
        cap.set(color="black", linewidth=1.0)
    for med in bp["medians"]:
        med.set(color="black", linewidth=1.2)

    # x 軸はラベル無し（モデル名は出さない）
    plt.xticks([])

    # グループ境界線と手法ラベル（baseline / CoT / feedback）
    # 箱は 1..6 で配置されるので、グループ中心は 1.5, 3.5, 5.5 近辺
    group_centers = [1.5, 3.5, 5.5]
    for gx in [2.5, 4.5]:
        plt.axvline(gx, color="black", linewidth=0.8, linestyle="--", alpha=0.4)
    y_min, y_max = plt.ylim()
    for gc, lab in zip(group_centers, TRACK_ORDER):
        plt.text(gc, y_min - 0.05*(y_max-y_min), lab, ha="center", va="top")

    # 右上の凡例
    patches = [mpatches.Patch(color=MODEL_COLORS[m], label=m) for m in MODEL_ORDER]
    plt.legend(handles=patches, loc="upper right", frameon=True)

    plt.ylabel(kcol)
    plt.title(f"{title_prefix} (6 boxes: model × track)")

    fig.tight_layout()
    fig.savefig(OUT_DIR / outname, dpi=220)
    plt.close(fig)

def main():
    joined = load_joined()
    joined = ensure_kcols(joined)

    # k50
    plot_six_boxes(
        kcol="k50",
        title_prefix="Fig 8.2 k50",
        outname="fig_8_2_k50_six_boxes.png",
        df_joined=joined
    )
    # k90
    plot_six_boxes(
        kcol="k90",
        title_prefix="Fig 8.2 k90",
        outname="fig_8_2_k90_six_boxes.png",
        df_joined=joined
    )
    print("出力先:", OUT_DIR.resolve())
    print("生成:", "fig_8_2_k50_six_boxes.png, fig_8_2_k90_six_boxes.png")

if __name__ == "__main__":
    main()
