# k50/k90 箱ひげ（CIベース）比較図を作るスクリプト
# - 入力: Llama と DeepSeek の 2 CSV（質問に貼られた形式）
# - 出力: k50_k90_box_ci.png（2面：左 k50, 右 k90。各面に hand=baseline/CoT/feedback、それぞれ Llama/DeepSeek の箱を並置）
#
# 使い方:
# 1) 質問文の「Llama: ～」部分を llama_stats.csv に、「DeepSeek: ～」部分を deepseek_stats.csv に保存
# 2) 下のパスを編集して実行

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ===== 設定（ここだけ編集） =====
LLAMA_CSV    = "./Llama-3.1-8B-Instruct/results/overall_summary.csv"       # Llama のCSV
DEEPSEEK_CSV = "./deepseek-coder-7b-instruct-v1.5/results/overall_summary.csv"    # DeepSeek のCSV
OUT_PNG      = "k50_k90_box_ci.png"

# ===== データ読み込み =====
def load_stats(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # 列名の揺れ対策
    need = ["track","metric","median","median_ci_lo","median_ci_hi"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: 必須列が見つかりません: {missing}")
    # 型を数値化
    for c in ["median","median_ci_lo","median_ci_hi"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # k50/k90 のみ
    df = df[df["metric"].isin(["k50","k90"])].copy()
    return df

llama = load_stats(LLAMA_CSV)
deep  = load_stats(DEEPSEEK_CSV)

# 手法の並びを固定
order_tracks = ["baseline","CoT","feedback"]
def order_category(s):
    cats = [t for t in order_tracks if t in s.unique()] + [t for t in s.unique() if t not in order_tracks]
    return pd.Categorical(s, categories=cats, ordered=True)

llama["track"] = order_category(llama["track"])
deep["track"]  = order_category(deep["track"])

# ===== 図の準備 =====
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)  # 左:k50 右:k90

def draw_ci_boxes(ax, metric: str):
    # metric = "k50" or "k90"
    dfL = llama[llama["metric"] == metric].dropna(subset=["median","median_ci_lo","median_ci_hi"])
    dfD = deep [deep ["metric"] == metric].dropna(subset=["median","median_ci_lo","median_ci_hi"])

    # x位置：手法ごとに1.0間隔、Llama/DeepSeek のオフセットで並置
    tracks = [t for t in order_tracks if (t in dfL["track"].astype(str).tolist()) or (t in dfD["track"].astype(str).tolist())]
    x_base = np.arange(len(tracks), dtype=float)
    offset = 0.15  # Llama と DeepSeek の横ずれ
    width  = 0.22  # 箱の横幅

    # ルック（色は環境デフォルトに任せる）
    def draw_one(df, x_shift, label):
        for i, tr in enumerate(tracks):
            row = df[df["track"].astype(str) == tr]
            if row.empty:
                continue
            med = float(row["median"].iloc[0])
            lo  = float(row["median_ci_lo"].iloc[0])
            hi  = float(row["median_ci_hi"].iloc[0])
            # 箱（CI）矩形
            x0 = x_base[i] + x_shift - width/2
            ax.add_patch(plt.Rectangle((x0, lo), width, hi - lo, fill=False))
            # 中央線
            ax.plot([x_base[i] + x_shift - width/2, x_base[i] + x_shift + width/2], [med, med])
            # ヒゲ（CI端を強調）
            ax.plot([x_base[i] + x_shift, x_base[i] + x_shift], [lo, lo], marker="_")
            ax.plot([x_base[i] + x_shift, x_base[i] + x_shift], [hi, hi], marker="_")
        # 凡例用ダミー
        ax.plot([], [], label=label)

    draw_one(dfL, -offset, "Llama")
    draw_one(dfD,  +offset, "DeepSeek")

    ax.set_title(metric.upper())
    ax.set_xticks(x_base, tracks)
    ax.set_xlabel("Track")
    ax.set_ylabel("k value (trials)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", alpha=0.3)

draw_ci_boxes(axes[0], "k50")
draw_ci_boxes(axes[1], "k90")

fig.suptitle("k50 / k90 by track (CI boxes: box = 95% CI, line = median)")
fig.tight_layout()
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved -> {OUT_PNG}")
