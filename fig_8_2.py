# make_model_curves.py
# -*- coding: utf-8 -*-
"""
モデルごとに、平均の pass@k / D-pass@k 曲線を描画するスクリプト。

前提:
- make_model_compare.py などで作成された
  `model_compare_out/overall_curves_mean_and_D.csv` を入力に使う。
- 列構成は以下を想定:
  model, track, k,
  mean_pass, mean_ci_lo, mean_ci_hi,
  D_pass,   D_ci_lo,   D_ci_hi,
  n_problems

出力:
- モデルごとに PNG/SVG の図を出力する。
  - <OUT_DIR>/<model>__pass_curves.png
  - <OUT_DIR>/<model>__dpass_curves.png

描画仕様:
- 各 track ごとに平滑な曲線（平均）を描画。
- 95%CI は同色の点線で上下限を描画。
- 平均線・CI 線ともに track に応じた marker を使用。
- marker の大きさと、marker を置く k（markevery）を設定可能。
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# =========================
# 設定（ここを編集して使う）
# =========================
CURVES_CSV: str = "model_compare_out/overall_curves_mean_and_D.csv"
OUT_DIR: str = "figs_8_2_model_curves5"

# 描画するモデル（空リストの場合は CSV 内の全モデル）
TARGET_MODELS: List[str] = []  # 例: ["Llama-3.1-8B-Instruct", "deepseek-coder-7b-instruct-v1.5"]

# 描画する track とラベル
TRACK_ORDER: List[str] = ["baseline", "CoT", "feedback"]
TRACK_LABELS: Dict[str, str] = {
    "baseline": "baseline",
    "CoT":      "CoT",
    "feedback": "feedback",
}

# track ごとの色（カラーコード指定）
TRACK_COLORS: Dict[str, str] = {
    "baseline": "#1f77b4",  # 青
    "CoT":      "#ff7f0e",  # オレンジ
    "feedback": "#2ca02c",  # 緑
}

# track ごとの線種（平均曲線用）
TRACK_LINESTYLES: Dict[str, str] = {
    "baseline": "-",
    "CoT":      "-",
    "feedback": "-",
}

# track ごとのマーカー（平均 & CI 線に適用）
TRACK_MARKERS: Dict[str, str] = {
    "baseline": "o",
    "CoT":      "s",
    "feedback": "D",
}

# marker の大きさ（全 track 共通）
MARKER_SIZE: float = 2.5

# marker を配置する k のリスト
# 空リストの場合: markevery=None（全点に marker）
# 例: [1, 2, 5, 10, 20, 50, 100]
MARKER_K_LIST: List[int] = [1, 2, 5, 10, 20, 50, 100]

# k の範囲制限（None なら全て使う）
K_MIN: int = 1
K_MAX: int = 100

# x軸スケール: "linear" または "log"
X_AXIS_SCALE: str = "log"  # "linear" or "log"

# x軸が log のときの目盛り（None の場合は matplotlib デフォルト）
LOG_X_TICKS: List[int] = [1, 2, 5, 10, 20, 50, 100]

# CI 線のスタイル（上限と下限で違う種類にする）
CI_LOWER_LS = "--"  # 下限
CI_UPPER_LS = ":"   # 上限

# 軸・見た目関係
Y_LIM_PASS = (0.6, 1.00)   # pass@k の y 軸範囲
Y_LIM_DPASS = (0.6, 1.00)  # D-pass@k の y 軸範囲
FIG_SIZE = (6.0, 4.0)      # 図のサイズ (inch)
DPI = 600                  # 解像度
SHOW_LEGEND = True
LEGEND_LOC = "lower right"
GRID = True

# フォントなど（必要なら）
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False  # 日本語環境でマイナスが化ける対策


# =========================
# 関数
# =========================

def load_curves(csv_path: str) -> pd.DataFrame:
    """overall_curves_mean_and_D.csv を読み込む。"""
    df = pd.read_csv(csv_path)
    # k範囲でフィルタ
    if K_MIN is not None:
        df = df[df["k"] >= K_MIN]
    if K_MAX is not None:
        df = df[df["k"] <= K_MAX]
    # log 軸用に k>0 を保証
    df = df[df["k"] > 0]
    return df


def get_target_models(df: pd.DataFrame) -> List[str]:
    """描画対象モデルのリストを返す。"""
    if TARGET_MODELS:
        return TARGET_MODELS
    return sorted(df["model"].unique().tolist())


def compute_markevery(k_values: np.ndarray) -> List[int] | None:
    """
    MARKER_K_LIST に基づき markevery（インデックスのリスト）を返す。
    - MARKER_K_LIST が空の場合は None（=全点に marker）。
    - 一致する k が1つもない場合は []（=marker なし）を返す。
    """
    if not MARKER_K_LIST:
        return None  # 全点
    idx = [i for i, kv in enumerate(k_values) if int(kv) in MARKER_K_LIST]
    return idx  # 空なら marker は実質なし


def plot_metric_curves_for_model(
    df: pd.DataFrame,
    model_name: str,
    metric: str,
    ci_lo_col: str,
    ci_hi_col: str,
    curve_label: str,   # "pass@k" or "D-pass@k"
    y_label: str,
    y_lim: tuple,
    out_path: Path,
) -> None:
    """
    1つのモデルについて、指定した指標（mean_pass / D_pass）を
    track ごとに k 対応曲線として描画する。
    CI は同色の点線で上下限を描画し、凡例には
    CI lower / CI upper のスタイルも追加する。
    marker は平均線・CI 線の両方に適用される。
    """
    df_m = df[df["model"] == model_name].copy()
    if df_m.empty:
        print(f"[WARN] model={model_name} のデータがありません。スキップします。")
        return

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=DPI)

    track_handles: List[Line2D] = []

    for track in TRACK_ORDER:
        df_mt = df_m[df_m["track"] == track].copy()
        if df_mt.empty:
            continue
        df_mt = df_mt.sort_values("k")

        color = TRACK_COLORS.get(track, "#000000")
        label = TRACK_LABELS.get(track, track)
        ls_mean = TRACK_LINESTYLES.get(track, "-")
        marker = TRACK_MARKERS.get(track, "")

        k_vals = df_mt["k"].to_numpy()
        markevery = compute_markevery(k_vals)

        # 平均曲線（線種・マーカー指定）
        line_mean, = ax.plot(
            df_mt["k"],
            df_mt[metric],
            color=color,
            linestyle=ls_mean,
            marker=marker if marker else None,
            markersize=MARKER_SIZE if marker else None,
            markevery=markevery,
            linewidth=2.0,
            label=label,
        )
        track_handles.append(line_mean)

        # 95%CI の下限（点線その1）
        if ci_lo_col in df_mt.columns:
            ax.plot(
                df_mt["k"],
                df_mt[ci_lo_col],
                color=color,
                linestyle=CI_LOWER_LS,
                marker=marker if marker else None,
                markersize=MARKER_SIZE if marker else None,
                markevery=markevery,
                linewidth=1.0,
            )

        # 95%CI の上限（点線その2）
        if ci_hi_col in df_mt.columns:
            ax.plot(
                df_mt["k"],
                df_mt[ci_hi_col],
                color=color,
                linestyle=CI_UPPER_LS,
                marker=marker if marker else None,
                markersize=MARKER_SIZE if marker else None,
                markevery=markevery,
                linewidth=1.0,
            )

    # x軸スケール設定
    if X_AXIS_SCALE == "log":
        ax.set_xscale("log")  # base=10 デフォルト
        if LOG_X_TICKS:
            ticks = [t for t in LOG_X_TICKS if (K_MIN or 0) <= t <= (K_MAX or t)]
            if ticks:
                ax.set_xticks(ticks)
    else:
        ax.set_xscale("linear")

    # タイトル: {model} - {curve}
    # ax.set_title(f"{model_name} - {curve_label}")
    ax.set_xlabel("k")
    ax.set_ylabel(y_label)
    if y_lim is not None:
        ax.set_ylim(*y_lim)
    if GRID:
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)

    # 凡例（右下）: 先頭に CI lower / CI upper、その後に track ごとのラベル
    if SHOW_LEGEND and track_handles:
        ci_lower_handle = Line2D(
            [0], [0],
            color="black",
            linestyle=CI_LOWER_LS,
            linewidth=1.0,
            label="CI lower",
        )
        ci_upper_handle = Line2D(
            [0], [0],
            color="black",
            linestyle=CI_UPPER_LS,
            linewidth=1.0,
            label="CI upper",
        )
        handles = [ci_lower_handle, ci_upper_handle] + track_handles
        labels = [h.get_label() for h in handles]
        ax.legend(handles, labels, loc=LEGEND_LOC)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    svg_path = out_path.with_suffix(".svg")
    fig.savefig(svg_path)
    plt.close(fig)

    print(f"[OK] saved: {out_path} / {svg_path}")


def main():
    outdir = Path(OUT_DIR)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_curves(CURVES_CSV)
    models = get_target_models(df)

    for model in models:
        safe_model = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in model)

        # 平均 pass@k 曲線
        pass_out = outdir / f"{safe_model}__pass_curves.png"
        plot_metric_curves_for_model(
            df=df,
            model_name=model,
            metric="mean_pass",
            ci_lo_col="mean_ci_lo",
            ci_hi_col="mean_ci_hi",
            curve_label="pass@k",
            y_label="pass@k",
            y_lim=Y_LIM_PASS,
            out_path=pass_out,
        )

        # D-pass@k 曲線
        dpass_out = outdir / f"{safe_model}__dpass_curves.png"
        plot_metric_curves_for_model(
            df=df,
            model_name=model,
            metric="D_pass",
            ci_lo_col="D_ci_lo",
            ci_hi_col="D_ci_hi",
            curve_label="D-pass@k",
            y_label="D-pass@k",
            y_lim=Y_LIM_DPASS,
            out_path=dpass_out,
        )


if __name__ == "__main__":
    main()
