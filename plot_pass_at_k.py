# make_pass_at_k_curve_multi.py
# -*- coding: utf-8 -*-
"""
複数の overall_summary.json を読み込み、k=1..100 の pass@k 曲線を1枚に複数描画します。
「正解」判定は iter ごとに (total == passed)。

出力:
  - 各 JSON と同じディレクトリに pass_at_k.csv（個別）
  - カレントディレクトリに pass_at_k_compare.png（全曲線を重ね描き）
  - カレントディレクトリに pass_at_k_compare.csv（横に並べた比較表）
"""

import json
import math
import re
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import matplotlib.pyplot as plt

# ================== ここを自分の環境に合わせて書き換えてください ==================
JSON_PATHS: List[Path] = [
    Path("problems_llama/baseline/ABC420/outputs/overall_summary.json"),
    Path("problems_llama/feedback/ABC420/outputs/overall_summary.json"),
    Path("problems_llama/CoT/ABC420/outputs/overall_summary.json"),
]

# 凡例の表示名を JSON_PATHS と同じ順に指定（None/"" や不足は自動ラベルで補完）
LEGEND_LABELS: List[Optional[str]] = [
    "Baseline",
    "Feedback",
    "CoT",
]

# 自動ラベル使用時に (n, c) を付けるか
APPEND_STATS_TO_AUTO_LABEL = True
# ==============================================================================


def load_overall(json_or_dir: Path) -> dict:
    """overall_summary.json を読み込む。ディレクトリが渡されたらその直下を探す。"""
    p = json_or_dir
    if p.is_dir():
        p = p / "overall_summary.json"
    if not p.exists():
        raise FileNotFoundError(f"overall_summary.json が見つかりません: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_problem_dir(json_or_dir: Path, loaded: dict) -> str:
    """表示用の問題名(ディレクトリ)を決める。JSON 内の problem_dir が最優先。"""
    if "problem_dir" in loaded and loaded["problem_dir"]:
        return str(loaded["problem_dir"])
    if json_or_dir.is_dir():
        return str(json_or_dir)
    return str(json_or_dir.parent.parent)  # .../outputs/overall_summary.json の親の親を想定


def short_label(problem_dir: str) -> str:
    """凡例用の短いラベル（末尾フォルダ名など）"""
    p = Path(problem_dir)
    if p.name == "outputs" and p.parent.name:
        return p.parent.name
    if p.name:
        return p.name
    return problem_dir


def sanitize_label_for_filename(s: str) -> str:
    """ファイル名用にラベルを安全化"""
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", "_", s.strip())
    return s or "series"


def count_correct(tries: List[dict]) -> Tuple[int, int]:
    """
    total == passed を「正解」としてカウント。
    :return: (n, c) = (総 iter 数, 正解 iter 数)
    """
    n = len(tries)
    c = 0
    for t in tries:
        total = int(t.get("total", 0))
        passed = int(t.get("passed", 0))
        if total > 0 and passed == total:
            c += 1
    return n, c


def pass_at_k(n: int, c: int, k: int) -> float:
    """
    1 - C(n - c, k) / C(n, k)
    k>n のときは k=n とみなす。
    端ケース: n<=0 or k<=0 or c<=0 → 0,  c>=n → 1
    """
    if n <= 0 or k <= 0 or c <= 0:
        return 0.0
    if c >= n:
        return 1.0
    k = min(k, n)
    try:
        num = math.comb(n - c, k)
        den = math.comb(n, k)
        return 1.0 - (num / den)
    except ValueError:
        return 0.0


def compute_curve(n: int, c: int, k_min: int = 1, k_max: int = 100) -> Tuple[List[int], List[float]]:
    ks = list(range(k_min, k_max + 1))
    ys = [pass_at_k(n, c, k) for k in ks]
    return ks, ys


def save_csv_per_problem(out_csv: Path, ks: List[int], ys: List[float], meta: dict) -> None:
    lines = ["k,pass_at_k,n,c,problem_dir,label"]
    for k, y in zip(ks, ys):
        lines.append(f"{k},{y:.6f},{meta['n']},{meta['c']},{meta['problem_dir']},{meta['label']}")
    out_csv.write_text("\n".join(lines), encoding="utf-8")


def save_csv_compare(out_csv: Path, ks: List[int], series: Dict[str, List[float]]) -> None:
    # 1列目: k、以降: 各ラベル列
    headers = ["k"] + list(series.keys())
    rows = [",".join(headers)]
    for idx, k in enumerate(ks):
        row = [str(k)]
        for label in series.keys():
            y = series[label][idx] if idx < len(series[label]) else float("nan")
            row.append(f"{y:.6f}")
        rows.append(",".join(row))
    out_csv.write_text("\n".join(rows), encoding="utf-8")


def plot_compare(ks: List[int], series: Dict[str, List[float]], title: str, out_png: Path) -> None:
    plt.figure()
    # 各系列を線のみで描画（★点は描かない）
    for label, ys in series.items():
        plt.plot(ks, ys, linewidth=2, label=label)

    # x 軸は 1..100、ログ表示 + 主要目盛のみ
    plt.xscale("log")
    plt.xticks([1, 2, 5, 10, 20, 50, 100], [1, 2, 5, 10, 20, 50, 100])
    plt.ylim(0, 1)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("k")
    plt.ylabel("pass@k")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def main():
    if not JSON_PATHS:
        raise ValueError("JSON_PATHS が空です。overall_summary.json のパスを1つ以上指定してください。")

    ks_ref = list(range(1, 101))
    series: Dict[str, List[float]] = {}
    info_lines: List[str] = []

    for idx, jp in enumerate(JSON_PATHS):
        data = load_overall(jp)
        tries = data.get("tries", [])
        problem_dir = detect_problem_dir(jp, data)
        n, c = count_correct(tries)

        if n <= 0:
            print(f"[warn] tries が空です。スキップ: {jp}")
            continue

        ks, ys = compute_curve(n, c, 1, 100)

        # ラベル決定：指定があれば優先、なければ自動ラベル
        custom_label = None
        if idx < len(LEGEND_LABELS):
            custom_label = LEGEND_LABELS[idx]
        if custom_label is not None and str(custom_label).strip() != "":
            label = str(custom_label).strip()
        else:
            base = short_label(problem_dir)
            label = f"{base} (n={n}, c={c})" if APPEND_STATS_TO_AUTO_LABEL else base

        # 個別CSVは JSON と同じディレクトリに保存（ラベル名をファイル名に反映して衝突回避）
        out_dir = jp if jp.is_dir() else jp.parent
        safe_label = sanitize_label_for_filename(label)
        out_csv = out_dir / f"pass_at_k_{safe_label}.csv"
        save_csv_per_problem(out_csv, ks, ys, meta={"n": n, "c": c, "problem_dir": problem_dir, "label": label})

        series[label] = ys

        info_lines.append(
            f"[info] {label} | dir={problem_dir} | n={n}, c={c} | "
            f"pass@1={ys[0]:.4f}, pass@10={ys[9]:.4f}, pass@100={ys[99]:.4f} | CSV: {out_csv}"
        )

    if not series:
        raise ValueError("有効なデータがありませんでした。パスや JSON の中身を確認してください。")

    # 比較用のまとめPNG/CSVはカレントディレクトリに保存
    out_png = Path("pass_at_k_compare.png")
    out_csv = Path("pass_at_k_compare.csv")

    plot_compare(ks_ref, series, title="pass@k compare (k=1..100)", out_png=out_png)
    save_csv_compare(out_csv, ks_ref, series)

    # 要約出力
    print("\n".join(info_lines))
    print(f"[save] {out_png}")
    print(f"[save] {out_csv}")


if __name__ == "__main__":
    main()
