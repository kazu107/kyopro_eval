# collect_problem_metrics_bytes_time.py
# -*- coding: utf-8 -*-
"""
problems_llama/ 配下の baseline, CoT, feedback の各 ABCxxx 問題ごとに、
実装量(Bytes)と実行時間(Time)をまとめて CSV に出力するスクリプト（引数なし）。

優先して outputs/overall_summary.json を読み、そこから:
  - averages.code_bytes -> 実装量(Bytes)
  - averages.time_sec   -> 実行時間(Time, 秒)  ※TLE除外で集計済み想定
  - averages.samples.time / averages.samples.iters も付帯情報として保存

※ overall_summary.json が無い問題はスキップ。
※ すべての設定はコード内の定数で変更します。
"""

from __future__ import annotations
import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ===== 設定 ==========================================================
ROOT = Path("problems_llama")                                    # ルート
CATEGORIES = ["baseline", "CoT", "feedback"]               # 走査対象（大文字小文字を無視して一致）
ABC_PATTERN = re.compile(r"^ABC\d{3}$", re.IGNORECASE)     # ABC ディレクトリ判定
OUT_CSV = Path("problem_metrics_bytes_time.csv")           # 出力CSV
# ====================================================================


def is_category_dir(p: Path) -> bool:
    if not p.is_dir():
        return False
    name_low = p.name.lower()
    return any(name_low == c.lower() for c in CATEGORIES)


def is_abc_dir(p: Path) -> bool:
    return p.is_dir() and bool(ABC_PATTERN.match(p.name))


def read_overall_summary(abc_dir: Path) -> Optional[Dict]:
    p = abc_dir / "outputs" / "overall_summary.json"
    if not p.exists():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def extract_metrics_from_overall(loaded: Dict) -> Tuple[Optional[float], Optional[float], Optional[int], Optional[int]]:
    """
    returns: (code_bytes_avg, time_sec_avg, samples_time, samples_iters)
    """
    try:
        avg = loaded.get("averages", {})
        code_b = avg.get("code_bytes", None)
        time_s = avg.get("time_sec", None)
        samples = avg.get("samples", {}) or {}
        s_time = samples.get("time", None)
        s_iters = samples.get("iters", None)
        # 数値化できないものは None に
        code_b = float(code_b) if code_b is not None else None
        time_s = float(time_s) if time_s is not None else None
        s_time = int(s_time) if s_time is not None else None
        s_iters = int(s_iters) if s_iters is not None else None
        return code_b, time_s, s_time, s_iters
    except Exception:
        return None, None, None, None


def collect() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    if not ROOT.exists():
        raise FileNotFoundError(f"ROOT not found: {ROOT.resolve()}")

    for cat_dir in sorted([d for d in ROOT.iterdir() if is_category_dir(d)], key=lambda p: p.name.lower()):
        for abc_dir in sorted([d for d in cat_dir.iterdir() if is_abc_dir(d)], key=lambda p: p.name.lower()):
            loaded = read_overall_summary(abc_dir)
            if not loaded:
                print(f"[skip] overall_summary.json not found: {abc_dir}")
                continue

            code_b, time_s, s_time, s_iters = extract_metrics_from_overall(loaded)
            # なにも取れなければスキップ
            if code_b is None and time_s is None:
                print(f"[skip] metrics not present in overall_summary.json: {abc_dir}")
                continue

            rows.append({
                "category": cat_dir.name,
                "problem": abc_dir.name,
                "problem_path": str(abc_dir.as_posix()),
                "code_bytes_avg": f"{code_b:.0f}" if isinstance(code_b, float) else "",
                "time_sec_avg": f"{time_s:.6f}" if isinstance(time_s, float) else "",
                "samples_time": s_time if isinstance(s_time, int) else "",
                "samples_iters": s_iters if isinstance(s_iters, int) else "",
            })
            # ログ
            cb_txt = rows[-1]["code_bytes_avg"] or "NA"
            ts_txt = rows[-1]["time_sec_avg"] or "NA"
            print(f"[ok] {cat_dir.name}/{abc_dir.name}: bytes={cb_txt}, time={ts_txt}")

    return rows


def save_csv(rows: List[Dict[str, object]], out_csv: Path) -> None:
    headers = ["category", "problem", "problem_path", "code_bytes_avg", "time_sec_avg", "samples_time", "samples_iters"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def main():
    rows = collect()
    save_csv(rows, OUT_CSV)
    print(f"[save] {OUT_CSV.resolve()}")
    if not rows:
        print("[note] 有効なデータがありませんでした。overall_summary.json の有無や内容をご確認ください。")


if __name__ == "__main__":
    main()
