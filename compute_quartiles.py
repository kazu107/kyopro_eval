#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pandas as pd
import numpy as np

# ============== CONFIG ==============
# 入力CSV（例：difficulty_metrics_all_axes_baseline_only.csv）
INPUT_CSV = "./results/difficulty_metrics_all_axes_baseline_only.csv"
# 出力先
OUTPUT_ALL_NUMERIC = "./results/quartiles_all_numeric.csv"
# z軸だけ別出力（列が存在する場合）
OUTPUT_Z_AXES = "./results/quartiles_z_axes.csv"
# 対象列：None の場合は CSV 内の**数値列すべて**
TARGET_COLUMNS = None  # 例: ["B_log","T_log1p","H_diff","WA_prob","z_B","z_T","z_H","z_WA"]
# ===================================

def quartile_summary(series: pd.Series) -> dict:
    s = pd.to_numeric(series, errors="coerce")
    s = s[np.isfinite(s)]
    if s.empty:
        return {
            "count": 0, "mean": np.nan, "std": np.nan,
            "min": np.nan, "q25": np.nan, "median": np.nan, "q75": np.nan, "max": np.nan
        }
    q = np.percentile(s, [0, 25, 50, 75, 100])
    return {
        "count": int(s.size),
        "mean": float(np.mean(s)),
        "std": float(np.std(s, ddof=0)),
        "min": float(q[0]),
        "q25": float(q[1]),
        "median": float(q[2]),
        "q75": float(q[3]),
        "max": float(q[4]),
    }

def main():
    if not os.path.isfile(INPUT_CSV):
        raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV} (このパスをコード先頭で修正してください)")

    df = pd.read_csv(INPUT_CSV)
    if TARGET_COLUMNS is None:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    else:
        num_cols = [c for c in TARGET_COLUMNS if c in df.columns]

    rows = []
    for col in num_cols:
        stat = quartile_summary(df[col])
        stat["column"] = col
        rows.append(stat)
    df_all = pd.DataFrame(rows)[["column","count","mean","std","min","q25","median","q75","max"]].sort_values("column")
    df_all.to_csv(OUTPUT_ALL_NUMERIC, index=False, encoding="utf-8")
    print(f"[OK] wrote {OUTPUT_ALL_NUMERIC} ({len(df_all)} columns)")

    z_axis_cols = [c for c in ["z_B","z_T","z_H","z_WA"] if c in df.columns]
    if z_axis_cols:
        rows_z = []
        for col in z_axis_cols:
            stat = quartile_summary(df[col])
            stat["column"] = col
            rows_z.append(stat)
        df_z = pd.DataFrame(rows_z)[["column","count","mean","std","min","q25","median","q75","max"]]
        df_z.to_csv(OUTPUT_Z_AXES, index=False, encoding="utf-8")
        print(f"[OK] wrote {OUTPUT_Z_AXES}")
    else:
        print("[INFO] z-axis columns (z_B/z_T/z_H/z_WA) not found; skipped")

if __name__ == "__main__":
    main()
