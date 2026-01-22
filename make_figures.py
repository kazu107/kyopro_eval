# Make figures for the paper from results/*.csv — y-lims matched & per-track colors (no CLI args)
# -----------------------------------------------------------------------------------------------
# - matplotlib only (no seaborn), one chart per figure.
# - Per-track Mean/D charts:
#     * y-limits are matched to the corresponding overall chart (configurable).
#     * line colors are forced to be the SAME as the overall chart's track colors.
#
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =============================
# CONFIG — EDIT FREELY
# =============================
CONFIG = {
    # I/O
    "RESULTS_DIR": "deepseek-coder-7b-instruct-v1.5/results",
    "FIG_DIR": None,              # None => RESULTS_DIR/figs
    "SAVE_DPI": 200,

    # General plot settings
    "LEGEND_LOC": "best",
    "FILL_ALPHA": 0.20,
    "IQR_ALPHA": 0.15,
    "BAR_ERR_CAPSIZE": 4,

    # Axis policy
    "AXIS": {
        "MATCH_PER_TRACK_YLIM_TO_OVERALL": True,  # 個別図の縦軸を overall と揃える
        "CLAMP_01": False,                          # 確率なら [0,1] に固定
        "PAD_FRAC": 0.02,                          # 全体 y 範囲に追加パディング（比率）
    },

    # Colors policy
    #  - TRACK_COLORS: 明示指定（例: {"baseline":"#1f77b4","CoT":"#ff7f0e","feedback":"#2ca02c"})
    #  - 空なら rcParams の色サイクルから tracks の順で割り当てます（overall/個別で共通化）
    "TRACK_COLORS": {},

    # Human-friendly names for tracks (left blank to use raw names)
    "TRACK_DISPLAY": {
        "baseline": "Baseline",
        "CoT": "CoT",
        "feedback": "Feedback",
    },
    # Optional fixed order of tracks (use [] for auto)
    "TRACK_ORDER": [],

    # Toggle figures ON/OFF
    "GENERATE": {
        "overall_mean_pass_curve": True,
        "overall_d_pass_curve": True,
        "per_track_mean_pass_curve": True,   # ← 個別 Mean
        "per_track_d_pass_curve": True,      # ← 個別 D

        "bar_mean_pass_at_1": True,
        "bar_mean_pass_at_10": True,
        "bar_mean_pass_at_100": True,
        "bar_d_pass_at_1": True,
        "bar_d_pass_at_10": True,
        "bar_d_pass_at_100": True,
        "bar_auc_logk": True,
        "bar_k50_median": True,
        "bar_k90_median": True,
        "spaghetti": True,
        "per_problem_curves": True,
        "attempt_labels_stacked": True,
        "pairwise_tests_summary": True,
    },

    # Titles / labels / filenames per figure
    "FIG_META": {
        "overall_mean_pass_curve": {
            "title": "Overall Mean-pass@k (95% CI)",
            "xlabel": "k (log scale)",
            "ylabel": "Mean-pass@k",
            "filename": "overall_mean_pass_curve.png",
            "log_x": True,
        },
        "overall_d_pass_curve": {
            "title": "Overall D-pass@k (95% CI)",
            "xlabel": "k (log scale)",
            "ylabel": "D-pass@k",
            "filename": "overall_d_pass_curve.png",
            "log_x": True,
        },

        "per_track_mean_pass_curve": {
            "title_template": "{track_label}: Mean-pass@k (95% CI)",
            "xlabel": "k (log scale)",
            "ylabel": "Mean-pass@k",
            "filename_template": "pertrack_mean_{track}.png",
            "log_x": True,
        },
        "per_track_d_pass_curve": {
            "title_template": "{track_label}: D-pass@k (95% CI)",
            "xlabel": "k (log scale)",
            "ylabel": "D-pass@k",
            "filename_template": "pertrack_d_{track}.png",
            "log_x": True,
        },

        "bar_mean_pass_at_1":   {"title":"pass@1 (Mean, 95% CI)",   "ylabel":"Mean-pass@1",   "filename":"bar_mean_pass_at_1.png"},
        "bar_mean_pass_at_10":  {"title":"pass@10 (Mean, 95% CI)",  "ylabel":"Mean-pass@10",  "filename":"bar_mean_pass_at_10.png"},
        "bar_mean_pass_at_100": {"title":"pass@100 (Mean, 95% CI)", "ylabel":"Mean-pass@100", "filename":"bar_mean_pass_at_100.png"},

        "bar_d_pass_at_1":   {"title":"pass@1 (D-weighted, 95% CI)","ylabel":"D-pass@1",   "filename":"bar_d_pass_at_1.png"},
        "bar_d_pass_at_10":  {"title":"pass@10 (D-weighted, 95% CI)","ylabel":"D-pass@10","filename":"bar_d_pass_at_10.png"},
        "bar_d_pass_at_100": {"title":"pass@100 (D-weighted, 95% CI)","ylabel":"D-pass@100","filename":"bar_d_pass_at_100.png"},

        "bar_auc_logk": {"title":"AUC (log-k) (Mean, 95% CI)","ylabel":"AUC (log-k)","filename":"bar_auc_logk.png"},
        "bar_k50_median": {"title":"k50 (Median, 95% CI)","ylabel":"k50","filename":"bar_k50_median.png"},
        "bar_k90_median": {"title":"k90 (Median, 95% CI)","ylabel":"k90","filename":"bar_k90_median.png"},

        "spaghetti": {
            "title_template": "Spaghetti (all problems_llama) — {track_label}",
            "xlabel": "k (log scale)",
            "ylabel": "pass@k",
            "filename_template": "spaghetti_{track}.png",
            "log_x": True,
        },
        "per_problem_curves": {
            "title_template": "{pid}: pass@k",
            "xlabel": "k (log scale)",
            "ylabel": "pass@k",
            "filename_template": "problem_{pid}_curves.png",
            "log_x": True,
            "MAX_PROBLEMS": None,   # None or int
        },
        "attempt_labels_stacked": {"title":"Attempt label distribution (stacked)","ylabel":"attempt count","filename":"attempt_labels_stacked.png"},
        "pairwise_tests_summary": {"heading":"Pairwise tests (@1/@10/@100/AUC):","filename":"pairwise_tests_summary.txt"},
    },
}
# =============================

def _save_fig(path, cfg):
    plt.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=cfg["SAVE_DPI"])
    plt.close()
    print(f"[saved] {path}")

def _read_csv_safe(path):
    if os.path.isfile(path):
        return pd.read_csv(path)
    print(f"[warn] missing: {path}".format(path))
    return None

def _tracks_from(df, col="track"):
    if df is None or col not in df.columns:
        return []
    return sorted(df[col].dropna().unique().tolist())

def _track_label(raw, cfg):
    disp = cfg["TRACK_DISPLAY"].get(raw, None)
    return disp if disp else raw

def _order_tracks(tracks, cfg):
    order = cfg.get("TRACK_ORDER") or []
    if not order:
        return tracks
    seen = set(); out = []
    for t in order:
        if t in tracks and t not in seen:
            out.append(t); seen.add(t)
    for t in tracks:
        if t not in seen:
            out.append(t)
    return out

# ---- colors ----
def _build_track_colors(tracks, cfg):
    # explicit mapping takes precedence
    explicit = cfg.get("TRACK_COLORS", {}) or {}
    colors = {}
    # base palette from rcParams
    base = plt.rcParams.get("axes.prop_cycle", None)
    base_list = []
    if base is not None:
        try:
            base_list = base.by_key().get("color", [])
        except Exception:
            base_list = []
    # fallback generic if needed
    if not base_list:
        base_list = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
                     "#9467bd","#8c564b","#e377c2","#7f7f7f",
                     "#bcbd22","#17becf"]
    # extend palette to cover all tracks deterministically
    extended = []
    while len(extended) < len(tracks):
        extended += base_list
    idx = 0
    for t in tracks:
        if t in explicit:
            colors[t] = explicit[t]
        else:
            colors[t] = extended[idx]
            idx += 1
    return colors

# ---- y-lim helpers ----
def _finite_min_max(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return (np.nan, np.nan)
    return (float(np.min(a)), float(np.max(a)))

def _compute_overall_ylim(df_curves, cols_value, cfg):
    col_val, col_lo, col_hi = cols_value
    if df_curves is None or df_curves.empty:
        return (0.0, 1.0) if cfg["AXIS"]["CLAMP_01"] else (0.0, 1.0)

    if col_lo in df_curves.columns and col_hi in df_curves.columns:
        ymin, ymax = _finite_min_max([
            df_curves[col_lo].min(skipna=True), df_curves[col_lo].max(skipna=True),
            df_curves[col_hi].min(skipna=True), df_curves[col_hi].max(skipna=True)
        ])
    else:
        ymin, ymax = _finite_min_max(df_curves[col_val].to_numpy(dtype=float))

    pad_frac = cfg["AXIS"]["PAD_FRAC"]
    if np.isfinite(ymin) and np.isfinite(ymax):
        span = max(ymax - ymin, 1e-6)
        ymin -= span * pad_frac
        ymax += span * pad_frac

    if cfg["AXIS"]["CLAMP_01"]:
        ymin = 0.0 if not np.isnan(ymin) else 0.0
        ymax = 1.0 if not np.isnan(ymax) else 1.0

    if not np.isfinite(ymin) or not np.isfinite(ymax) or (ymax - ymin) < 1e-6:
        ymin, ymax = (0.0, 1.0) if cfg["AXIS"]["CLAMP_01"] else (0.0, 1.0)
    return (ymin, ymax)

def main():
    cfg = CONFIG
    RESULTS_DIR = cfg["RESULTS_DIR"]
    FIG_DIR = cfg["FIG_DIR"] or os.path.join(RESULTS_DIR, "figs")
    os.makedirs(FIG_DIR, exist_ok=True)

    df_per = _read_csv_safe(os.path.join(RESULTS_DIR, "per_problem_pass_at_k.csv"))
    df_curves = _read_csv_safe(os.path.join(RESULTS_DIR, "overall_curves.csv"))
    df_summary = _read_csv_safe(os.path.join(RESULTS_DIR, "overall_summary.csv"))
    df_labels = _read_csv_safe(os.path.join(RESULTS_DIR, "attempt_label_counts.csv"))
    df_tests = _read_csv_safe(os.path.join(RESULTS_DIR, "pairwise_tests.csv"))

    tracks = sorted(set(_tracks_from(df_per) + _tracks_from(df_curves) + _tracks_from(df_labels)))
    tracks = _order_tracks(tracks, cfg)
    print("[info] tracks:", tracks)

    # colors mapping (used in ALL charts)
    track_colors = _build_track_colors(tracks, cfg)
    print("[info] colors:", {t: track_colors[t] for t in tracks})

    # overall y-limits for Mean / D
    mean_ylim = _compute_overall_ylim(df_curves, ("mean_pass", "mean_pass_ci_lo", "mean_pass_ci_hi"), cfg)
    d_ylim = _compute_overall_ylim(
        df_curves if (df_curves is not None and "d_pass" in df_curves.columns) else None,
        ("d_pass", "d_pass_ci_lo", "d_pass_ci_hi"), cfg
    )
    print(f"[info] overall y-lims  Mean={mean_ylim}, D={d_ylim}")

    # 1) Overall curves (Mean) — use fixed colors per track
    if cfg["GENERATE"].get("overall_mean_pass_curve", False) and df_curves is not None and not df_curves.empty:
        meta = cfg["FIG_META"]["overall_mean_pass_curve"]
        plt.figure(figsize=(7, 4.5))
        for t in tracks:
            d = df_curves[df_curves["track"] == t]
            if d.empty: continue
            ks = d["k"].to_numpy()
            y  = d["mean_pass"].to_numpy(dtype=float)
            lo = d["mean_pass_ci_lo"].to_numpy(dtype=float)
            hi = d["mean_pass_ci_hi"].to_numpy(dtype=float)
            c = track_colors[t]
            plt.plot(ks, y, label=_track_label(t, cfg), color=c)
            plt.fill_between(ks, lo, hi, alpha=cfg["FILL_ALPHA"], facecolor=c, edgecolor=None)
        if meta.get("log_x", True): plt.xscale("log")
        plt.ylim(*mean_ylim)
        plt.xlabel(meta.get("xlabel", "k"))
        plt.ylabel(meta.get("ylabel", "Mean-pass@k"))
        plt.title(meta.get("title", "Overall Mean-pass@k"))
        plt.legend(loc=cfg["LEGEND_LOC"])
        _save_fig(os.path.join(FIG_DIR, meta["filename"]), cfg)

    # 1') Per-track Mean curves — force same colors and y-lims
    if cfg["GENERATE"].get("per_track_mean_pass_curve", False) and df_curves is not None and not df_curves.empty:
        meta = cfg["FIG_META"]["per_track_mean_pass_curve"]
        for t in tracks:
            d = df_curves[df_curves["track"] == t]
            if d.empty: continue
            ks = d["k"].to_numpy()
            y  = d["mean_pass"].to_numpy(dtype=float)
            lo = d["mean_pass_ci_lo"].to_numpy(dtype=float)
            hi = d["mean_pass_ci_hi"].to_numpy(dtype=float)
            c = track_colors[t]
            plt.figure(figsize=(7, 4.5))
            plt.plot(ks, y, color=c)
            plt.fill_between(ks, lo, hi, alpha=cfg["FILL_ALPHA"], facecolor=c, edgecolor=None)
            if meta.get("log_x", True): plt.xscale("log")
            if cfg["AXIS"]["MATCH_PER_TRACK_YLIM_TO_OVERALL"]:
                plt.ylim(*mean_ylim)
            plt.xlabel(meta.get("xlabel", "k"))
            plt.ylabel(meta.get("ylabel", "Mean-pass@k"))
            plt.title(meta.get("title_template", "{track_label}: Mean-pass@k").format(
                track=t, track_label=_track_label(t, cfg)))
            _save_fig(os.path.join(FIG_DIR, meta.get("filename_template", "pertrack_mean_{track}.png").format(track=t)), cfg)

    # 2) Overall curves (D) — use fixed colors per track
    if cfg["GENERATE"].get("overall_d_pass_curve", False) and df_curves is not None and not df_curves.empty and "d_pass" in df_curves.columns:
        meta = cfg["FIG_META"]["overall_d_pass_curve"]
        plt.figure(figsize=(7, 4.5))
        for t in tracks:
            d = df_curves[df_curves["track"] == t]
            if d.empty: continue
            ks = d["k"].to_numpy()
            y  = d["d_pass"].to_numpy(dtype=float)
            lo = d["d_pass_ci_lo"].to_numpy(dtype=float)
            hi = d["d_pass_ci_hi"].to_numpy(dtype=float)
            c = track_colors[t]
            plt.plot(ks, y, label=_track_label(t, cfg), color=c)
            plt.fill_between(ks, lo, hi, alpha=cfg["FILL_ALPHA"], facecolor=c, edgecolor=None)
        if meta.get("log_x", True): plt.xscale("log")
        plt.ylim(*d_ylim)
        plt.xlabel(meta.get("xlabel", "k"))
        plt.ylabel(meta.get("ylabel", "D-pass@k"))
        plt.title(meta.get("title", "Overall D-pass@k"))
        plt.legend(loc=cfg["LEGEND_LOC"])
        _save_fig(os.path.join(FIG_DIR, meta["filename"]), cfg)

    # 2') Per-track D curves — force same colors and y-lims
    if cfg["GENERATE"].get("per_track_d_pass_curve", False) and df_curves is not None and not df_curves.empty and "d_pass" in df_curves.columns:
        meta = cfg["FIG_META"]["per_track_d_pass_curve"]
        for t in tracks:
            d = df_curves[df_curves["track"] == t]
            if d.empty: continue
            ks = d["k"].to_numpy()
            y  = d["d_pass"].to_numpy(dtype=float)
            lo = d["d_pass_ci_lo"].to_numpy(dtype=float)
            hi = d["d_pass_ci_hi"].to_numpy(dtype=float)
            c = track_colors[t]
            plt.figure(figsize=(7, 4.5))
            plt.plot(ks, y, color=c)
            plt.fill_between(ks, lo, hi, alpha=cfg["FILL_ALPHA"], facecolor=c, edgecolor=None)
            if meta.get("log_x", True): plt.xscale("log")
            if cfg["AXIS"]["MATCH_PER_TRACK_YLIM_TO_OVERALL"]:
                plt.ylim(*d_ylim)
            plt.xlabel(meta.get("xlabel", "k"))
            plt.ylabel(meta.get("ylabel", "D-pass@k"))
            plt.title(meta.get("title_template", "{track_label}: D-pass@k").format(
                track=t, track_label=_track_label(t, cfg)))
            _save_fig(os.path.join(FIG_DIR, meta.get("filename_template", "pertrack_d_{track}.png").format(track=t)), cfg)

    # 3) Representative bars
    def _bar_with_ci(metric_name, value_col, lo_col, hi_col, meta_key):
        if df_summary is None or df_summary.empty: return
        meta = cfg["FIG_META"][meta_key]
        d = df_summary[df_summary["metric"] == metric_name]
        if d.empty: return
        plt.figure(figsize=(6, 4.2))
        x = np.arange(len(tracks))
        vals = []; los = []; his = []; labels = []
        for t in tracks:
            row = d[d["track"] == t]
            if row.empty:
                vals.append(np.nan); los.append(np.nan); his.append(np.nan)
            else:
                vals.append(float(row.iloc[0][value_col]))
                los.append(float(row.iloc[0][lo_col]))
                his.append(float(row.iloc[0][hi_col]))
            labels.append(_track_label(t, cfg))
        plt.bar(x, vals)
        errs = np.array([(u - l)/2 if (not (np.isnan(u) or np.isnan(l))) else np.nan for l,u in zip(los, his)])
        centers = np.array(vals)
        plt.errorbar(x, centers, yerr=errs, fmt='none', capsize=cfg["BAR_ERR_CAPSIZE"])
        plt.xticks(x, labels, rotation=0)
        plt.ylabel(meta.get("ylabel", metric_name))
        plt.title(meta.get("title", f"{metric_name} (95% CI)"))
        _save_fig(os.path.join(FIG_DIR, meta["filename"]), cfg)

    if CONFIG["GENERATE"].get("bar_mean_pass_at_1", False):   _bar_with_ci("pass@1", "mean", "mean_ci_lo", "mean_ci_hi", "bar_mean_pass_at_1")
    if CONFIG["GENERATE"].get("bar_mean_pass_at_10", False):  _bar_with_ci("pass@10", "mean", "mean_ci_lo", "mean_ci_hi", "bar_mean_pass_at_10")
    if CONFIG["GENERATE"].get("bar_mean_pass_at_100", False): _bar_with_ci("pass@100", "mean", "mean_ci_lo", "mean_ci_hi", "bar_mean_pass_at_100")

    if CONFIG["GENERATE"].get("bar_d_pass_at_1", False) and df_summary is not None and {"d_mean","d_mean_ci_lo","d_mean_ci_hi"}.issubset(df_summary.columns):
        _bar_with_ci("pass@1", "d_mean", "d_mean_ci_lo", "d_mean_ci_hi", "bar_d_pass_at_1")
    if CONFIG["GENERATE"].get("bar_d_pass_at_10", False) and df_summary is not None and {"d_mean","d_mean_ci_lo","d_mean_ci_hi"}.issubset(df_summary.columns):
        _bar_with_ci("pass@10", "d_mean", "d_mean_ci_lo", "d_mean_ci_hi", "bar_d_pass_at_10")
    if CONFIG["GENERATE"].get("bar_d_pass_at_100", False) and df_summary is not None and {"d_mean","d_mean_ci_lo","d_mean_ci_hi"}.issubset(df_summary.columns):
        _bar_with_ci("pass@100", "d_mean", "d_mean_ci_lo", "d_mean_ci_hi", "bar_d_pass_at_100")

    # 4) Spaghetti per track（色は median/IQR のみなので固定色不要）
    if CONFIG["GENERATE"].get("spaghetti", False) and df_per is not None and not df_per.empty:
        pass_cols = [c for c in df_per.columns if c.startswith("pass_at_")]
        ks = [int(c.split("_")[-1]) for c in pass_cols]
        order = np.argsort(ks); pass_cols = [pass_cols[i] for i in order]; ks = [ks[i] for i in order]
        meta = CONFIG["FIG_META"]["spaghetti"]
        for t in tracks:
            d = df_per[df_per["track"] == t]
            if d.empty: continue
            plt.figure(figsize=(7, 4.5))
            for _, row in d.iterrows():
                y = row[pass_cols].to_numpy(dtype=float)
                plt.plot(ks, y, alpha=0.4)
            Y = d[pass_cols].to_numpy(dtype=float)
            med = np.nanmedian(Y, axis=0)
            q1 = np.nanpercentile(Y, 25, axis=0)
            q3 = np.nanpercentile(Y, 75, axis=0)
            plt.plot(ks, med, linewidth=2.5, label="median")
            plt.fill_between(ks, q1, q3, alpha=CONFIG["IQR_ALPHA"], label="IQR")
            if meta.get("log_x", True): plt.xscale("log")
            plt.ylim(0, 1.0)
            plt.xlabel(meta.get("xlabel", "k"))
            plt.ylabel(meta.get("ylabel", "pass@k"))
            plt.title(meta.get("title_template", "Spaghetti — {track_label}").format(
                track=t, track_label=_track_label(t, CONFIG)))
            plt.legend(loc=CONFIG["LEGEND_LOC"])
            _save_fig(os.path.join(FIG_DIR, meta.get("filename_template", "spaghetti_{track}.png").format(track=t)), CONFIG)

    # 5) Per-problem curves
    if CONFIG["GENERATE"].get("per_problem_curves", False) and df_per is not None and not df_per.empty:
        pass_cols = [c for c in df_per.columns if c.startswith("pass_at_")]
        ks = [int(c.split("_")[-1]) for c in pass_cols]
        order = np.argsort(ks); pass_cols = [pass_cols[i] for i in order]; ks = [ks[i] for i in order]
        meta = CONFIG["FIG_META"]["per_problem_curves"]
        problems = sorted(df_per["problem_id"].unique().tolist())
        if isinstance(meta.get("MAX_PROBLEMS"), int): problems = problems[:meta["MAX_PROBLEMS"]]
        for pid in problems:
            plt.figure(figsize=(6, 4.2))
            for t in tracks:
                d = df_per[(df_per["problem_id"] == pid) & (df_per["track"] == t)]
                if d.empty: continue
                y = d.iloc[0][pass_cols].to_numpy(dtype=float)
                plt.plot(ks, y, label=_track_label(t, CONFIG), color=track_colors[t])
            if meta.get("log_x", True): plt.xscale("log")
            plt.ylim(0, 1.0)
            plt.xlabel(meta.get("xlabel", "k"))
            plt.ylabel(meta.get("ylabel", "pass@k"))
            plt.title(meta.get("title_template", "{pid}: pass@k").format(pid=pid))
            plt.legend(loc=CONFIG["LEGEND_LOC"])
            _save_fig(os.path.join(FIG_DIR, meta.get("filename_template", "problem_{pid}_curves.png").format(pid=pid)), CONFIG)

    # 6) Attempt labels (stacked)
    if CONFIG["GENERATE"].get("attempt_labels_stacked", False) and df_labels is not None and not df_labels.empty:
        cols = [c for c in ["AC","WA","TLE","RE","MLE","CE","OTHER"] if c in df_labels.columns]
        agg = df_labels.groupby("track")[cols].sum().reindex(tracks)
        meta = CONFIG["FIG_META"]["attempt_labels_stacked"]
        plt.figure(figsize=(7, 4.5))
        bottom = np.zeros(len(tracks))
        for col in cols:
            vals = agg[col].to_numpy(dtype=float)
            plt.bar(np.arange(len(tracks)), vals, bottom=bottom, label=col)
            bottom += vals
        labels = [_track_label(t, CONFIG) for t in tracks]
        plt.xticks(np.arange(len(tracks)), labels, rotation=0)
        plt.ylabel(meta.get("ylabel", "attempt count"))
        plt.title(meta.get("title", "Attempt label distribution"))
        plt.legend(loc=CONFIG["LEGEND_LOC"])
        _save_fig(os.path.join(FIG_DIR, meta["filename"]), CONFIG)

    # 7) Pairwise tests summary
    if CONFIG["GENERATE"].get("pairwise_tests_summary", False) and df_tests is not None and not df_tests.empty:
        meta = CONFIG["FIG_META"]["pairwise_tests_summary"]
        lines = [meta.get("heading", "Pairwise tests:")]
        for _, r in df_tests.iterrows():
            try:
                lines.append(
                    f"{r['pair']} | {r['metric']}: "
                    f"Δ={float(r['mean_diff_t2_minus_t1']):.4f} "
                    f"[{float(r['mean_diff_ci_lo']):.4f},{float(r['mean_diff_ci_hi']):.4f}], "
                    f"sign-test p={float(r['sign_test_p']):.4g}"
                )
            except Exception:
                lines.append(f"{r.get('pair','?')} | {r.get('metric','?')}: (some values missing)")
        os.makedirs(FIG_DIR, exist_ok=True)
        out_txt = os.path.join(FIG_DIR, meta.get("filename", "pairwise_tests_summary.txt"))
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[saved] {out_txt}")

if __name__ == "__main__":
    main()
