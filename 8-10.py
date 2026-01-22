# coding: utf-8
"""
Case-study figs/tables generator (8.25–8.27 / 8.15–8.17) – robust verdict parser付き

【不具合→原因→検証→修正パッチ】
- 症状: 図が描画されない／失敗内訳がすべて OTHER になる
- 原因候補:
  (1) problems_dir の指定が実レイアウトと不一致（例: "./Llama/problems" ではなく "./problems_llama"）
  (2) verdict.json のスキーマが想定と異なり、"verdict"/"label"/"result" 以外のキーや表記（"Accepted","OK","Time Limit Exceeded","MLE" 等）
  (3) verdict.json がネストしている or 文字列大小/表記ゆれ
- 検証: 1) 収集時に verdict.json のサンプルをログ出力、2) verdict 候補の全文検索でヒット数を表示
- 修正パッチ:
  - MODELS に problems_dir/results_dir を**個別指定**（ルート直下の "problems_llama" 等に対応）
  - verdict.json の**ロバスト抽出**: ネスト/表記ゆれ/別名キーに対応し、文字列ダンプから "AC/Accepted/OK" 等も検出
  - MLE を追加、優先度 = CE > RE > TLE > MLE > WA > AC（いずれかを検出できない場合は OTHER）

依存: pandas, numpy, matplotlib（seaborn不要）
"""

import os, re, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================
# ======== CONFIG =========
# =========================

# ★ここをあなたの実レイアウトに合わせて設定してください（例を下に記載）
MODELS = [
    {
        "name": "Llama-3.1-8B-Instruct",
        # スクショ例: ルート直下に problems_llama/baseline/ABC392/...
        "problems_dir": "./problems_llama",
        # スクショ例: ルート直下に Llama-3.1-8B-Instruct/results/...
        "results_dir":  "./Llama-3.1-8B-Instruct/results",
    },
    {
        "name": "deepseek-coder-7b-instruct-v1.5",
        "problems_dir": "./problems_deepseek",              # 例（実レイアウトに合わせて変更）
        "results_dir":  "./deepseek-coder-7b-instruct-v1.5/results",  # 例
    },
]

TRACKS = ["baseline", "CoT", "feedback"]

CASE_PROBLEMS = {
    "fig8_25": "ABC415",  # RE→AC の典型
    "fig8_26": "ABC398",  # WA が多い典型
    "fig8_27": "ABC414",  # CoT が効く典型
}

K_MAX = 100
OUTDIR = "./case_study_out"
os.makedirs(OUTDIR, exist_ok=True)

# ログ出力（問題発見用）
DEBUG_SNIFF = True
DEBUG_MAX_SAMPLES = 5

# プロット設定
plt.rcParams["figure.dpi"] = 150
plt.rcParams["font.size"] = 11

# =========================
# ===== Helper Utils ======
# =========================

def _natural_key(s: str):
    return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)', s)]

def _safe_read_text(p: Path):
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""

def _safe_load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8", errors="replace"))
    except Exception:
        return None

# verdict の優先順（“少数の致命”を優先して採択）
PRIORITY = ["CE", "RE", "TLE", "MLE", "WA", "AC"]

# verdict を文字列ダンプから検出するためのパターン（小文字で判定）
_DETECT_PATTERNS = {
    "AC":  [r"\bac\b", r"\baccepted\b", r"\bok\b", r"\bpass\b", r"\bsuccess\b"],
    "WA":  [r"\bwa\b", r"\bwrong\s*answer\b"],
    "RE":  [r"\bre\b", r"\bruntime\s*error\b", r"\bexception\b", r"\bsegmentation\s*fault\b"],
    "TLE": [r"\btle\b", r"\btime\s*limit\b", r"\btimeout\b"],
    "CE":  [r"\bce\b", r"\bcompile\s*error\b", r"\bcompilation\s*failed\b"],
    "MLE": [r"\bmle\b", r"\bmemory\s*limit\b"],
}

def _detect_labels_from_text(text: str):
    """
    verdict.json が未知スキーマでも、全文テキストから既知語をサーチして候補集合を返す。
    """
    s = text.lower()
    found = set()
    for key, pats in _DETECT_PATTERNS.items():
        for pat in pats:
            if re.search(pat, s):
                found.add(key)
                break
    return found

def _extract_label_from_json(obj) -> str | None:
    """
    verdict.json のオブジェクトから最終ラベルを抽出する（キー/ネスト/表記ゆれにロバスト）。
    優先順に基づき、複数検出時は最優先を返す。検出不能なら None。
    """
    try:
        # 1) 代表的キーを走査（深さ優先）
        stack = [obj]
        acc_texts = []
        while stack:
            cur = stack.pop()
            if isinstance(cur, dict):
                for k, v in cur.items():
                    # 候補キー名にヒットしやすくする
                    if isinstance(v, str):
                        acc_texts.append(v)
                    elif isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(cur, list):
                stack.extend(cur)
            elif isinstance(cur, str):
                acc_texts.append(cur)
        # 2) 文字列群をまとめて検出
        joined = " || ".join(acc_texts)
        found = _detect_labels_from_text(joined)
        if found:
            for cand in PRIORITY:
                if cand in found:
                    return cand
        return None
    except Exception:
        return None

def _iter_attempt_units(problem_dir: Path):
    """
    .../<track>/<ABCnnn>/outputs/ 以下から attempt 単位（=ひとつの生成コード）を列挙。
    feedback: outputs/iter_XXX/round_YYY/runs/<case>/
    baseline/CoT: outputs/iter_XXX/runs/<case>/
    """
    outputs = problem_dir / "outputs"
    if not outputs.exists():
        return []

    attempts = []
    for iter_dir in sorted(outputs.glob("iter_*"), key=lambda p: _natural_key(p.name)):
        rounds = sorted(iter_dir.glob("round_*"), key=lambda p: _natural_key(p.name))
        if rounds:
            for rdir in rounds:
                runs_root = rdir / "runs"
                if not runs_root.exists():
                    continue
                run_dirs = sorted([d for d in runs_root.glob("*") if d.is_dir()], key=lambda p: _natural_key(p.name))
                verdicts = [d / "verdict.json" for d in run_dirs]
                verdicts = [v for v in verdicts if v.exists()]
                if verdicts:
                    attempts.append({"key": (iter_dir.name, rdir.name), "run_dirs": run_dirs})
        else:
            runs_root = iter_dir / "runs"
            if not runs_root.exists():
                continue
            run_dirs = sorted([d for d in runs_root.glob("*") if d.is_dir()], key=lambda p: _natural_key(p.name))
            verdicts = [d / "verdict.json" for d in run_dirs]
            verdicts = [v for v in verdicts if v.exists()]
            if verdicts:
                attempts.append({"key": (iter_dir.name, None), "run_dirs": run_dirs})
    return attempts

def _attempt_verdict(run_dirs, sniff_log_prefix=None):
    """
    1 attempt（=ひとつの生成コード）で全ケースの verdict.json を集約し、最終 verdict を返す。
    優先順: CE > RE > TLE > MLE > WA > AC
    """
    seen = set()
    sniffed = 0

    for r in run_dirs:
        vpath = r / "verdict.json"
        obj = _safe_load_json(vpath)
        label = None
        if obj is not None:
            label = _extract_label_from_json(obj)

        if label is None:
            # JSON パース不可 or 既知キーが無い → テキスト全文から検出
            text = _safe_read_text(vpath)
            cand = _detect_labels_from_text(text)
            if cand:
                # 優先順で1つ採択
                for c in PRIORITY:
                    if c in cand:
                        label = c
                        break

        if label is None:
            label = "OTHER"

        seen.add(label)

        # デバッグ出力
        if DEBUG_SNIFF and sniffed < DEBUG_MAX_SAMPLES and sniff_log_prefix:
            print(f"[SNIFF] {sniff_log_prefix} :: {vpath} -> {label}")
            sniffed += 1

    # attempt verdict の決定（優先順）
    for cand in PRIORITY:
        if cand in seen:
            return cand
    if "OTHER" in seen:
        return "OTHER"
    # 一つも拾えない場合
    return "OTHER"

def build_timeline_and_counts(problems_dir: Path, track: str, problem_id: str, sniff_tag=None):
    """
    問題の attempt タイムライン（AC/WA/RE/TLE/CE/MLE/OTHER）と失敗カウント。
    """
    prob_dir = problems_dir / track / problem_id
    attempts = _iter_attempt_units(prob_dir)
    timeline = []
    for a in attempts:
        tag = f"{problem_id}/{track}/{a['key'][0]}/{a['key'][1] or '-'}" if DEBUG_SNIFF else None
        verdict = _attempt_verdict(a["run_dirs"], sniff_log_prefix=tag if sniff_tag else None)
        timeline.append(verdict)

    counts = {k: 0 for k in ["AC","WA","RE","TLE","CE","MLE","OTHER"]}
    for v in timeline:
        counts[v] = counts.get(v, 0) + 1
    return timeline, counts, (len(timeline) > 0)

def pass_curve_from_timeline(timeline, k_max=100):
    """
    単一問題の attempt タイムラインから pass@k（0/1）を作る（ACが出たら以降1）。
    """
    if len(timeline) == 0:
        return pd.Series([np.nan]*k_max, index=np.arange(1, k_max+1), dtype=float)
    out = []
    seen_ac = False
    for k in range(1, min(k_max, len(timeline))+1):
        if timeline[k-1] == "AC":
            seen_ac = True
        out.append(1.0 if seen_ac else 0.0)
    # 足りない分は最後の値で延長
    if len(out) < k_max:
        out += [out[-1]] * (k_max - len(out))
    return pd.Series(out, index=np.arange(1, k_max+1), dtype=float)

def load_per_problem_csv_pass(results_dir: Path):
    """
    results_dir/per_problem_pass_at_k.csv を tidy で返す（なければ None）。
    """
    if not results_dir:
        return None
    p = results_dir / "per_problem_pass_at_k.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    pass_cols = [c for c in df.columns if re.fullmatch(r"pass_at_\d+", c)]
    if not pass_cols:
        pass_cols = [c for c in df.columns if c.startswith("pass_at_")]
    if not pass_cols:
        return None
    m = df.melt(id_vars=[c for c in df.columns if c not in pass_cols],
                value_vars=pass_cols,
                var_name="k_col",
                value_name="pass_value")
    m["k"] = m["k_col"].str.replace("pass_at_", "", regex=False).astype(int)
    return m

def ensure_pass_curve(model: dict, track: str, problem_id: str, k_max=100):
    """
    raw → CSV フォールバックの順で pass 曲線を取得。
    """
    probs_dir = Path(model["problems_dir"])
    timeline, _, ok = build_timeline_and_counts(probs_dir, track, problem_id, sniff_tag=True)
    if ok:
        return pass_curve_from_timeline(timeline, k_max=k_max)

    df = load_per_problem_csv_pass(Path(model["results_dir"])) if model.get("results_dir") else None
    if df is None:
        return pd.Series([np.nan]*k_max, index=np.arange(1, k_max+1), dtype=float)
    sub = df[(df["track"]==track) & (df["problem_id"]==problem_id)]
    if sub.empty:
        return pd.Series([np.nan]*k_max, index=np.arange(1, k_max+1), dtype=float)
    cur = sub.set_index("k").sort_index()["pass_value"].astype(float)
    out = pd.Series(index=np.arange(1, k_max+1), dtype=float)
    for k in out.index:
        out.loc[k] = cur[k] if k in cur.index else (cur[cur.index<=k].max() if len(cur)>0 else np.nan)
    return out

def failure_table_for_problem(model: dict, problem_id: str):
    """
    3トラック×失敗内訳＋pass@1/10/100 をまとめた DataFrame を返す。
    raw が拾えない場合は pass@k のみ results_dir CSV から補完し、失敗内訳は NaN。
    """
    rows = []
    perprob_df = load_per_problem_csv_pass(Path(model["results_dir"])) if model.get("results_dir") else None
    probs_dir = Path(model["problems_dir"])

    for track in TRACKS:
        timeline, counts, ok = build_timeline_and_counts(probs_dir, track, problem_id)
        pc = pass_curve_from_timeline(timeline, k_max=K_MAX) if ok else None
        if (not ok) and (perprob_df is not None):
            sub = perprob_df[(perprob_df["track"]==track) & (perprob_df["problem_id"]==problem_id)]
            if not sub.empty:
                pc = sub.set_index("k").sort_index()["pass_value"].astype(float)

        def _get_pass(pc_like, k):
            if pc_like is None: return np.nan
            try:
                if isinstance(pc_like, pd.Series):
                    if k in pc_like.index:
                        return float(pc_like.loc[k])
                    prev = pc_like[pc_like.index<=k]
                    return float(prev.max()) if len(prev)>0 else np.nan
                return float("nan")
            except Exception:
                return np.nan

        row = {
            "track": track,
            "attempts": int(sum(counts.values())) if ok else np.nan,
            "AC": counts.get("AC", np.nan) if ok else np.nan,
            "WA": counts.get("WA", np.nan) if ok else np.nan,
            "RE": counts.get("RE", np.nan) if ok else np.nan,
            "TLE": counts.get("TLE", np.nan) if ok else np.nan,
            "CE": counts.get("CE", np.nan) if ok else np.nan,
            "MLE": counts.get("MLE", np.nan) if ok else np.nan,
            "OTHER": counts.get("OTHER", np.nan) if ok else np.nan,
            "pass@1": _get_pass(pc, 1),
            "pass@10": _get_pass(pc, 10),
            "pass@100": _get_pass(pc, 100),
        }
        rows.append(row)
    out = pd.DataFrame(rows)
    out.insert(0, "model", model["name"])
    out.insert(1, "problem_id", problem_id)
    return out

# =========================
# ========= FIGS ==========
# =========================

def make_fig8_25_pass_curves():
    prob = CASE_PROBLEMS["fig8_25"]
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7, 9), sharex=True)
    for i, track in enumerate(TRACKS):
        ax = axes[i]
        for m in MODELS:
            curve = ensure_pass_curve(m, track, prob, k_max=K_MAX)
            ax.plot(curve.index, curve.values, label=m["name"], linewidth=2)
        ax.set_title(f"{prob} – Pass@k (track: {track})")
        ax.set_ylabel("Pass probability (0/1)")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        if i == len(TRACKS)-1:
            ax.set_xlabel("k (number of attempts)")
        ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    out = Path(OUTDIR) / f"fig8_25_{prob}_pass_curves.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"[SAVE] {out}")

COLOR_MAP = {
    "AC":  "#2ca02c",
    "WA":  "#1f77b4",
    "RE":  "#d62728",
    "TLE": "#ff7f0e",
    "CE":  "#9467bd",
    "MLE": "#8c564b",
    "OTHER":"#7f7f7f"
}

def make_fig8_26_timeline():
    prob = CASE_PROBLEMS["fig8_26"]
    lanes, max_len = [], 0
    for track in TRACKS:
        for m in MODELS:
            timeline, _, _ = build_timeline_and_counts(Path(m["problems_dir"]), track, prob)
            lanes.append((f"{track} | {m['name']}", timeline))
            max_len = max(max_len, len(timeline))
    if max_len == 0:
        print(f"[WARN] No raw attempts found for {prob}. Skip fig8_26.")
        return

    fig_h = 1.0 + 0.4*len(lanes)
    fig, ax = plt.subplots(figsize=(min(12, max(6, max_len*0.12)), fig_h))
    y = 0
    for label, timeline in lanes:
        for x, verdict in enumerate(timeline, start=1):
            ax.add_patch(plt.Rectangle((x-0.5, y-0.4), 1.0, 0.8, color=COLOR_MAP.get(verdict, "#7f7f7f")))
        ax.text(0.0, y, label, va="center", ha="right", fontsize=9, transform=ax.transData)
        y += 1

    ax.set_xlim(0.5, max_len+0.5)
    ax.set_ylim(-0.5, len(lanes)-0.5)
    ax.set_yticks([]); ax.set_xlabel("Attempt index")
    ax.set_title(f"{prob} – Failure timeline (color = verdict)")

    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker='s', color=c, label=k, markersize=8, linestyle='None')
                    for k, c in COLOR_MAP.items()]
    ax.legend(handles=legend_elems, loc="upper right", ncol=4, fontsize=8)
    plt.tight_layout()
    out = Path(OUTDIR) / f"fig8_26_{prob}_timeline.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"[SAVE] {out}")

def make_fig8_27_bar_pass10():
    prob = CASE_PROBLEMS["fig8_27"]
    recs = []
    for track in TRACKS:
        for m in MODELS:
            curve = ensure_pass_curve(m, track, prob, k_max=K_MAX)
            val = float(curve.loc[10]) if (10 in curve.index and pd.notna(curve.loc[10])) else np.nan
            recs.append({"track": track, "model": m["name"], "pass@10": val})
    df = pd.DataFrame(recs)

    if df["pass@10"].isna().all():
        print(f"[WARN] No pass@10 available for {prob}. Skip fig8_27.")
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(TRACKS))
    width = 0.35
    for i, m in enumerate(MODELS):
        y = [df[(df["track"]==t) & (df["model"]==m["name"])]["pass@10"].values[0] for t in TRACKS]
        ax.bar(x + (i-0.5)*width, y, width=width, label=m["name"])
    ax.set_xticks(x); ax.set_xticklabels(TRACKS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Pass@10")
    ax.set_title(f"{prob} – Pass@10 by track and model")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = Path(OUTDIR) / f"fig8_27_{prob}_pass10_bar.png"
    plt.savefig(out)
    plt.close(fig)
    print(f"[SAVE] {out}")

# =========================
# ========= TABLES ========
# =========================

def make_tables():
    """
    表8.15（ABC415）, 表8.16（ABC398）, 表8.17（ABC414）
    3トラック×2モデルの行で、失敗内訳と pass@1/10/100 を列に持つ CSV を保存。
    """
    def one_problem_tables(prob_id: str, tab_no: str):
        rows = []
        for m in MODELS:
            df = failure_table_for_problem(m, prob_id)
            rows.append(df)
        out = pd.concat(rows, axis=0, ignore_index=True)
        out_path = Path(OUTDIR) / f"tab{tab_no}_{prob_id}_failure_breakdown.csv"
        out.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"[SAVE] {out_path}")
        return out_path

    t1 = one_problem_tables(CASE_PROBLEMS["fig8_25"], "8_15")
    t2 = one_problem_tables(CASE_PROBLEMS["fig8_26"], "8_16")
    t3 = one_problem_tables(CASE_PROBLEMS["fig8_27"], "8_17")
    return [t1, t2, t3]

# =========================
# ========= MAIN ==========
# =========================

if __name__ == "__main__":
    print("[INFO] Generating case-study figures/tables ...")
    print("[INFO] MODELS problems_dir/results_dir =")
    for m in MODELS:
        print("   -", m["name"], "problems_dir=", m["problems_dir"], "results_dir=", m["results_dir"])

    try:
        make_fig8_25_pass_curves()
    except Exception as e:
        print(f"[WARN] fig8_25 failed: {e}", file=sys.stderr)

    try:
        make_fig8_26_timeline()
    except Exception as e:
        print(f"[WARN] fig8_26 failed: {e}", file=sys.stderr)

    try:
        make_fig8_27_bar_pass10()
    except Exception as e:
        print(f"[WARN] fig8_27 failed: {e}", file=sys.stderr)

    try:
        _ = make_tables()
    except Exception as e:
        print(f"[WARN] tables failed: {e}", file=sys.stderr)

    print("[DONE] See outputs in:", OUTDIR)
