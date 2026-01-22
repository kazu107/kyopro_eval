# %%
# Revised analyzer with auto-discovery & richer diagnostics.
# - Auto-detect tracks under ROOT_DIR (unless TRACKS_WHITELIST is set)
# - Flexible ABC directory match (case-insensitive, 3+ digits)
# - Fallback: os.walk to discover <ABC***/outputs> even with extra nesting
# - Verbose diagnostics when 0 problems_llama found (lists actual dirs)
#
import os, re, json, math, numpy as np, pandas as pd
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# ==========================
# CONFIG
# ==========================
ROOT_DIR = "problems_llama"
TRACKS_WHITELIST = None  # e.g., ["baseline","CoT","feedback"]; None => auto-discover
K_MAX = 100
OUTPUT_DIR = "Llama-3.1-8B-Instruct/results"
WEIGHTS_CSV = "./results/difficulty_weights_A_baseline_only.csv"
BOOTSTRAP_B = 2000
RANDOM_SEED = 1729
ABC_DIR_PATTERN = re.compile(r"(?i)^abc\d{3,}$")  # case-insensitive, 3+ digits
VERBOSE = True
# ==========================

np.random.seed(RANDOM_SEED)

def list_dirs(path: str) -> List[str]:
    try:
        return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    except FileNotFoundError:
        return []

def safe_read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def stable_pass_at_k(n: int, x: int, k: int) -> float:
    if k < 0 or x < 0 or n < 0:
        return float("nan")
    if k == 0:
        return 0.0
    if k > n:
        return float("nan")
    if x == 0:
        return 0.0
    if x == n:
        return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - x - i) / (n - i)
        if prod <= 0:
            prod = 0.0
            break
    return float(1.0 - prod)

def auc_log_k(pass_curve: Dict[int, float]) -> float:
    ks = sorted([k for k, v in pass_curve.items() if v == v])
    if not ks: return float("nan")
    area = 0.0
    for i in range(len(ks)-1):
        k1,k2 = ks[i], ks[i+1]
        y1,y2 = pass_curve[k1], pass_curve[k2]
        if not (np.isfinite(y1) and np.isfinite(y2)): continue
        area += 0.5*(y1+y2)*(math.log(k2)-math.log(k1))
    return float(area)

def k_at_tau(pass_curve: Dict[int, float], tau: float) -> Optional[int]:
    for k in sorted(pass_curve.keys()):
        v = pass_curve[k]
        if v == v and v >= tau:
            return k
    return None

def latest_round_dir(iter_dir: str) -> Optional[str]:
    rounds = [d for d in list_dirs(iter_dir) if re.match(r"round_\d{3}", d)]
    if not rounds: return None
    rounds_sorted = sorted(rounds, key=lambda s: int(re.search(r"\d{3}", s).group(0)))
    return os.path.join(iter_dir, rounds_sorted[-1])

def attempt_label_from_summary(summary_json: dict) -> str:
    if summary_json is None: return "OTHER"
    if summary_json.get("iter_pass_all") is True: return "AC"
    cases = summary_json.get("cases", [])
    for c in cases:
        st = c.get("status") or (c.get("verdict", {}) or {}).get("status")
        if st and st != "AC": return st
    counts = summary_json.get("counts", {})
    for k in ["WA", "TLE", "RE", "MLE", "CE"]:
        if counts.get(k, 0) > 0: return k
    return "OTHER"

def read_attempts_for_problem(outputs_dir: str) -> Tuple[int,int,Counter]:
    overall_path = os.path.join(outputs_dir, "overall_summary.json")
    overall = safe_read_json(overall_path)
    n_attempts = 0; x_ac = 0
    if overall and isinstance(overall.get("tries"), list):
        tries = overall["tries"]
        n_attempts = len(tries)
        x_ac = sum(1 for t in tries if bool(t.get("iter_pass_all")) or (t.get("passed",0)==t.get("total",0)))
    else:
        iter_dirs = sorted([d for d in list_dirs(outputs_dir) if re.match(r"iter_\d{3}", d)])
        n_attempts = len(iter_dirs)
        x_ac = 0
        for d in iter_dirs:
            summ = safe_read_json(os.path.join(outputs_dir, d, "summary.json"))
            if summ and summ.get("iter_pass_all"): x_ac += 1
    # labels
    label_counts = Counter()
    iter_dirs = sorted([d for d in list_dirs(outputs_dir) if re.match(r"iter_\d{3}", d)])
    for d in iter_dirs:
        iterd = os.path.join(outputs_dir, d)
        roundd = latest_round_dir(iterd)
        summ_path = os.path.join(roundd, "summary.json") if roundd else os.path.join(iterd, "summary.json")
        summ = safe_read_json(summ_path)
        lab = attempt_label_from_summary(summ)
        u = (lab or "OTHER").upper()
        if u.startswith("AC"): label_counts["AC"] += 1
        elif u.startswith("WA"): label_counts["WA"] += 1
        elif u.startswith("TL"): label_counts["TLE"] += 1
        elif u.startswith("RE"): label_counts["RE"] += 1
        elif u.startswith("ML"): label_counts["MLE"] += 1
        elif u in {"CE","NO_FENCE","SYNTAX","POLICY","GEN_TIMEOUT","NOEXP"}: label_counts["CE"] += 1
        else: label_counts["OTHER"] += 1
    for key in ["AC","WA","TLE","RE","MLE","CE","OTHER"]:
        label_counts.setdefault(key,0)
    return n_attempts, x_ac, label_counts

@dataclass
class ProblemResult:
    problem_id: str
    track: str
    n: int
    x: int
    pass_curve: Dict[int,float]
    k50: Optional[int]
    k90: Optional[int]
    auc: float
    labels: Dict[str,int]

def discover_tracks(root_dir: str) -> List[str]:
    if TRACKS_WHITELIST:
        return TRACKS_WHITELIST
    tracks = list_dirs(root_dir)
    # filter to dirs that actually contain some ABC***/outputs somewhere
    good = []
    for t in tracks:
        tpath = os.path.join(root_dir, t)
        found = False
        for name in list_dirs(tpath):
            if ABC_DIR_PATTERN.match(name) and os.path.isdir(os.path.join(tpath, name, "outputs")):
                found = True; break
        if not found:
            # fallback: search one level deeper just in case
            for dirpath, dirnames, _ in os.walk(tpath):
                if os.path.basename(dirpath).lower() == "outputs":
                    comps = dirpath.split(os.sep)
                    if any(ABC_DIR_PATTERN.match(c or "") for c in comps):
                        found = True; break
        if found:
            good.append(t)
    if VERBOSE:
        print(f"[DEBUG] Auto-discovered tracks under '{root_dir}': {good or tracks} (good first)")
    return good or tracks

def discover_problems(track_dir: str) -> List[Tuple[str,str]]:
    """Return list of (problem_id, outputs_dir) for a given track_dir."""
    res = []
    # First try direct children
    for name in list_dirs(track_dir):
        if ABC_DIR_PATTERN.match(name):
            out = os.path.join(track_dir, name, "outputs")
            if os.path.isdir(out):
                res.append((name.upper(), out))
    # Fallback: walk
    if not res:
        for dirpath, dirnames, _ in os.walk(track_dir):
            if os.path.basename(dirpath).lower() == "outputs":
                comps = dirpath.split(os.sep)
                pid = None
                for c in comps[::-1]:
                    if ABC_DIR_PATTERN.match(c or ""):
                        pid = c.upper(); break
                if pid:
                    res.append((pid, dirpath))
    return sorted(set(res))

def bootstrap_ci(vals: np.ndarray, func=np.mean, B: int = BOOTSTRAP_B, alpha=0.05):
    vals = np.asarray(vals, dtype=float); vals = vals[np.isfinite(vals)]
    if vals.size == 0: return (float("nan"), float("nan"))
    n = len(vals); stats = []
    for _ in range(B):
        idx = np.random.randint(0, n, size=n)
        stats.append(func(vals[idx]))
    lo, hi = np.percentile(stats, [100*(alpha/2), 100*(1-alpha/2)]).tolist()
    return float(lo), float(hi)

def sign_test_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float); diffs = diffs[np.isfinite(diffs)]; diffs = diffs[diffs!=0.0]
    n = len(diffs)
    if n == 0: return float("nan")
    k = int((diffs > 0).sum())
    from math import comb
    tail = min(k, n-k); p = 0.0
    for i in range(0, tail+1):
        p += comb(n, i) / (2**n)
    return float(2*p)

def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load weights (optional)
    weights: Dict[str,float] = {}
    if WEIGHTS_CSV and os.path.isfile(WEIGHTS_CSV):
        dfw = pd.read_csv(WEIGHTS_CSV)
        for _, row in dfw.iterrows():
            weights[str(row["problem_id"]).upper()] = float(row["w"])

    # Tracks
    tracks = discover_tracks(ROOT_DIR)
    if not tracks:
        print(f"[ERROR] No tracks found under '{ROOT_DIR}'. Check ROOT_DIR.")
        return

    all_results: List[ProblemResult] = []
    errors = []

    for track in tracks:
        track_dir = os.path.join(ROOT_DIR, track)
        if not os.path.isdir(track_dir):
            if VERBOSE: print(f"[WARN] Track dir not found: {track_dir}")
            continue
        probs = discover_problems(track_dir)
        if VERBOSE:
            child_dirs = list_dirs(track_dir)[:10]
            print(f"[INFO] Track '{track}': discovered {len(probs)} problems_llama. First-level dirs sample: {child_dirs}")
        for pid, outputs_dir in probs:
            try:
                n, x, labels = read_attempts_for_problem(outputs_dir)
                if n <= 0:
                    raise RuntimeError("No attempts found")
                pass_curve = {k: (stable_pass_at_k(n, x, k) if k <= n else float("nan"))
                              for k in range(1, K_MAX+1)}
                k50 = k_at_tau(pass_curve, 0.5)
                k90 = k_at_tau(pass_curve, 0.9)
                auc_val = auc_log_k(pass_curve)
                all_results.append(ProblemResult(
                    problem_id=pid, track=track, n=n, x=x,
                    pass_curve=pass_curve, k50=k50, k90=k90, auc=float(auc_val),
                    labels=dict(labels)
                ))
            except Exception as e:
                errors.append((outputs_dir, str(e)))

    if not all_results:
        # extra diagnostics
        if VERBOSE:
            print("[DIAG] No problems_llama assembled. Directory listing:")
            for t in list_dirs(ROOT_DIR):
                print("  -", t, "->", list_dirs(os.path.join(ROOT_DIR, t))[:20])
        print("[ERROR] No results assembled (no problems_llama found?). Nothing to write.")
        return

    # Per-problem CSV
    rows = []
    for r in all_results:
        row = {
            "problem_id": r.problem_id,
            "track": r.track,
            "n_attempts": r.n,
            "x_ac": r.x,
            "k50": r.k50 if r.k50 is not None else "",
            "k90": r.k90 if r.k90 is not None else "",
            "auc_logk": r.auc,
            "label_AC": r.labels.get("AC",0),
            "label_WA": r.labels.get("WA",0),
            "label_TLE": r.labels.get("TLE",0),
            "label_RE": r.labels.get("RE",0),
            "label_MLE": r.labels.get("MLE",0),
            "label_CE": r.labels.get("CE",0),
            "label_OTHER": r.labels.get("OTHER",0),
        }
        for k in range(1, K_MAX+1):
            row[f"pass_at_{k}"] = r.pass_curve.get(k, float("nan"))
        rows.append(row)

    df_per = pd.DataFrame(rows).sort_values(["track","problem_id"])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df_per.to_csv(os.path.join(OUTPUT_DIR, "per_problem_pass_at_k.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(OUTPUT_DIR, 'per_problem_pass_at_k.csv')}")

    # Overall curves (Mean & D)
    overall_rows = []
    for track in sorted(set(r.track for r in all_results)):
        df_t = df_per[df_per["track"] == track]
        ws = np.array([weights.get(pid, 1.0) for pid in df_t["problem_id"]], dtype=float)
        if ws.size > 0: ws = ws * (len(ws)/ws.sum())
        for k in range(1, K_MAX+1):
            vals = df_t[f"pass_at_{k}"].to_numpy(dtype=float)
            mask = np.isfinite(vals)
            if not mask.any():
                mean_val = d_mean_val = np.nan
                ci_lo = ci_hi = d_ci_lo = d_ci_hi = np.nan
            else:
                mean_val = float(np.nanmean(vals[mask]))
                d_mean_val = float(np.nansum(ws[mask]*vals[mask]) / np.nansum(ws[mask]))
                idx = np.where(mask)[0]; n_idx = len(idx); B = BOOTSTRAP_B
                stats_mean=[]; stats_d=[]
                for _ in range(B):
                    bs = np.random.randint(0, n_idx, size=n_idx)
                    v_bs = vals[idx][bs]; w_bs = ws[idx][bs]
                    stats_mean.append(np.nanmean(v_bs))
                    stats_d.append(np.nansum(w_bs*v_bs)/np.nansum(w_bs))
                ci_lo, ci_hi = np.percentile(stats_mean,[2.5,97.5]).tolist()
                d_ci_lo, d_ci_hi = np.percentile(stats_d,[2.5,97.5]).tolist()
            overall_rows.append({
                "track": track, "k": k,
                "mean_pass": mean_val, "mean_pass_ci_lo": ci_lo, "mean_pass_ci_hi": ci_hi,
                "d_pass": d_mean_val, "d_pass_ci_lo": d_ci_lo, "d_pass_ci_hi": d_ci_hi,
            })
    pd.DataFrame(overall_rows).to_csv(os.path.join(OUTPUT_DIR,"overall_curves.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(OUTPUT_DIR, 'overall_curves.csv')}")

    # Summary: @1/@10/@100, AUC, k50/k90
    def summarize_series(vals: np.ndarray, w: np.ndarray):
        mask = np.isfinite(vals)
        if not mask.any(): return (np.nan, np.nan, (np.nan,np.nan), (np.nan,np.nan))
        mean_val = float(np.nanmean(vals[mask]))
        d_val = float(np.nansum(w[mask]*vals[mask]) / np.nansum(w[mask]))
        idx = np.where(mask)[0]; n_idx=len(idx); B=BOOTSTRAP_B
        stats_mean=[]; stats_d=[]
        for _ in range(B):
            bs = np.random.randint(0, n_idx, size=n_idx)
            v_bs = vals[idx][bs]; w_bs = w[idx][bs]
            stats_mean.append(np.nanmean(v_bs))
            stats_d.append(np.nansum(w_bs*v_bs)/np.nansum(w_bs))
        ci_mean = tuple(np.percentile(stats_mean,[2.5,97.5]).tolist())
        ci_d = tuple(np.percentile(stats_d,[2.5,97.5]).tolist())
        return mean_val, d_val, ci_mean, ci_d

    summary_rows = []
    for track in sorted(set(r.track for r in all_results)):
        df_t = df_per[df_per["track"] == track]
        ws = np.array([weights.get(pid, 1.0) for pid in df_t["problem_id"]], dtype=float)
        if ws.size > 0: ws = ws * (len(ws)/ws.sum())

        for kval in [1,10,100]:
            vals = df_t[f"pass_at_{kval}"].to_numpy(dtype=float)
            mv, dv, ci_m, ci_d = summarize_series(vals, ws)
            summary_rows.append({"track": track, "metric": f"pass@{kval}",
                                 "mean": mv, "mean_ci_lo": ci_m[0], "mean_ci_hi": ci_m[1],
                                 "d_mean": dv, "d_mean_ci_lo": ci_d[0], "d_mean_ci_hi": ci_d[1]})
        auc_vals = df_t["auc_logk"].to_numpy(dtype=float)
        mv, dv, ci_m, ci_d = summarize_series(auc_vals, ws)
        summary_rows.append({"track": track, "metric": "AUC_logk",
                             "mean": mv, "mean_ci_lo": ci_m[0], "mean_ci_hi": ci_m[1],
                             "d_mean": dv, "d_mean_ci_lo": ci_d[0], "d_mean_ci_hi": ci_d[1]})
        for tag, col in [("k50","k50"), ("k90","k90")]:
            arr = df_t[col].replace("", np.nan).astype(float).to_numpy()
            mask = np.isfinite(arr)
            if mask.any():
                med = float(np.nanmedian(arr[mask]))
                # simple bootstrap for median
                vals = arr[mask]; n=len(vals); B=BOOTSTRAP_B; stats=[]
                for _ in range(B):
                    idx = np.random.randint(0,n,size=n)
                    stats.append(np.median(vals[idx]))
                lo,hi = np.percentile(stats,[2.5,97.5]).tolist()
            else:
                med=lo=hi=np.nan
            summary_rows.append({"track": track, "metric": tag,
                                 "median": med, "median_ci_lo": lo, "median_ci_hi": hi})
    pd.DataFrame(summary_rows).to_csv(os.path.join(OUTPUT_DIR,"overall_summary.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(OUTPUT_DIR, 'overall_summary.csv')}")

    # Pairwise tests
    tracks_present = sorted(set(r.track for r in all_results))
    pairs = [(tracks_present[i], tracks_present[j]) for i in range(len(tracks_present)) for j in range(i+1,len(tracks_present))]
    tests = []
    for (t1,t2) in pairs:
        d1 = df_per[df_per["track"]==t1].set_index("problem_id")
        d2 = df_per[df_per["track"]==t2].set_index("problem_id")
        common = sorted(set(d1.index)&set(d2.index))
        if not common: continue
        for metric,col in [("pass@1","pass_at_1"),("pass@10","pass_at_10"),("pass@100","pass_at_100"),("AUC_logk","auc_logk")]:
            v1 = d1.loc[common, col].to_numpy(dtype=float)
            v2 = d2.loc[common, col].to_numpy(dtype=float)
            diffs = v2 - v1
            diffs_finite = diffs[np.isfinite(diffs)]
            if diffs_finite.size>0:
                diff_mean = float(np.mean(diffs_finite))
                # bootstrap CI
                n=len(diffs_finite); B=BOOTSTRAP_B; stats=[]
                for _ in range(B):
                    idx = np.random.randint(0,n,size=n)
                    stats.append(float(np.mean(diffs_finite[idx])))
                lo,hi = np.percentile(stats,[2.5,97.5]).tolist()
                pval = (lambda arr: (lambda x: x)(
                    (lambda d: (lambda n,k: (lambda prob: 2*prob)(sum(math.comb(n,i) for i in range(0,min(k,n-k)+1))/2**n))(len(d), int((d>0).sum())))
                    )(arr[np.isfinite(arr) & (arr!=0)])
                )(diffs)
            else:
                diff_mean=lo=hi=pval=np.nan
            tests.append({"pair": f"{t1} -> {t2}", "metric": metric, "n_problems": len(common),
                          "mean_diff_t2_minus_t1": diff_mean, "mean_diff_ci_lo": lo, "mean_diff_ci_hi": hi,
                          "sign_test_p": pval})
    pd.DataFrame(tests).to_csv(os.path.join(OUTPUT_DIR,"pairwise_tests.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(OUTPUT_DIR, 'pairwise_tests.csv')}")

    # Attempt label counts
    lab_rows=[]
    for _, r in df_per.iterrows():
        lab_rows.append({"track": r["track"], "problem_id": r["problem_id"],
                         "AC": int(r["label_AC"]), "WA": int(r["label_WA"]), "TLE": int(r["label_TLE"]),
                         "RE": int(r["label_RE"]), "MLE": int(r["label_MLE"]), "CE": int(r["label_CE"]),
                         "OTHER": int(r["label_OTHER"]), "n_attempts": int(r["n_attempts"])})
    pd.DataFrame(lab_rows).to_csv(os.path.join(OUTPUT_DIR,"attempt_label_counts.csv"), index=False)
    print(f"[OK] Wrote {os.path.join(OUTPUT_DIR, 'attempt_label_counts.csv')}")

    print("\n[DONE] All numeric results computed.\n")

if __name__ == "__main__":
    analyze()
