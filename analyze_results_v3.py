# Analyze competitive programming LLM results (pass@k etc.) — v4 (feedback fixes)
# - Robust AC detection: pass_all / passed==total_cases / counts['AC']==total_cases
# - Attempt discovery order:
#     (A) iter_***/round_***  → attempt = latest round per iter
#     (B) outputs/round_***   → attempt = each round under outputs root
#     (C) iter_***/summary.json (no rounds)
# - Fallback to runs/*/verdict.json when summary.json lacks final status
#
import os, re, json, math
import numpy as np
import pandas as pd
from dataclasses import dataclass
from collections import Counter
from typing import Dict, List, Optional, Tuple

# ==========================
# CONFIG (edit here)
# ==========================
ROOT_DIR = "problems_deepseek"
TRACKS_WHITELIST = ["baseline","CoT","feedback"]   # None → autodetect
K_MAX = 100
OUTPUT_DIR = "deepseek-coder-7b-instruct-v1.5/results"
WEIGHTS_CSV = "./Llama-3.1-8B-Instruct/results/difficulty_weights_A_baseline_only.csv"
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
    if k < 0 or x < 0 or n < 0: return float("nan")
    if k == 0: return 0.0
    if k > n: return float("nan")
    if x == 0: return 0.0
    if x == n: return 1.0
    prod = 1.0
    for i in range(k):
        prod *= (n - x - i) / (n - i)
        if prod <= 0:
            prod = 0.0
            break
    return float(1.0 - prod)

def auc_log_k(pass_curve: Dict[int, float]) -> float:
    ks = sorted([k for k,v in pass_curve.items() if v == v])
    if not ks: return float("nan")
    area = 0.0
    for i in range(len(ks)-1):
        k1,k2 = ks[i], ks[i+1]
        y1,y2 = pass_curve[k1], pass_curve[k2]
        if not (np.isfinite(y1) and np.isfinite(y2)): continue
        area += 0.5*(y1+y2)*(math.log(k2)-math.log(k1))
    return float(area)

def k_at_tau(curve: Dict[int,float], tau: float) -> Optional[int]:
    for k in sorted(curve.keys()):
        v = curve[k]
        if v == v and v >= tau:
            return k
    return None

def find_latest_round(dirpath: str) -> Optional[str]:
    rounds = [d for d in list_dirs(dirpath) if re.match(r"round_\d{3}", d)]
    if not rounds: return None
    rounds.sort(key=lambda s: int(re.search(r"\d{3}", s).group(0)))
    return os.path.join(dirpath, rounds[-1])

def attempt_label_from_summary_or_runs(summ_path: str, runs_dir: Optional[str]) -> str:
    """
    Try summary.json first. If inconclusive, scan runs/*/verdict.json.
    """
    summ = safe_read_json(summ_path) if summ_path else None
    # --- summary-based AC detection ---
    if summ:
        # 1) explicit flags
        if summ.get("iter_pass_all") is True or summ.get("pass_all") is True:
            return "AC"
        # 2) counters
        total = summ.get("total_cases") or summ.get("total")
        passed = summ.get("passed")
        counts = summ.get("counts") or {}
        if isinstance(total, int):
            if isinstance(passed, int) and passed == total:
                return "AC"
            if counts and counts.get("AC", -1) == total:
                return "AC"
        # 3) first non-AC in cases
        cases = summ.get("cases") or []
        for c in cases:
            st = (c.get("status") or
                  ((c.get("verdict") or {}).get("status")))
            if st and st != "AC":
                return st
        # if cases exist and all AC
        if cases and all((c.get("status") == "AC" or ((c.get("verdict") or {}).get("status") == "AC")) for c in cases):
            return "AC"
        # counts fallback: prefer error types if any >0
        for key in ["WA","TLE","RE","MLE","CE"]:
            if counts.get(key, 0) > 0:
                return key
    # --- runs fallback ---
    if runs_dir and os.path.isdir(runs_dir):
        non_ac = None
        for case in list_dirs(runs_dir):
            vpath = os.path.join(runs_dir, case, "verdict.json")
            v = safe_read_json(vpath)
            if not v:
                continue
            st = (v.get("status") or (v.get("verdict") or {}).get("status"))
            if st and st != "AC":
                non_ac = st
                break
        return non_ac or "AC"
    return "OTHER"

def read_attempts_for_problem(outputs_dir: str) -> Tuple[int,int,Counter]:
    """
    Returns:
      n_attempts, x_ac, label_counts
    Attempt discovery order:
      (A) iter_***/round_*** (latest per iter)
      (B) outputs/round_***  (root rounds)
      (C) iter_***/summary.json (no rounds)
    AC counting for pass@k:
      - prefer outputs/overall_summary.json (tries[].iter_pass_all or pass_all / passed==total / counts)
      - else sum( attempt_labels == AC )
    """
    # 1) If overall_summary.json exists, use it to count x
    overall_path = os.path.join(outputs_dir, "overall_summary.json")
    overall = safe_read_json(overall_path)
    overall_x = None
    overall_n = None
    if overall and isinstance(overall.get("tries"), list):
        tries = overall["tries"]
        overall_n = len(tries)
        x = 0
        for t in tries:
            if t.get("iter_pass_all") is True or t.get("pass_all") is True:
                x += 1
            else:
                tot = t.get("total_cases") or t.get("total")
                pas = t.get("passed")
                if isinstance(tot,int) and isinstance(pas,int) and pas==tot:
                    x += 1
        overall_x = x

    # 2) discover attempts
    attempts = []  # list of (summary_path, runs_dir)
    # (A) latest round per iter
    iter_dirs = [d for d in list_dirs(outputs_dir) if re.match(r"iter_\d{3}", d)]
    for it in sorted(iter_dirs):
        round_dir = find_latest_round(os.path.join(outputs_dir, it))
        if round_dir:
            attempts.append((os.path.join(round_dir, "summary.json"),
                             os.path.join(round_dir, "runs")))
    # (B) root rounds if (A) produced nothing
    root_rounds = [d for d in list_dirs(outputs_dir) if re.match(r"round_\d{3}", d)]
    if not attempts and root_rounds:
        for rd in sorted(root_rounds, key=lambda s: int(re.search(r"\d{3}", s).group(0))):
            round_path = os.path.join(outputs_dir, rd)
            attempts.append((os.path.join(round_path, "summary.json"),
                             os.path.join(round_path, "runs")))
    # (C) iter summaries if neither has rounds
    if not attempts and iter_dirs:
        for it in sorted(iter_dirs):
            itdir = os.path.join(outputs_dir, it)
            attempts.append((os.path.join(itdir, "summary.json"),
                             os.path.join(itdir, "runs")))  # runs may not exist

    # 3) label each attempt
    label_counts = Counter()
    ac_flags = []  # for x if overall missing
    for summ_path, runs_dir in attempts:
        lab = attempt_label_from_summary_or_runs(summ_path, runs_dir)
        u = (lab or "OTHER").upper()
        if u.startswith("AC"):
            label_counts["AC"] += 1; ac_flags.append(1)
        elif u.startswith("WA"):
            label_counts["WA"] += 1; ac_flags.append(0)
        elif u.startswith("TL"):
            label_counts["TLE"] += 1; ac_flags.append(0)
        elif u.startswith("RE"):
            label_counts["RE"] += 1; ac_flags.append(0)
        elif u.startswith("ML"):
            label_counts["MLE"] += 1; ac_flags.append(0)
        elif u in {"CE","NO_FENCE","SYNTAX","POLICY","GEN_TIMEOUT","NOEXP"}:
            label_counts["CE"] += 1; ac_flags.append(0)
        else:
            label_counts["OTHER"] += 1; ac_flags.append(0)

    # fill zeros
    for key in ["AC","WA","TLE","RE","MLE","CE","OTHER"]:
        label_counts.setdefault(key, 0)

    # n, x
    n_attempts = overall_n if overall_n is not None else len(attempts)
    if overall_x is not None:
        x_ac = overall_x
    else:
        x_ac = int(sum(ac_flags))

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
        return [t for t in TRACKS_WHITELIST if os.path.isdir(os.path.join(root_dir, t))]
    return [t for t in list_dirs(root_dir)]

def discover_problems(track_dir: str) -> List[Tuple[str,str]]:
    res = []
    # direct children
    for name in list_dirs(track_dir):
        if ABC_DIR_PATTERN.match(name):
            out = os.path.join(track_dir, name, "outputs")
            if os.path.isdir(out):
                res.append((name.upper(), out))
    # walk fallback
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
    n = len(vals); stats=[]
    for _ in range(B):
        idx = np.random.randint(0,n,size=n)
        stats.append(func(vals[idx]))
    lo,hi = np.percentile(stats,[2.5,97.5]).tolist()
    return float(lo), float(hi)

def sign_test_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]; diffs = diffs[diffs!=0.0]
    n = len(diffs)
    if n == 0: return float("nan")
    k = int((diffs > 0).sum())
    from math import comb
    tail = min(k, n-k); p=0.0
    for i in range(0, tail+1):
        p += comb(n, i) / (2**n)
    return float(2*p)

def analyze():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # optional weights
    weights: Dict[str,float] = {}
    if WEIGHTS_CSV and os.path.isfile(WEIGHTS_CSV):
        dfw = pd.read_csv(WEIGHTS_CSV)
        for _, row in dfw.iterrows():
            weights[str(row["problem_id"]).upper()] = float(row["w"])

    tracks = discover_tracks(ROOT_DIR)
    if VERBOSE: print("[DEBUG] tracks:", tracks)
    results: List[ProblemResult] = []
    errors = []

    for track in tracks:
        tdir = os.path.join(ROOT_DIR, track)
        probs = discover_problems(tdir)
        if VERBOSE:
            print(f"[INFO] Track '{track}': {len(probs)} problems")
        for pid, outdir in probs:
            try:
                n,x,labels = read_attempts_for_problem(outdir)
                if n <= 0:
                    raise RuntimeError("No attempts found")
                curve = {k: (stable_pass_at_k(n,x,k) if k<=n else float('nan')) for k in range(1,K_MAX+1)}
                k50 = k_at_tau(curve, 0.5); k90 = k_at_tau(curve, 0.9)
                aucv = auc_log_k(curve)
                results.append(ProblemResult(pid, track, n, x, curve, k50, k90, float(aucv), dict(labels)))
            except Exception as e:
                errors.append((pid, str(e)))

    if not results:
        print("[ERROR] No results. Check directory layout.")
        return

    # per-problem CSV
    rows=[]
    for r in results:
        row = {"problem_id": r.problem_id, "track": r.track, "n_attempts": r.n, "x_ac": r.x,
               "k50": (r.k50 if r.k50 is not None else ""), "k90": (r.k90 if r.k90 is not None else ""),
               "auc_logk": r.auc,
               "label_AC": r.labels.get("AC",0),"label_WA": r.labels.get("WA",0),"label_TLE": r.labels.get("TLE",0),
               "label_RE": r.labels.get("RE",0),"label_MLE": r.labels.get("MLE",0),"label_CE": r.labels.get("CE",0),
               "label_OTHER": r.labels.get("OTHER",0)}
        for k in range(1, K_MAX+1):
            row[f"pass_at_{k}"] = r.pass_curve.get(k, float("nan"))
        rows.append(row)
    df_per = pd.DataFrame(rows).sort_values(["track","problem_id"])
    df_per.to_csv(os.path.join(OUTPUT_DIR,"per_problem_pass_at_k.csv"), index=False)
    print("[OK] per_problem_pass_at_k.csv")

    # overall curves (Mean & D)
    overall=[]
    for track in sorted(set(r.track for r in results)):
        dft = df_per[df_per["track"]==track]
        ws = np.array([weights.get(pid,1.0) for pid in dft["problem_id"]], dtype=float)
        if ws.size>0: ws = ws * (len(ws)/ws.sum())
        for k in range(1, K_MAX+1):
            vals = dft[f"pass_at_{k}"].to_numpy(dtype=float)
            mask = np.isfinite(vals)
            if not mask.any():
                mean_val = d_mean_val = np.nan; ci_lo=ci_hi=d_ci_lo=d_ci_hi=np.nan
            else:
                mean_val = float(np.nanmean(vals[mask]))
                d_mean_val = float(np.nansum(ws[mask]*vals[mask]) / np.nansum(ws[mask]))
                idx = np.where(mask)[0]; n_idx=len(idx); B=BOOTSTRAP_B
                stats_mean=[]; stats_d=[]
                for _ in range(B):
                    bs = np.random.randint(0,n_idx,size=n_idx)
                    v_bs = vals[idx][bs]; w_bs = ws[idx][bs]
                    stats_mean.append(np.nanmean(v_bs))
                    stats_d.append(np.nansum(w_bs*v_bs)/np.nansum(w_bs))
                ci_lo,ci_hi = np.percentile(stats_mean,[2.5,97.5]).tolist()
                d_ci_lo,d_ci_hi = np.percentile(stats_d,[2.5,97.5]).tolist()
            overall.append({"track":track,"k":k,"mean_pass":mean_val,"mean_pass_ci_lo":ci_lo,"mean_pass_ci_hi":ci_hi,
                            "d_pass":d_mean_val,"d_pass_ci_lo":d_ci_lo,"d_pass_ci_hi":d_ci_hi})
    pd.DataFrame(overall).to_csv(os.path.join(OUTPUT_DIR,"overall_curves.csv"), index=False)
    print("[OK] overall_curves.csv")

    # summary table (@1/@10/@100, AUC, k50/k90)
    def summarize_series(vals, w):
        vals = np.asarray(vals, dtype=float); mask = np.isfinite(vals)
        if not mask.any(): return (np.nan,np.nan,(np.nan,np.nan),(np.nan,np.nan))
        mean_val = float(np.nanmean(vals[mask]))
        d_val = float(np.nansum(w[mask]*vals[mask])/np.nansum(w[mask]))
        idx = np.where(mask)[0]; n_idx=len(idx); B=BOOTSTRAP_B
        stats_mean=[]; stats_d=[]
        for _ in range(B):
            bs = np.random.randint(0,n_idx,size=n_idx)
            v_bs = vals[idx][bs]; w_bs = w[idx][bs]
            stats_mean.append(np.nanmean(v_bs))
            stats_d.append(np.nansum(w_bs*v_bs)/np.nansum(w_bs))
        ci_m = tuple(np.percentile(stats_mean,[2.5,97.5]).tolist())
        ci_d = tuple(np.percentile(stats_d,[2.5,97.5]).tolist())
        return mean_val, d_val, ci_m, ci_d

    summary=[]
    for track in sorted(set(r.track for r in results)):
        dft = df_per[df_per["track"]==track]
        ws = np.array([weights.get(pid,1.0) for pid in dft["problem_id"]], dtype=float)
        if ws.size>0: ws = ws*(len(ws)/ws.sum())

        for kval in [1,10,100]:
            vals = dft[f"pass_at_{kval}"].to_numpy(dtype=float)
            mv,dv,ci_m,ci_d = summarize_series(vals, ws)
            summary.append({"track":track,"metric":f"pass@{kval}",
                            "mean":mv,"mean_ci_lo":ci_m[0],"mean_ci_hi":ci_m[1],
                            "d_mean":dv,"d_mean_ci_lo":ci_d[0],"d_mean_ci_hi":ci_d[1]})
        auc_vals = dft["auc_logk"].to_numpy(dtype=float)
        mv,dv,ci_m,ci_d = summarize_series(auc_vals, ws)
        summary.append({"track":track,"metric":"AUC_logk",
                        "mean":mv,"mean_ci_lo":ci_m[0],"mean_ci_hi":ci_m[1],
                        "d_mean":dv,"d_mean_ci_lo":ci_d[0],"d_mean_ci_hi":ci_d[1]})
        # k50/k90: median + bootstrap CI
        for tag,col in [("k50","k50"),("k90","k90")]:
            arr = dft[col].replace("", np.nan).astype(float).to_numpy()
            mask = np.isfinite(arr)
            if mask.any():
                med = float(np.nanmedian(arr[mask]))
                vals = arr[mask]; n=len(vals); B=BOOTSTRAP_B; stats=[]
                for _ in range(B):
                    idx = np.random.randint(0,n,size=n); stats.append(np.median(vals[idx]))
                lo,hi = np.percentile(stats,[2.5,97.5]).tolist()
            else:
                med=lo=hi=np.nan
            summary.append({"track":track,"metric":tag,"median":med,"median_ci_lo":lo,"median_ci_hi":hi})
    pd.DataFrame(summary).to_csv(os.path.join(OUTPUT_DIR,"overall_summary.csv"), index=False)
    print("[OK] overall_summary.csv")

    # attempt label counts
    lab_rows=[]
    for _, r in df_per.iterrows():
        lab_rows.append({"track": r["track"], "problem_id": r["problem_id"],
                         "AC": int(r["label_AC"]), "WA": int(r["label_WA"]), "TLE": int(r["label_TLE"]),
                         "RE": int(r["label_RE"]), "MLE": int(r["label_MLE"]), "CE": int(r["label_CE"]),
                         "OTHER": int(r["label_OTHER"]), "n_attempts": int(r["n_attempts"])})
    pd.DataFrame(lab_rows).to_csv(os.path.join(OUTPUT_DIR,"attempt_label_counts.csv"), index=False)
    print("[OK] attempt_label_counts.csv")

    print("\n[DONE] All numeric results computed.\n")

if __name__ == "__main__":
    analyze()
