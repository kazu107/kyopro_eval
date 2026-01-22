# sign_test_holm.py — DeepSeek vs Llama の符号検定（Holm補正）を一括出力
import pandas as pd, numpy as np, math, re, os

# === 設定（ここだけ編集） ===
INPUT_CSV = "./viz_out/joined_per_problem.csv"
OUT_CSV   = "./sign_test__deepseek_vs_llama.csv"
MODEL_A_KEYWORDS = ["deepseek"]  # A側（勝ち数を数える側）
MODEL_B_KEYWORDS = ["llama"]     # B側
K_PRIORITY = [1, 10, 100]        # 優先して見る k。無ければ検出したkを全部
TRACK_FILTERS = None             # 例: ["baseline","CoT","feedback"] / Noneで全トラック

# === ユーティリティ ===
def find_col(df, cands):
    low = {c.lower(): c for c in df.columns}
    for name in cands:
        if name.lower() in low: return low[name.lower()]
    for c in df.columns:
        if any(name.lower() in c.lower() for name in cands): return c
    return None

def detect_pass_k_columns(cols):
    mp = {}
    for c in cols:
        cl = c.lower()
        if any(p in cl for p in ["pass@", "pass_at_", "pass_at", "pass", "passk", "pass_k_", "ac@", "ac_at_", "acc@"]):
            m = re.search(r'(\d+)', cl)
            if m: mp[int(m.group(1))] = c
    # k→列名（重複は先勝ち）
    out = {}
    for k in sorted(mp): out.setdefault(k, mp[k])
    return out

def is_model(name, keywords):
    s = str(name).lower()
    return any(kw.lower() in s for kw in keywords)

def binom_sf_geq(k, n, p=0.5):
    # P[X >= k], X~Binom(n,p)（厳密）
    s = 0.0
    for x in range(k, n+1):
        s += math.comb(n,x)*(p**x)*((1-p)**(n-x))
    return s

def holm_adjust(pairs):
    # pairs: [(key, p_raw), ...]
    orded = sorted(pairs, key=lambda x: x[1])
    m = len(orded); interim = []; out = {}; mx = 0.0
    for i,(k,p) in enumerate(orded, start=1):
        adj = min(1.0, (m - i + 1)*p); interim.append((k, adj))
    for k,adj in interim:
        mx = max(mx, adj); out[k] = mx
    return out

# === 本体 ===
if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(f"INPUT_CSV not found: {INPUT_CSV}")

df = pd.read_csv(INPUT_CSV)

col_model   = find_col(df, ["model","model_name"])
col_track   = find_col(df, ["track","method","setting"])
col_problem = find_col(df, ["problem","problem_id","task_id"])
if col_model is None or col_problem is None:
    raise RuntimeError(f"Required columns not found. Columns={list(df.columns)}")
if col_track is None:
    col_track = "_track"; df[col_track] = "all"

k_cols = detect_pass_k_columns(df.columns)
ks = [k for k in K_PRIORITY if k in k_cols] or sorted(k_cols)

tracks = sorted(df[col_track].dropna().unique().tolist())
if TRACK_FILTERS:
    tracks = [t for t in tracks if any(t == f for f in TRACK_FILTERS)]

rows = []
for t in tracks:
    dt = df[df[col_track] == t]
    A = dt[dt[col_model].apply(lambda x: is_model(x, MODEL_A_KEYWORDS))]
    B = dt[dt[col_model].apply(lambda x: is_model(x, MODEL_B_KEYWORDS))]
    if A.empty or B.empty: continue
    A_grp = A.groupby(col_problem).agg({k_cols[k]:"first" for k in ks}).rename(columns={k_cols[k]:f"A@{k}" for k in ks})
    B_grp = B.groupby(col_problem).agg({k_cols[k]:"first" for k in ks}).rename(columns={k_cols[k]:f"B@{k}" for k in ks})
    M = A_grp.join(B_grp, how="inner")
    for k in ks:
        a = pd.to_numeric(M[f"A@{k}"], errors="coerce")
        b = pd.to_numeric(M[f"B@{k}"], errors="coerce")
        a = (a > 0).astype(int); b = (b > 0).astype(int)
        d = a - b
        ties = int((d==0).sum()); diffs = d[d!=0]; n = int(len(diffs))
        if n == 0:
            rows.append({"track":t,"k":k,"n_effective":0,"ties":ties,"S_positive":0,"win_rate":np.nan,"p_raw_two_sided":np.nan})
            continue
        S = int((diffs>0).sum())
        p_hi = binom_sf_geq(S, n, 0.5)
        p_lo = 1.0 - binom_sf_geq(S-1, n, 0.5)  # P[X<=S]
        p = float(min(1.0, max(0.0, 2*min(p_hi, p_lo))))
        rows.append({"track":t,"k":k,"n_effective":n,"ties":ties,"S_positive":S,"win_rate":round(S/n,4),"p_raw_two_sided":p})

res = pd.DataFrame(rows).sort_values(["track","k"]).reset_index(drop=True)
if not res.empty:
    keys = list(zip(res["track"].astype(str), res["k"].astype(int)))
    pmap = holm_adjust([(k,p) for k,p in zip(keys, res["p_raw_two_sided"]) if not np.isnan(p)])
    res["p_holm"] = [pmap.get(k, np.nan) for k in keys]

res.to_csv(OUT_CSV, index=False)
print(f"OK -> {OUT_CSV}\n"); print(res)
