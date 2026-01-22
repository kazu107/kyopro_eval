# -*- coding: utf-8 -*-
"""
難易度重み w のマージを堅牢化した版
- w 衝突（w_x / w_y / 既存w）を必ず単一列 'w' に統合
- problem_id を str 化・strip してキー不一致を最小化
- 詳細ログで欠落箇所を把握
- Llama-baseline（なければ自動）で corr(w, pass@k) と四分位平均を出力
"""

from pathlib import Path
import pandas as pd
import numpy as np

# ====== 設定 ======
JOINED_CSV = Path("model_compare_out/joined_all_models.csv")   # 例: Path("/mnt/data/joined_all_models.csv")
WEIGHT_CSV = Path("C:/Users/kazuu/PycharmProjects/atc/Llama-3.1-8B-Instruct/results/difficulty_weights_A_baseline_only.csv")    # 例: Path("/mnt/data/difficulty_weights_A_baseline_only.csv")
PREFERRED_MODEL = "Llama-3.1-8B-Instruct"
PREFERRED_TRACK = "baseline"
PASS_KS = [1, 10, 100]
# ==================

def need(p: Path) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"ファイルが見つかりません: {p.resolve()}")
    return p

def std_problem_id(df: pd.DataFrame) -> pd.DataFrame:
    """problem_id を作成/正規化（str化・strip）。"""
    df = df.copy()
    # 列名のゆれ吸収
    if "problem_id" not in df.columns:
        # よくある代替名
        for alt in ["problem", "task_id", "id"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "problem_id"})
                break
    if "problem_id" not in df.columns:
        raise KeyError("problem_id / problem / task_id / id のいずれも見つかりません")
    # 文字列化 + 前後空白除去 + 明らかな余計な suffix/prefix の簡易正規化を追加したければここで
    df["problem_id"] = df["problem_id"].astype(str).str.strip()
    return df

def detect_weight_col(wdf: pd.DataFrame) -> str:
    """重み列を自動検出。"""
    cands = ["w", "w_p", "weight", "weight_p"]
    for c in cands:
        if c in wdf.columns:
            return c
    raise KeyError(f"重み列が見つかりません（候補: {cands}）。実列: {list(wdf.columns)}")

def load_joined(path: Path) -> pd.DataFrame:
    df = pd.read_csv(need(path))
    print(f"[INFO] joined shape: {df.shape}  cols(sample)={list(df.columns)[:12]} ...")
    df = std_problem_id(df)
    # 型調整
    for col in ["track", "model"]:
        if col not in df.columns:
            raise KeyError(f"joined に {col} 列がありません")
        df[col] = df[col].astype(str).str.strip()
    return df

def load_weights(path: Path) -> pd.DataFrame:
    wdf = pd.read_csv(need(path))
    print(f"[INFO] weights shape: {wdf.shape}  cols={list(wdf.columns)}")
    wdf = std_problem_id(wdf)
    wcol = detect_weight_col(wdf)
    if wcol != "w":
        wdf = wdf.rename(columns={wcol: "w"})
    wdf["w"] = pd.to_numeric(wdf["w"], errors="coerce")
    return wdf[["problem_id", "w"]]

def merge_weights_resilient(joined: pd.DataFrame, weights: pd.DataFrame | None) -> pd.DataFrame:
    df = joined.copy()
    # 既存の w があれば退避しておく
    existing_w_cols = [c for c in df.columns if c.lower() == "w"]
    if existing_w_cols:
        print(f"[INFO] joined 側に既存 'w' 列あり -> {existing_w_cols} を 'w_joined' へ退避")
        # 複数あれば最初だけ使う
        df = df.rename(columns={existing_w_cols[0]: "w_joined"})
        for c in existing_w_cols[1:]:
            if c != "w_joined":
                df = df.drop(columns=[c])

    if weights is not None:
        before = df.shape[0]
        df = df.merge(weights.rename(columns={"w":"w_weights"}), on="problem_id", how="left", validate="m:1")
        after = df.shape[0]
        if before != after:
            print(f"[WARN] merge 行数変化: before={before}, after={after}（キー重複の可能性）")

        # キー突合の確認
        miss_cnt = df["w_weights"].isna().sum()
        if miss_cnt > 0:
            total = df.shape[0]
            # 重み側にあるが joined に無い problem_id も出す
            missing_keys = set(weights["problem_id"]) - set(joined["problem_id"])
            print(f"[WARN] w_weights 欠損 {miss_cnt}/{total} 行（joinedに problem_id が無い）。例: {list(sorted(missing_keys))[:10]}")

        # w の確定（weights を最優先、無ければ joined の退避版、残りは 1.0）
        df["w"] = df["w_weights"]
        if "w_joined" in df.columns:
            df["w"] = df["w"].fillna(df["w_joined"])
        df["w"] = df["w"].fillna(1.0)

        # 後片付け
        drop_cols = [c for c in ["w_joined","w_weights"] if c in df.columns]
        if drop_cols:
            df = df.drop(columns=drop_cols)
    else:
        print("[WARN] 重みCSV無し -> 全て w=1.0 を採用")
        df["w"] = 1.0

    # 最終チェック
    if "w" not in df.columns:
        raise KeyError("最終的に 'w' 列が存在しません（統合に失敗）")

    # w のダイアグ
    print(f"[INFO] w 完成: min={df['w'].min():.6f}, max={df['w'].max():.6f}, mean={df['w'].mean():.6f}, isna={df['w'].isna().sum()}")
    # 参考: w 系列が残っていないか
    w_like = [c for c in df.columns if c.lower().startswith("w")]
    if len(w_like) > 1:
        print(f"[INFO] w系列（確認用）: {w_like}")
    return df

def choose_subset(df: pd.DataFrame, preferred_model: str, preferred_track: str) -> pd.DataFrame:
    sub = df[(df["model"] == preferred_model) & (df["track"] == preferred_track)].copy()
    if sub.empty:
        print(f"[WARN] 指定の組合せが無いため自動選択: {preferred_model=} {preferred_track=}")
        pairs = df[["model","track"]].drop_duplicates().values.tolist()
        if not pairs:
            raise ValueError("model/track の組合せが 0 件です")
        auto_model, auto_track = pairs[0]
        print(f"[INFO] 自動選択 -> model={auto_model}, track={auto_track}")
        sub = df[(df["model"] == auto_model) & (df["track"] == auto_track)].copy()
    print(f"[INFO] subset rows={len(sub)}  problems={sub['problem_id'].nunique()}  model={sub['model'].iloc[0]}  track={sub['track'].iloc[0]}")
    return sub

def corr_and_quartiles(df_one: pd.DataFrame, ks: list[int]):
    print("\n=== 相関（w vs pass@k） ===")
    for k in ks:
        col = f"pass_at_{k}"
        if col not in df_one.columns:
            print(f"[skip] {col} が無い")
            continue
        c = df_one[[col, "w"]].replace([np.inf,-np.inf], np.nan).dropna()
        if c.empty:
            print(f"[skip] {col}: データ無し")
            continue
        r = c["w"].corr(c[col])
        print(f"corr(w, {col}) = {r:.4f}（負が期待）  n={len(c)}")

    print("\n=== 四分位（w_bin=Q1..Q4 の pass@k 平均と件数） ===")
    wv = df_one["w"].replace([np.inf,-np.inf], np.nan).dropna()
    if len(wv) < 4:
        print("[WARN] w の有効データが少なく四分位不可")
        return
    q = np.quantile(wv, [0.25, 0.5, 0.75])
    def bin_w(v):
        if v <= q[0]: return "Q1"
        if v <= q[1]: return "Q2"
        if v <= q[2]: return "Q3"
        return "Q4"
    temp = df_one.copy()
    temp["w_bin"] = temp["w"].apply(bin_w)
    order = ["Q1","Q2","Q3","Q4"]
    for k in ks:
        col = f"pass_at_{k}"
        if col not in temp.columns:
            continue
        g = temp.groupby("w_bin")[col].mean().reindex(order)
        c = temp.groupby("w_bin")[col].count().reindex(order)
        print(f"\n[pass@{k}] mean by quartile:")
        print(g)
        print("[count per quartile]:")
        print(c)

def main():
    joined = load_joined(JOINED_CSV)
    try:
        weights = load_weights(WEIGHT_CSV)
    except Exception as e:
        print(f"[WARN] 重みCSVの読み込みに失敗: {e} -> w=1.0 を採用")
        weights = None

    df = merge_weights_resilient(joined, weights)
    # 参考: 左右どちらにしか無い problem_id を出しておく
    if weights is not None:
        only_in_weights = sorted(set(weights["problem_id"]) - set(joined["problem_id"]))
        only_in_joined  = sorted(set(joined["problem_id"])  - set(weights["problem_id"]))
        if only_in_weights: print(f"[INFO] 重み側のみの problem_id（一部）: {only_in_weights[:10]}")
        if only_in_joined:  print(f"[INFO] joined 側のみの problem_id（一部）: {only_in_joined[:10]}")

    sub = choose_subset(df, PREFERRED_MODEL, PREFERRED_TRACK)
    corr_and_quartiles(sub, PASS_KS)

if __name__ == "__main__":
    main()
