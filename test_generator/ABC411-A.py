from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC411/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC411/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 411            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC411 A） =========
MAX_LEN: int = 100                    # |P| と L の上限
MIN_LEN: int = 1                      # |P| と L の下限
LOWER = string.ascii_lowercase        # 'a'..'z'
# ======================================================

@dataclass
class Case:
    P: str
    L: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    return "Yes" if len(case.P) >= case.L else "No"

# ---------- ケース生成ユーティリティ ----------
def rand_len(rng: random.Random, lo: int = MIN_LEN, hi: int = MAX_LEN) -> int:
    """1..100 の長さを、極端長(1,100)を出しやすく混合分布で返す。"""
    t = rng.random()
    if t < 0.20:
        return lo
    if t < 0.40:
        return hi
    if t < 0.55:
        return rng.randint(2, 5)
    if t < 0.70:
        return rng.randint(95, hi)
    return rng.randint(lo, hi)

def rand_lower_str(rng: random.Random, L: int) -> str:
    return "".join(rng.choice(LOWER) for _ in range(L))

def make_directed_cases() -> List[Case]:
    """サンプル＋境界・代表パターンを網羅するハンドクラフト"""
    cases: List[Case] = []

    # --- 公式サンプル（問題ページより）
    cases.append(Case("chokudai", 5))   # Yes
    cases.append(Case("ac", 3))         # No
    cases.append(Case("atcoder", 7))    # Yes

    # --- 最小・最大・等号境界
    cases.append(Case("a", 1))                          # 最小長 & ちょうど -> Yes
    cases.append(Case("a", 2))                          # L が |P| を超える -> No
    cases.append(Case("a"*MAX_LEN, MAX_LEN))            # 最大長 & ちょうど -> Yes
    cases.append(Case("z"*(MAX_LEN-1), MAX_LEN))        # 1 だけ不足 -> No

    # --- 代表パターン（長さ比較が錯覚しやすいもの）
    cases.append(Case("aaaaab", 6))                     # 6=6 -> Yes
    cases.append(Case("aaaaab", 7))                     # 6<7 -> No
    cases.append(Case("abc", 2))                        # 3>=2 -> Yes
    cases.append(Case("xyz", 4))                        # 3<4 -> No
    cases.append(Case("m"*50, 50))                      # 50=50 -> Yes
    cases.append(Case("n"*50, 51))                      # 50<51 -> No

    # --- ランダムでは出にくい端寄せ
    cases.append(Case("b"*2, 1))                        # L=1 -> Yes
    cases.append(Case("c"*2, 100))                      # 2<100 -> No

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: Yes/No をバランスよく作成。"""
    # まず P を作る
    p_len = rand_len(rng)
    P = rand_lower_str(rng, p_len)

    # Yes/No のバランスを制御して L を決める
    if rng.random() < 0.5:
        # Yes にする: L ∈ [1, |P|]
        L = rng.randint(MIN_LEN, p_len)
    else:
        # No にする: L ∈ [|P|+1, 100]（可能なら）
        if p_len < MAX_LEN:
            L = rng.randint(p_len + 1, MAX_LEN)
        else:
            # 既に |P|=100 なら No が作れないので Yes にフォールバック
            L = rng.randint(MIN_LEN, p_len)
    return Case(P, L)

def case_to_input_text(case: Case) -> str:
    return f"{case.P}\n{case.L}\n"

def write_case_files(abs_idx: int, width: int, case: Case, in_dir: str, out_dir: str) -> Tuple[str, str]:
    """abs_idx は連番の絶対番号（START_INDEX からの値）"""
    in_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{INPUT_EXT}"
    out_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{OUTPUT_SUFFIX}"
    in_path = os.path.join(in_dir, in_name)
    out_path = os.path.join(out_dir, out_name)

    with open(in_path, "w", encoding="utf-8", newline="\n") as f_in:
        f_in.write(case_to_input_text(case))

    ans = solve_case(case)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write(ans + "\n")

    return in_path, out_path

def main() -> None:
    rng = random.Random(RNG_SEED)

    safe_mkdir(INPUT_DIR)
    safe_mkdir(OUTPUT_DIR)

    directed = make_directed_cases()

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))

    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁。終了番号を考慮して自動決定
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # START_INDEX から連番でファイルを書き出し
    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_LEN <= len(case.P) <= MAX_LEN and case.P.islower() and case.P.isalpha()
        assert MIN_LEN <= case.L <= MAX_LEN

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (|P|={len(case.P)}, L={case.L}, ans={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
