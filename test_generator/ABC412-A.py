from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC412/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC412/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 412            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC412 A） =========
MIN_N: int = 1
MAX_N: int = 100
VAL_MIN: int = 1
VAL_MAX: int = 100
# ======================================================

@dataclass
class Case:
    N: int
    P: List[Tuple[int, int]]  # (A_i, B_i)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力（B_i > A_i の日数）"""
    ans = sum(1 for a, b in case.P if b > a)
    return str(ans)

# ---------- ケース生成ユーティリティ ----------
def clamp(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))

def make_directed_cases() -> List[Case]:
    """網羅用ハンドクラフト（公式サンプル＋境界・代表パターン）"""
    cases: List[Case] = []

    # --- 公式サンプル（問題ページより）
    cases.append(Case(4, [(2,8), (5,5), (5,4), (6,7)]))  # 出力2
    cases.append(Case(5, [(1,1)]*5))                     # 出力0
    cases.append(Case(6, [(1,6),(2,5),(3,4),(4,3),(5,2),(6,1)]))  # 出力3

    # --- N最小/最大、端値
    cases.append(Case(1, [(1,100)]))                     # 全勝 -> 1
    cases.append(Case(1, [(100,1)]))                     # 全敗 -> 0
    cases.append(Case(MAX_N, [(1,100)]*MAX_N))           # すべて勝ち -> 100
    cases.append(Case(MAX_N, [(100,1)]*MAX_N))           # すべて負け -> 0
    cases.append(Case(MAX_N, [(50,50)]*MAX_N))           # すべて同点 -> 0

    # --- 境界近傍（= を含む）
    cases.append(Case(6, [(99,100),(100,100),(100,99),(1,1),(1,2),(2,1)]))  # 2

    # --- 混在パターン
    mix = [(10,9),(20,20),(30,31),(40,39),(50,50),(60,70),(70,60),(80,81)]
    cases.append(Case(len(mix), mix))  # 勝ち: (30,31),(60,70),(80,81) -> 3

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: N と (A,B) の分布をいくつかに振り分け"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)]
    N = rng.choice(n_choices)

    P: List[Tuple[int,int]] = []
    for i in range(N):
        mode = rng.random()
        if mode < 0.25:
            # A 周辺に B を ±2 で寄せる（同点や±1,±2 を出しやすく）
            a = rng.randint(VAL_MIN, VAL_MAX)
            delta = rng.randint(-2, 2)
            b = clamp(a + delta, VAL_MIN, VAL_MAX)
        elif mode < 0.50:
            # 勝ち寄り
            a = rng.randint(VAL_MIN, VAL_MAX-1)
            b = rng.randint(a+1, VAL_MAX)
        elif mode < 0.75:
            # 負け寄り
            b = rng.randint(VAL_MIN, VAL_MAX-1)
            a = rng.randint(b+1, VAL_MAX)
        else:
            # 一様
            a = rng.randint(VAL_MIN, VAL_MAX)
            b = rng.randint(VAL_MIN, VAL_MAX)
        P.append((a, b))

    return Case(N, P)

def case_to_input_text(case: Case) -> str:
    lines = [str(case.N)]
    lines += [f"{a} {b}" for a, b in case.P]
    return "\n".join(lines) + "\n"

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

        # 制約チェック（公式）: 1 ≤ N ≤ 100, 1 ≤ A_i, B_i ≤ 100
        assert MIN_N <= case.N <= MAX_N
        assert len(case.P) == case.N
        assert all(VAL_MIN <= a <= VAL_MAX and VAL_MIN <= b <= VAL_MAX for a, b in case.P)

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (ans={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
