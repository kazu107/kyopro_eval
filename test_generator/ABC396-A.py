from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 95                  # 生成するテストケース総数
START_INDEX: int = 6                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC396/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC396/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 396            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC396 A） =========
MIN_N: int = 3
MAX_N: int = 100
VAL_MIN: int = 1
VAL_MAX: int = 100
# ======================================================

@dataclass
class Case:
    N: int
    A: List[int]

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def has_triple_run(A: List[int]) -> bool:
    """同じ要素が3つ以上連続する箇所があるか"""
    n = len(A)
    for i in range(n - 2):
        if A[i] == A[i+1] == A[i+2]:
            return True
    return False

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    return "Yes" if has_triple_run(case.A) else "No"

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル（問題ページ）
    cs.append(Case(5,  [1,4,4,4,2]))                    # Yes
    cs.append(Case(6,  [2,4,4,2,2,4]))                  # No
    cs.append(Case(8,  [1,4,2,5,7,7,7,2]))              # Yes
    cs.append(Case(10, [1,2,3,4,5,6,7,8,9,10]))         # No
    cs.append(Case(13, [1]*13))                         # Yes

    # --- N 最小/端値・等号境界
    cs.append(Case(3, [5,5,5]))                         # Yes（最小長で3連）
    cs.append(Case(3, [1,1,2]))                         # No（最小長で不成立）
    cs.append(Case(MAX_N, [VAL_MIN]*MAX_N))             # Yes（すべて同じ）
    cs.append(Case(MAX_N, [i%2+1 for i in range(MAX_N)])) # No（交互）

    # --- 位置バリエーション：先頭/中央/末尾で3連
    base = [2,2,2,3,4,5,6]
    cs.append(Case(len(base), base[:]))                 # 先頭で3連 -> Yes
    mid  = [9,8,7,7,7,6,5]
    cs.append(Case(len(mid), mid[:]))                   # 中央で3連 -> Yes
    tail = [3,4,5,6,7,8,8,8]
    cs.append(Case(len(tail), tail[:]))                 # 末尾で3連 -> Yes

    # --- 連続2個まで・分断されるパターン
    cs.append(Case(8, [4,4,5,4,4,6,6,7]))              # No（3連できない）
    cs.append(Case(7, [1,1,1,1,2,2,2]))                # Yes（複数の3連）

    # --- 値の端（1/100）での3連
    cs.append(Case(5, [1,1,1,100,100]))                # Yes
    cs.append(Case(6, [100,100, 99, 99, 99, 1]))       # Yes（中間3連）

    return cs

# ---------- ランダムケース ----------
def make_yes_case(rng: random.Random) -> Case:
    """必ず3連が存在する配列を構成"""
    N = rng.choice([MIN_N, MAX_N] + [rng.randint(4, MAX_N) for _ in range(2)])
    A = [rng.randint(VAL_MIN, VAL_MAX) for _ in range(N)]
    # 長さ L>=3 の連続ブロックを挿入
    L = rng.choice([3,3,3,4,5])  # 3 を厚めに
    L = min(L, N)
    pos = rng.randint(0, N - L)
    v = rng.randint(VAL_MIN, VAL_MAX)
    for i in range(L):
        A[pos + i] = v
    # 偶発的に前後が同値で L が伸びても OK（Yes 条件を満たす）
    return Case(N, A)

def make_no_case(rng: random.Random) -> Case:
    """3連が存在しない配列を構成"""
    N = rng.choice([MIN_N, MAX_N] + [rng.randint(4, MAX_N) for _ in range(2)])
    A: List[int] = []
    for i in range(N):
        if i >= 2 and A[i-1] == A[i-2]:
            # 直前2つが同じなら、同じ値を避ける
            forbidden = A[i-1]
            # 1..100 から forbidden を避けて選ぶ
            x = rng.randint(VAL_MIN, VAL_MAX-1)
            if x >= forbidden:
                x += 1
            A.append(x)
        else:
            A.append(rng.randint(VAL_MIN, VAL_MAX))
    assert not has_triple_run(A)
    return Case(N, A)

def make_random_case(rng: random.Random) -> Case:
    return make_yes_case(rng) if rng.random() < 0.5 else make_no_case(rng)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.N}\n{' '.join(map(str, case.A))}\n"

def write_case_files(abs_idx: int, width: int, case: Case,
                     in_dir: str, out_dir: str) -> Tuple[str, str]:
    in_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{INPUT_EXT}"
    out_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{OUTPUT_SUFFIX}"
    in_path = os.path.join(in_dir, in_name)
    out_path = os.path.join(out_dir, out_name)

    with open(in_path, "w", encoding="utf-8", newline="\n") as fi:
        fi.write(case_to_input_text(case))

    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(solve_case(case) + "\n")

    return in_path, out_path

def main() -> None:
    rng = random.Random(RNG_SEED)
    safe_mkdir(INPUT_DIR); safe_mkdir(OUTPUT_DIR)

    directed = make_directed_cases()

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁（終了番号で自動決定）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 出力（制約チェック込み）
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_N <= cs.N <= MAX_N
        assert len(cs.A) == cs.N
        assert all(VAL_MIN <= a <= VAL_MAX for a in cs.A)

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] N={cs.N}, has_triple={has_triple_run(cs.A)} -> {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
