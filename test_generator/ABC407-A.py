from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Tuple, List

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC407/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC407/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 407            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC407 A） =========
A_MIN: int = 1
A_MAX: int = 407
B_MIN: int = 1
B_MAX: int = 407  # B は奇数
# ======================================================

@dataclass
class Case:
    A: int
    B: int  # odd

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def nearest_int_div(A: int, B: int) -> int:
    """B は正の奇数。A/B に最も近い整数（正の A に対して整数演算でOK）"""
    return (A + B // 2) // B

def solve_case(case: Case) -> str:
    return str(nearest_int_div(case.A, case.B))

# ---------- ケース生成ユーティリティ ----------
def odd_between(lo: int, hi: int, rng: random.Random) -> int:
    """[lo, hi] から奇数を1つ返す"""
    lo_odd = lo if lo % 2 == 1 else lo + 1
    if lo_odd > hi:
        raise ValueError("no odd in range")
    k = rng.randrange(0, ((hi - lo_odd) // 2) + 1)
    return lo_odd + 2 * k

def make_directed_cases() -> List[Case]:
    """サンプル＋境界・代表パターンを網羅するハンドクラフト"""
    cases: List[Case] = []

    # --- 公式サンプル
    cases.append(Case(4, 7))     # -> 1
    cases.append(Case(407, 29))  # -> 14
    cases.append(Case(22, 11))   # -> 2

    # --- 端値＆等号境界
    cases.append(Case(1, 1))                 # 1/1=1 -> 1
    cases.append(Case(A_MAX, 1))             # 407/1=407 -> 407
    cases.append(Case(1, A_MAX if A_MAX % 2 else A_MAX-1))  # 小/大 -> 0 付近
    cases.append(Case(A_MAX, B_MAX if B_MAX % 2 else B_MAX-2))  # A/B が大きめ

    # --- 「しきい値」近傍（m と m+1 の境界 0.5B の前後）
    # B が奇数なので 0.5B は非整数。B//2 と B//2+1 で丸め方向が変わる。
    for B in [3, 5, 101, 407]:
        for m in [0, 1, 2, 10]:
            A0 = m * B + (B // 2)          # -> m に丸め
            A1 = m * B + (B // 2) + 1      # -> m+1 に丸め
            if 1 <= A0 <= A_MAX:
                cases.append(Case(A0, B))
            if 1 <= A1 <= A_MAX:
                cases.append(Case(A1, B))

    # --- A が B の倍数（商がちょうど整数）
    for B in [1, 3, 7, 11, 101]:
        for k in [1, 2, 10, 20, 300]:
            A = k * B
            if 1 <= A <= A_MAX and 1 <= B <= B_MAX:
                cases.append(Case(A, B))

    # --- A=1..10, いくつかの奇数 B で網羅
    for A in range(1, 11):
        for B in [1, 3, 9, 21, 101]:
            if B <= B_MAX:
                cases.append(Case(A, B))

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: しきい値付近を多めに混ぜる"""
    mode = rng.random()
    if mode < 0.5:
        # 素直に一様
        A = rng.randint(A_MIN, A_MAX)
        B = odd_between(B_MIN, B_MAX, rng)
    else:
        # しきい値付近: A ≈ k*B + (B//2)±{0,1}
        B = odd_between(B_MIN, B_MAX, rng)
        k_max = A_MAX // B
        if k_max == 0:
            k = 0
        else:
            k = rng.randint(0, k_max)
        off = (B // 2) + rng.choice([0, 1, -1, 2, -2])
        A = k * B + off
        if A < A_MIN:
            A = A_MIN
        if A > A_MAX:
            A = A_MAX
    return Case(A, B)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.A} {case.B}\n"

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

    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数（少なくとも3桁）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert A_MIN <= case.A <= A_MAX
        assert B_MIN <= case.B <= B_MAX and (case.B % 2 == 1)

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] A={case.A}, B={case.B} -> {solve_case(case)} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
