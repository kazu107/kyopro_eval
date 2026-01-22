from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC403/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC403/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル保存先拡張子
RNG_SEED: int | None = 403            # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC403 A） =========
MIN_N: int = 1
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

def solve_case(case: Case) -> str:
    """期待出力（奇数番目(1-indexed)の総和）"""
    s = sum(case.A[::2])  # 0-indexの偶数位置 = 1-indexの奇数番目
    return str(s)

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル（問題ページ）
    cs.append(Case(7,  [3,1,4,1,5,9,2]))  # 3+4+5+2=14
    cs.append(Case(1,  [100]))            # 100
    cs.append(Case(14, [100,10,1,10,100,10,1,10,100,10,1,10,100,10]))  # 403

    # --- N最小/最大・端値
    cs.append(Case(1, [1]))                                  # 最小
    cs.append(Case(MAX_N, [VAL_MAX]*MAX_N))                  # 100 を 50 回 -> 5000
    cs.append(Case(MAX_N, [VAL_MIN]*MAX_N))                  # 1 を 50 回 -> 50

    # --- 昇順/降順/交互
    cs.append(Case(10, list(range(1, 11))))                  # 1+3+5+7+9=25
    cs.append(Case(10, list(range(10, 0, -1))))              # 10,8,6,4,2 の総和=30
    cs.append(Case(9,  [100 if i%2==0 else 1 for i in range(9)]))  # 100*5=500
    cs.append(Case(9,  [1 if i%2==0 else 100 for i in range(9)]))  # 1*5=5

    # --- ランダムでは出にくい境界（N奇数/偶数の差）
    cs.append(Case(8,  [VAL_MAX]*8))                         # 100*4=400
    cs.append(Case(9,  [VAL_MAX]*9))                         # 100*5=500

    return cs

# ---------- ランダムケース ----------
def rand_array(rng: random.Random, n: int) -> List[int]:
    """[VAL_MIN, VAL_MAX] の整数配列。偏りのある分布を混ぜる。"""
    mode = rng.random()
    if mode < 0.2:
        # 全要素同一
        v = rng.randint(VAL_MIN, VAL_MAX)
        return [v]*n
    elif mode < 0.4:
        # 低め中心
        return [rng.randint(VAL_MIN, rng.randint(10, 40)) for _ in range(n)]
    elif mode < 0.6:
        # 高め中心
        lo = rng.randint(60, 95)
        return [rng.randint(lo, VAL_MAX) for _ in range(n)]
    elif mode < 0.8:
        # 交互に大/小
        big = rng.randint(80, 100)
        small = rng.randint(1, 5)
        return [big if i%2==0 else small for i in range(n)]
    else:
        # 一様
        return [rng.randint(VAL_MIN, VAL_MAX) for _ in range(n)]

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: 極端な N を混ぜつつ、多様な並びを作る。"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(4)]
    N = rng.choice(n_choices)
    A = rand_array(rng, N)
    return Case(N, A)

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

    # 必要数までランダム補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数（少なくとも3桁）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_N <= cs.N <= MAX_N
        assert len(cs.A) == cs.N
        assert all(VAL_MIN <= a <= VAL_MAX for a in cs.A)

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] N={cs.N}, sum_odd={solve_case(cs)}  -> {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
