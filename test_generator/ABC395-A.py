from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 96                     # 生成するテストケース総数
START_INDEX: int = 5                     # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC395/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC395/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"                # ファイル名プリフィクス
INPUT_EXT: str = ".txt"                  # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"          # 出力ファイル拡張子
RNG_SEED: int | None = 395               # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC395 A） =========
MIN_N: int = 2
MAX_N: int = 100
VAL_MIN: int = 1
VAL_MAX: int = 1000
# ======================================================

@dataclass
class Case:
    N: int
    A: List[int]

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def is_strictly_increasing(a: List[int]) -> bool:
    return all(a[i] < a[i+1] for i in range(len(a)-1))

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    return "Yes" if is_strictly_increasing(case.A) else "No"

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases(rng: random.Random) -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル（問題ページ） :contentReference[oaicite:1]{index=1}
    cs.append(Case(3, [1, 2, 5]))                         # Yes
    cs.append(Case(3, [3, 9, 5]))                         # No
    cs.append(Case(10, [1,1,2,3,5,8,13,21,34,55]))        # No

    # --- 最小/最大・端値
    cs.append(Case(2, [1, 2]))                            # Yes（最小N）
    cs.append(Case(2, [2, 1]))                            # No（降順）
    cs.append(Case(2, [1000, 1000]))                      # No（等号）
    cs.append(Case(MAX_N, list(range(1, MAX_N+1))))       # Yes（1ずつ増加）
    cs.append(Case(MAX_N, [VAL_MAX]*(MAX_N-1) + [VAL_MAX]))  # No（全等号）

    # --- 代表パターン
    cs.append(Case(6, [10, 20, 30, 40, 50, 60]))          # Yes（公差一定）
    cs.append(Case(6, [10, 20, 20, 21, 22, 23]))          # No（途中で=）
    cs.append(Case(7, [5, 7, 9, 8, 10, 12, 14]))          # No（途中で下降）
    cs.append(Case(7, [1, 1000, 1000, 1000, 1000, 1000, 1000]))  # No
    cs.append(Case(7, [1, 2, 1000, 1000, 1000, 1001-1, 1000]))   # No（=混在）

    # --- 大小端を使った Yes
    inc = [1, 2, 3, 10, 100, 500, 1000]
    cs.append(Case(len(inc), inc))

    return cs

# ---------- ランダムケース ----------
def make_yes_case(rng: random.Random) -> Case:
    """必ず狭義単調増加：distinct 値を昇順に並べる"""
    N = rng.choice([MIN_N, MAX_N] + [rng.randint(3, MAX_N-1) for _ in range(2)])
    # 1..1000 から N 個の相異なる数をサンプル -> 昇順
    vals = rng.sample(range(VAL_MIN, VAL_MAX + 1), k=N)
    vals.sort()
    return Case(N, vals)

def make_no_case(rng: random.Random) -> Case:
    """どこか1箇所以上で不成立（= または 下降）"""
    N = rng.choice([MIN_N, MAX_N] + [rng.randint(3, MAX_N-1) for _ in range(2)])
    # まず Yes 配列を作る
    vals = rng.sample(range(VAL_MIN, VAL_MAX + 1), k=N)
    vals.sort()
    # 壊す位置を選んで、= または 下降にする
    k = rng.randint(1, N-1)
    mode = rng.random()
    if mode < 0.5:
        # 等号にする
        vals[k] = vals[k-1]
    else:
        # 下降にする（前要素以下に下げる）
        lo = VAL_MIN
        hi = vals[k-1]  # <= 前要素
        vals[k] = rng.randint(lo, hi)
    return Case(N, vals)

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

    directed = make_directed_cases(rng)

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁。終了番号で自動決定
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 連番でファイル出力（制約チェック込み）
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_N <= cs.N <= MAX_N
        assert len(cs.A) == cs.N
        assert all(VAL_MIN <= x <= VAL_MAX for x in cs.A)

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] N={cs.N}, inc={is_strictly_increasing(cs.A)} -> {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
