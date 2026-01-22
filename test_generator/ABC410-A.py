from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC410/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC410/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 410            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC410 A） =========
MIN_N: int = 1
MAX_N: int = 100
VAL_MIN: int = 1
VAL_MAX: int = 100
# ======================================================

@dataclass
class Case:
    N: int
    A: List[int]
    K: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力（K 歳の馬が出場可能なレース数）"""
    ans = sum(1 for a in case.A if a >= case.K)
    return str(ans)

# ---------- ケース生成ユーティリティ ----------
def make_directed_cases() -> List[Case]:
    """公式サンプル＋境界・代表パターン"""
    cases: List[Case] = []

    # --- 公式サンプル
    cases.append(Case(5, [3, 1, 4, 1, 5], 4))  # 出力2
    cases.append(Case(1, [1], 100))            # 出力0
    cases.append(Case(15, [18,89,31,2,15,93,64,78,58,19,79,59,24,50,30], 38))  # 出力8

    # --- N最小/最大・端値
    cases.append(Case(1, [100], 1))                           # 1
    cases.append(Case(1, [1], 1))                             # 1
    cases.append(Case(MAX_N, [VAL_MAX]*MAX_N, VAL_MAX))       # 100
    cases.append(Case(MAX_N, [VAL_MIN]*MAX_N, VAL_MIN+1))     # 0
    cases.append(Case(MAX_N, list(range(1, MAX_N+1)), 50))    # 51 (50..100)
    cases.append(Case(MAX_N, list(range(MAX_N, 0, -1)), 100)) # 1

    # --- 境界近傍（= を含む）
    cases.append(Case(5, [10,10,9,11,10], 10))  # 4
    cases.append(Case(5, [9,9,9,9,9], 10))      # 0
    cases.append(Case(5, [10,11,12,13,14], 10)) # 5

    # --- 混在・偏り
    cases.append(Case(8, [1,100,50,49,51,2,99,98], 50))  # 5
    cases.append(Case(10, [20]*10, 21))                  # 0
    cases.append(Case(10, [20]*10, 20))                  # 10

    return cases

def rand_array(rng: random.Random, n: int) -> List[int]:
    """[1,100] の整数列。特殊形を混ぜる。"""
    mode = rng.random()
    if mode < 0.15:
        v = rng.randint(VAL_MIN, VAL_MAX)
        return [v]*n                     # 全要素同一
    elif mode < 0.30:
        lo = rng.randint(1, 40)
        hi = rng.randint(60, 100)
        return [rng.randint(lo, hi) for _ in range(n)]  # 中央寄り
    elif mode < 0.45:
        # 低めに寄せる
        return [rng.randint(VAL_MIN, rng.randint(5, 30)) for _ in range(n)]
    elif mode < 0.60:
        # 高めに寄せる
        lo = rng.randint(70, 95)
        return [rng.randint(lo, VAL_MAX) for _ in range(n)]
    elif mode < 0.75 and n <= (VAL_MAX - VAL_MIN + 1):
        # なるべくユニーク
        vals = list(range(VAL_MIN, VAL_MAX + 1))
        rng.shuffle(vals)
        return vals[:n]
    else:
        return [rng.randint(VAL_MIN, VAL_MAX) for _ in range(n)]

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: N や K の極端、境界一致を多めに発生させる"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)]
    N = rng.choice(n_choices)
    A = rand_array(rng, N)

    # K の選び方をバラけさせる（境界/極端/一様）
    r = rng.random()
    if r < 0.33:
        # A のいずれか or その±1（境界付近を作りやすく）
        base = rng.choice(A)
        delta = rng.choice([-1, 0, 0, 0, +1])  # 0 多め
        K = max(VAL_MIN, min(VAL_MAX, base + delta))
    elif r < 0.66:
        # 極端
        K = rng.choice([VAL_MIN, VAL_MAX])
    else:
        # 一様
        K = rng.randint(VAL_MIN, VAL_MAX)

    return Case(N, A, K)

def case_to_input_text(case: Case) -> str:
    lines = [str(case.N)]
    lines.append(" ".join(map(str, case.A)))
    lines.append(str(case.K))
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

        # 制約チェック（公式）
        assert MIN_N <= case.N <= MAX_N
        assert len(case.A) == case.N
        assert all(VAL_MIN <= a <= VAL_MAX for a in case.A)
        assert VAL_MIN <= case.K <= VAL_MAX

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (K={case.K}, >=K={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
