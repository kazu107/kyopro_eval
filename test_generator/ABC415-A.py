from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97               # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC415/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC415/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 415            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC415 A） =========
MAX_N: int = 100
MIN_N: int = 1
VAL_MIN: int = 1
VAL_MAX: int = 100
# ======================================================

@dataclass
class Case:
    N: int
    A: List[int]
    X: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力を計算 ('Yes' / 'No')"""
    return "Yes" if case.X in case.A else "No"

# ---------- ケース生成ユーティリティ ----------
def make_directed_cases() -> List[Case]:
    """網羅用のハンドクラフトケース（サンプル相当＋境界・代表パターン）"""
    cases: List[Case] = []

    # --- サンプル（公式）
    cases.append(Case(5, [3, 1, 4, 1, 5], 4))                     # Yes
    cases.append(Case(4, [100, 100, 100, 100], 100))              # Yes
    cases.append(Case(6, [2, 3, 5, 7, 11, 13], 1))                # No

    # --- N最小/最大、端値・重複
    cases.append(Case(1, [1], 1))                                 # Yes: 最小
    cases.append(Case(1, [100], 1))                               # No: 最小N & 端値
    cases.append(Case(MAX_N, [1]*MAX_N, 1))                       # Yes: 全て同じ=1
    cases.append(Case(MAX_N, [100]*MAX_N, 99))                    # No : 全て同じ=100

    # --- 並びパターン
    cases.append(Case(10, [1,2,3,4,5,6,7,8,9,10], 10))            # Yes: 昇順
    cases.append(Case(10, [10,9,8,7,6,5,4,3,2,1], 11-1))          # Yes: 降順かつ末尾一致
    cases.append(Case(10, [10,9,8,7,6,5,4,3,2,1], 11))            # No : 範囲外ではなく未出現(=11は使わないので置換)
    cases[-1] = Case(10, [10,9,8,7,6,5,4,3,2,1], 100)             # 修正: 有効範囲の未出現値

    # --- 多重出現
    cases.append(Case(7, [42, 2, 42, 3, 4, 42, 5], 42))           # Yes: 3回出現
    cases.append(Case(7, [42, 2, 42, 3, 4, 42, 5], 6))            # No

    # --- 全種網羅系
    cases.append(Case(99, list(range(1, 100)), 100))              # No: 1..99 のみ
    cases.append(Case(100, list(range(1, 101)), 1))               # Yes: 1..100 全部
    cases.append(Case(2, [1, 100], 1))                            # Yes: 先頭一致
    cases.append(Case(2, [1, 100], 100))                          # Yes: 末尾一致
    cases.append(Case(2, [1, 100], 50))                           # No : 離散

    return cases

def rand_array(rng: random.Random, n: int) -> List[int]:
    """[VAL_MIN, VAL_MAX] の整数から重複多めで配列を作る。時々特殊形を混ぜる。"""
    mode = rng.random()
    if mode < 0.15:
        # 全要素同一
        v = rng.randint(VAL_MIN, VAL_MAX)
        return [v] * n
    elif mode < 0.30 and n <= (VAL_MAX - VAL_MIN + 1):
        # できるだけユニーク（シャッフル）
        vals = list(range(VAL_MIN, VAL_MAX + 1))
        rng.shuffle(vals)
        return vals[:n]
    else:
        # 通常: 重複多め
        return [rng.randint(VAL_MIN, VAL_MAX) for _ in range(n)]

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: 極端な N を混ぜつつ Yes/No をバランスよく作る"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)]
    N = rng.choice(n_choices)
    A = rand_array(rng, N)

    want_yes = rng.random() < 0.5  # Yes/No 半々くらい
    if want_yes:
        X = rng.choice(A)
    else:
        all_vals = set(range(VAL_MIN, VAL_MAX + 1))
        diff = list(all_vals - set(A))
        if diff:
            X = rng.choice(diff)
        else:
            # A が 1..100 を全て含む場合は No が作れないため Yes にフォールバック
            X = rng.choice(A)

    return Case(N, A, X)

def case_to_input_text(case: Case) -> str:
    lines = [str(case.N)]
    lines.append(" ".join(map(str, case.A)))
    lines.append(str(case.X))
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

        # 制約チェック（公式）: 1 ≤ N ≤ 100, 1 ≤ A_i ≤ 100, 1 ≤ X ≤ 100
        assert MIN_N <= case.N <= MAX_N
        assert all(VAL_MIN <= a <= VAL_MAX for a in case.A)
        assert VAL_MIN <= case.X <= VAL_MAX
        assert len(case.A) == case.N

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  ({solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
