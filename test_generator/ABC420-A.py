"""
AtCoder ABC420 A - What month is it? 用テストケース自動生成スクリプト
問題: 1..12 の整数 X, Y が与えられる。X 月の Y ヶ月後が何月か出力せよ。
答えは ((X-1) + Y) % 12 + 1 で求まる。  (例: 5 月の 9 ヶ月後は 2)
参照: https://atcoder.jp/contests/abc420/tasks/abc420_a

生成ファイル:
    inputs/case001.txt, outputs/case001.out.txt などのように連番で作成
"""

from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 100                 # 生成するテストケース総数
START_INDEX: int = 4                 # 連番の開始番号（例: 1 -> case001 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC420/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC420/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 420           # 乱数シード (再現性が必要なら整数、完全ランダムなら None)
INCLUDE_FULL_GRID: bool = True       # True: (X,Y)=(1..12,1..12)全144通りを体系的に用意（NUM_CASES で切り詰め）
# ======================================================

@dataclass(frozen=True)
class Case:
    X: int
    Y: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> int:
    """期待出力（1..12 の整数）"""
    return ((case.X - 1) + case.Y) % 12 + 1

def case_to_input_text(case: Case) -> str:
    return f"{case.X} {case.Y}\n"

def write_case_files(abs_idx: int, width: int, case: Case, in_dir: str, out_dir: str) -> Tuple[str, str]:
    in_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{INPUT_EXT}"
    out_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{OUTPUT_SUFFIX}"
    in_path = os.path.join(in_dir, in_name)
    out_path = os.path.join(out_dir, out_name)

    with open(in_path, "w", encoding="utf-8", newline="\n") as f_in:
        f_in.write(case_to_input_text(case))

    ans = solve_case(case)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write(str(ans) + "\n")

    return in_path, out_path

# -- 手作り（サンプル相当＋境界代表） -------------------------
def make_directed_cases() -> List[Case]:
    cases: List[Case] = []
    # サンプル
    cases += [Case(5, 9), Case(1, 1), Case(12, 12)]
    # 境界/代表
    cases += [
        Case(1, 12),   # 1→(12ヶ月後)=1
        Case(12, 1),   # 12→(1ヶ月後)=1 に回る
        Case(6, 6),    # 対称例
        Case(2, 11),   # 2→(11ヶ月後)=1
        Case(11, 2),   # 11→(2ヶ月後)=1
        Case(3, 9),    # wrap
        Case(9, 3),    # wrap
        Case(4, 8),
        Case(8, 4),
        Case(7, 5),
        Case(5, 7),
        Case(10, 10),
    ]
    # 1..12 をそれぞれ始点に持つケース（Y=1 と Y=12）
    for x in range(1, 13):
        cases.append(Case(x, 1))
        cases.append(Case(x, 12))
    # Y を1..12一通り（Xは固定と回転）
    for y in range(1, 13):
        cases.append(Case(1, y))
        cases.append(Case(12, y))
    return cases

# -- 体系的: 全グリッド (X=1..12, Y=1..12) -------------------
def make_full_grid_cases() -> List[Case]:
    return [Case(x, y) for x in range(1, 13) for y in range(1, 13)]

# -- ランダム補充 ---------------------------------------------
def make_random_case(rng: random.Random) -> Case:
    return Case(rng.randint(1, 12), rng.randint(1, 12))

def main() -> None:
    rng = random.Random(RNG_SEED)

    safe_mkdir(INPUT_DIR)
    safe_mkdir(OUTPUT_DIR)

    cases: List[Case] = []
    # 体系的に入れる（必要なら）
    if INCLUDE_FULL_GRID:
        cases.extend(make_full_grid_cases())

    # 手作りで網羅強化
    cases.extend(make_directed_cases())

    # 重複排除
    cases = list(dict.fromkeys(cases))

    # 必要数までランダムで補充
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))

    # 先頭から NUM_CASES 件に切り詰め
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数（少なくとも3桁）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 出力
    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset
        # 制約チェック
        assert 1 <= case.X <= 12 and 1 <= case.Y <= 12
        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] X={case.X:2d} Y={case.Y:2d} -> {solve_case(case):2d}  | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")
    print(f"Grid   : {'ON' if INCLUDE_FULL_GRID else 'OFF'}  (RNG_SEED={RNG_SEED})")

if __name__ == "__main__":
    main()
