from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                 # 生成するテストケース総数
START_INDEX: int = 4                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC400/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC400/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 400           # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC400 A） =========
A_MIN: int = 1
A_MAX: int = 400
TOTAL: int = 400                     # 並べたい人数（固定）
DIVS_400: Tuple[int, ...] = (        # 400 の約数（検証・生成に使用）
    1,2,4,5,8,10,16,20,25,40,50,80,100,200,400
)
# ======================================================

@dataclass
class Case:
    A: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：A*B=400 を満たす B（存在しなければ -1）"""
    a = case.A
    if TOTAL % a == 0:
        return str(TOTAL // a)
    return "-1"

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル（問題ページ）
    cs.append(Case(10))    # -> 40
    cs.append(Case(11))    # -> -1
    cs.append(Case(400))   # -> 1

    # --- 端値・等号境界
    cs.append(Case(1))     # -> 400
    cs.append(Case(2))     # -> 200
    cs.append(Case(5))     # -> 80
    cs.append(Case(25))    # -> 16
    cs.append(Case(200))   # -> 2
    cs.append(Case(399))   # -> -1
    cs.append(Case(3))     # -> -1
    cs.append(Case(7))     # -> -1
    cs.append(Case(17))    # -> -1
    cs.append(Case(50))    # -> 8

    # --- 約数・非約数を多めに網羅
    for a in DIVS_400:
        if a not in [1,2,5,10,25,50,200,400]:  # 既出を除いて追加
            cs.append(Case(a))
    # 非約数の代表
    for a in [6,9,12,13,14,15,18,19,21,22,24,26,27,28,29,31,33,34,35,36]:
        cs.append(Case(a))

    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """Yes（約数）/No（非約数）を半々くらいで作る"""
    if rng.random() < 0.5:
        # 約数から選ぶ
        A = rng.choice(DIVS_400)
    else:
        # 1..400 から約数を除いて選ぶ
        while True:
            A = rng.randint(A_MIN, A_MAX)
            if A not in DIVS_400:
                break
    return Case(A)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.A}\n"

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

    # ゼロ埋め桁数（少なくとも3桁、終了番号に合わせる）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 出力
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）: 1 ≤ A ≤ 400
        assert A_MIN <= cs.A <= A_MAX

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] A={cs.A:3d} -> {solve_case(cs):>3} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
