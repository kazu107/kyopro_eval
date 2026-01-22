from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 96                  # 生成するテストケース総数
START_INDEX: int = 5                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC405/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC405/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル保存先拡張子
RNG_SEED: int | None = 405            # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC405 A） =========
R_MIN: int = 1
R_MAX: int = 4229
X_MIN: int = 1
X_MAX: int = 2
DIV1_LO: int = 1600
DIV1_HI: int = 2999
DIV2_LO: int = 1200
DIV2_HI: int = 2399
# ======================================================

@dataclass
class Case:
    R: int
    X: int  # 1 or 2

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def is_rated(R: int, X: int) -> bool:
    if X == 1:
        return DIV1_LO <= R <= DIV1_HI
    else:
        return DIV2_LO <= R <= DIV2_HI

def solve_case(case: Case) -> str:
    return "Yes" if is_rated(case.R, case.X) else "No"

# ---------- ハンドクラフト（サンプル＋境界網羅） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # 公式サンプル
    cs += [Case(2000, 1),  # Yes
           Case(1000, 1),  # No
           Case(1500, 2),  # Yes
           Case(2800, 2)]  # No

    # Div.1 境界
    cs += [Case(1599, 1),  # No
           Case(1600, 1),  # Yes
           Case(2999, 1),  # Yes
           Case(3000, 1)]  # No

    # Div.2 境界
    cs += [Case(1199, 2),  # No
           Case(1200, 2),  # Yes
           Case(2399, 2),  # Yes
           Case(2400, 2)]  # No

    # 端値
    cs += [Case(R_MIN, 1), Case(R_MIN, 2),  # 最小R
           Case(R_MAX, 1), Case(R_MAX, 2)]  # 最大R

    # 代表例いくつか
    cs += [Case(1700, 1), Case(1700, 2),
           Case(1300, 2), Case(1300, 1),
           Case(2500, 1), Case(2500, 2)]
    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """Yes/No をバランスよく、しきい値付近を多めに混ぜる。"""
    want_yes = rng.random() < 0.5
    X = rng.randint(X_MIN, X_MAX)

    if want_yes:
        # Rated になる R を選ぶ
        if X == 1:
            R = rng.randint(DIV1_LO, DIV1_HI)
        else:
            R = rng.randint(DIV2_LO, DIV2_HI)
    else:
        # Rated にならない R（各Divの外側）を選ぶ。境界±1 を出しやすく。
        if X == 1:
            pool = []
            if R_MIN <= DIV1_LO - 1:
                pool += [rng.randint(R_MIN, DIV1_LO - 1)]
            if DIV1_HI + 1 <= R_MAX:
                # 境界周りを厚めに
                pool += [min(R_MAX, DIV1_HI + d) for d in [1, 2, 10, 500] if DIV1_HI + d <= R_MAX]
                pool += [rng.randint(DIV1_HI + 1, R_MAX)]
            R = rng.choice(pool)
        else:
            pool = []
            if R_MIN <= DIV2_LO - 1:
                pool += [rng.randint(R_MIN, DIV2_LO - 1)]
            if DIV2_HI + 1 <= R_MAX:
                pool += [min(R_MAX, DIV2_HI + d) for d in [1, 2, 10, 500] if DIV2_HI + d <= R_MAX]
                pool += [rng.randint(DIV2_HI + 1, R_MAX)]
            R = rng.choice(pool)

    # ときどき境界直近を明示的に追加
    if rng.random() < 0.25:
        if X == 1:
            R = rng.choice([DIV1_LO - 1, DIV1_LO, DIV1_HI, DIV1_HI + 1])
            R = min(max(R, R_MIN), R_MAX)
        else:
            R = rng.choice([DIV2_LO - 1, DIV2_LO, DIV2_HI, DIV2_HI + 1])
            R = min(max(R, R_MIN), R_MAX)

    return Case(R, X)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.R} {case.X}\n"

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
        assert R_MIN <= cs.R <= R_MAX
        assert X_MIN <= cs.X <= X_MAX

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] R={cs.R}, X={cs.X} -> {solve_case(cs)} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
