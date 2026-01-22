from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                   # 生成するテストケース総数
START_INDEX: int = 4                   # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC406/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC406/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"              # ファイル名プリフィクス
INPUT_EXT: str = ".txt"                # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"        # 出力ファイル保存先拡張子
RNG_SEED: int | None = 406             # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC406 A） =========
H_MIN: int = 0
H_MAX: int = 23
M_MIN: int = 0
M_MAX: int = 59
DAY_MINUTES: int = 24 * 60
# ======================================================

@dataclass
class Case:
    A: int; B: int   # 締切 A:B
    C: int; D: int   # 提出 C:D

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_minutes(h: int, m: int) -> int:
    return 60 * h + m

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    deadline = to_minutes(case.A, case.B)
    submitted = to_minutes(case.C, case.D)
    return "Yes" if submitted < deadline else "No"

# ---------- ハンドクラフト（サンプル＋境界網羅） ----------
def make_directed_cases() -> List[Case]:
    cases: List[Case] = []

    # --- 公式サンプル
    cases.append(Case(22, 40, 22, 30))  # Yes
    cases.append(Case(22, 40, 22, 45))  # No
    cases.append(Case(12,  0, 11, 30))  # Yes

    # --- 端値・等号境界（分だけ違う／時間だけ違う）
    cases.append(Case(0,  1, 0,  0))    # Yes（同時刻は不可条件なので1分差）
    cases.append(Case(0,  0, 0,  1))    # No
    cases.append(Case(23, 0, 22, 59))   # Yes（時間跨ぎ手前）
    cases.append(Case(0, 59, 1,  0))    # No（分59→次の時間）
    cases.append(Case(23,59, 0,  0))    # No（同日内で遅い提出）

    # --- 同一時刻は与えないことの確認（近傍ケース）
    cases.append(Case(10,  0, 9, 59))   # Yes（直前）
    cases.append(Case(10,  0, 10, 1))   # No（直後）
    cases.append(Case(7,  30, 7, 29))   # Yes（同時刻-1分）
    cases.append(Case(7,  30, 7, 31))   # No（同時刻+1分）

    # --- ランダムでは出にくいきわどい境界
    cases.append(Case(0,  0, 0, 59))    # No（最小締切に対して遅い）
    cases.append(Case(23,59, 23,58))    # Yes（最大締切直前）

    return cases

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """Yes/No 半々くらい。(A,B)!=(C,D) を常に満たす。"""
    # ランダムに締切を決める
    A = rng.randint(H_MIN, H_MAX)
    B = rng.randint(M_MIN, M_MAX)
    dead_min = to_minutes(A, B)

    want_yes = rng.random() < 0.5
    if want_yes and dead_min > 0:
        s_min = rng.randint(0, dead_min - 1)   # 締切より前に提出
    elif (not want_yes) and dead_min < DAY_MINUTES - 1:
        s_min = rng.randint(dead_min + 1, DAY_MINUTES - 1)  # 締切より後
    else:
        # 片側が作れない極端（dead_min が 0 or 1439）のときのフォールバック
        if dead_min == 0:
            s_min = rng.randint(1, DAY_MINUTES - 1)         # 必ず後
        else:
            s_min = rng.randint(0, dead_min - 1)            # 必ず前

    C, D = divmod(s_min, 60)
    return Case(A, B, C, D)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.A} {case.B} {case.C} {case.D}\n"

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
        assert H_MIN <= cs.A <= H_MAX and H_MIN <= cs.C <= H_MAX
        assert M_MIN <= cs.B <= M_MAX and M_MIN <= cs.D <= M_MAX
        assert not (cs.A == cs.C and cs.B == cs.D)  # (A,B)!=(C,D)

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] ({cs.A:02d}:{cs.B:02d}) vs ({cs.C:02d}:{cs.D:02d}) -> {solve_case(cs)} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
