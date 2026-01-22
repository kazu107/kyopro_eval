from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                   # 生成するテストケース総数
START_INDEX: int = 4                   # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC397/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC397/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"              # ファイル名プリフィクス
INPUT_EXT: str = ".txt"                # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"        # 出力ファイル拡張子
RNG_SEED: int | None = 397             # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC397 A） =========
T10_MIN: int = 300   # X=30.0 を 10 倍した整数
T10_MAX: int = 500   # X=50.0 を 10 倍した整数
# 分類しきい値（10倍整数で扱う）
HIGH_FEVER: int = 380  # X>=38.0 -> 1
FEVER_LO: int = 375    # 37.5<=X<38.0 -> 2
# ======================================================

@dataclass
class Case:
    X10: int   # 体温 X の 10 倍整数（例: 37.5 -> 375）

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力（1/2/3）を 10 倍整数で厳密判定"""
    x = case.X10
    if x >= HIGH_FEVER:
        return "1"
    elif x >= FEVER_LO:
        return "2"
    else:
        return "3"

# ---------- ユーティリティ ----------
def x10_to_str(x10: int) -> str:
    """10倍整数を '%.1f' 文字列へ"""
    return f"{x10 / 10:.1f}"

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル
    cs += [Case(400),  # 40.0 -> 1
           Case(377),  # 37.7 -> 2
           Case(366)]  # 36.6 -> 3

    # --- 端値・等号境界
    cs += [Case(300),  # 30.0 -> 3
           Case(500),  # 50.0 -> 1
           Case(380),  # 38.0 ちょうど -> 1
           Case(379),  # 37.9 -> 2
           Case(375),  # 37.5 ちょうど -> 2
           Case(374)]  # 37.4 -> 3

    # --- 代表例
    cs += [Case(381), Case(385), Case(395),  # 1 の範囲
           Case(376), Case(378),             # 2 の範囲
           Case(301), Case(325), Case(374)]  # 3 の範囲（374 は重複でもOK）

    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """出力 1/2/3 がバランスよく出るよう区間を選んで生成"""
    bucket = rng.choice([1, 2, 3])
    if bucket == 1:
        x10 = rng.randint(HIGH_FEVER, T10_MAX)           # [380,500]
    elif bucket == 2:
        x10 = rng.randint(FEVER_LO, HIGH_FEVER - 1)      # [375,379]
    else:
        x10 = rng.randint(T10_MIN, FEVER_LO - 1)         # [300,374]

    # ときどき境界直近を強調
    if rng.random() < 0.25:
        neighbors = [HIGH_FEVER-1, HIGH_FEVER, FEVER_LO, FEVER_LO-1, T10_MIN, T10_MAX]
        x10 = rng.choice([v for v in neighbors if T10_MIN <= v <= T10_MAX])
    return Case(x10)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{x10_to_str(case.X10)}\n"

def write_case_files(abs_idx: int, width: int, case: Case,
                     in_dir: str, out_dir: str) -> Tuple[str, str]:
    in_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{INPUT_EXT}"
    out_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{OUTPUT_SUFFIX}"
    in_path = os.path.join(in_dir, in_name)
    out_path = os.path.join(out_dir, out_name)

    with open(in_path, "w", encoding="utf-8", newline="\n") as fi:
        fi.write(case_to_input_text(case))

    ans = solve_case(case)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(ans + "\n")

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

    # ゼロ埋め桁数（少なくとも3桁、終了番号から自動決定）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 出力
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約・形式チェック（公式）
        assert T10_MIN <= cs.X10 <= T10_MAX
        # 入力文字列は常に小数第1位まで
        s = x10_to_str(cs.X10)
        assert s.count(".") == 1 and len(s.split(".")[1]) == 1

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] X={s} -> {solve_case(cs)} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
