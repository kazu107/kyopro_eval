from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 98                 # 生成するテストケース総数
START_INDEX: int = 3                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC398/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC398/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 398           # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC398 A） =========
N_MIN: int = 1
N_MAX: int = 100
# ======================================================

@dataclass
class Case:
    N: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：条件を満たす一意の文字列
       N が奇数 -> ---=---, 偶数 -> --==--（中央に = / ==）"""
    N = case.N
    if N % 2 == 1:
        k = N // 2
        return "-" * k + "=" + "-" * k
    else:
        k = N // 2 - 1
        return "-" * k + "==" + "-" * k

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []
    # 公式サンプル
    cs.append(Case(4))   # -> -==-
    cs.append(Case(7))   # -> ---=---

    # 端値
    cs.append(Case(1))   # -> =
    cs.append(Case(2))   # -> ==
    cs.append(Case(99))  # 奇数最大近傍
    cs.append(Case(100)) # 偶数最大

    # 代表例（奇数/偶数をバランスよく）
    for n in [3,5,6,8,9,10,25,26,49,50,73,74]:
        cs.append(Case(n))
    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """N を 1..100 から、極端値を厚めに混ぜつつ一様にサンプリング"""
    if rng.random() < 0.4:
        N = rng.choice([1,2,3,4,5,50,51,99,100])
    else:
        N = rng.randint(N_MIN, N_MAX)
    return Case(N)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.N}\n"

def write_case_files(abs_idx: int, width: int, case: Case,
                     in_dir: str, out_dir: str) -> Tuple[str, str]:
    in_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{INPUT_EXT}"
    out_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{OUTPUT_SUFFIX}"
    in_path = os.path.join(in_dir, in_name)
    out_path = os.path.join(out_dir, out_name)

    with open(in_path, "w", encoding="utf-8", newline="\n") as fi:
        fi.write(case_to_input_text(case))

    ans = solve_case(case)
    # 念のため検証：回文 & '=' が 1 個 or 隣接 2 個
    assert ans == ans[::-1]
    eq_cnt = ans.count("=")
    assert eq_cnt in (1, 2) and (eq_cnt == 1 or "==" in ans)

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

    # ゼロ埋め桁数（少なくとも3桁、終了番号を考慮して自動決定）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 出力
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert N_MIN <= cs.N <= N_MAX

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] N={cs.N:3d} -> out='{solve_case(cs)}' | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
