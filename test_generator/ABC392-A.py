from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 98                  # 生成するテストケース総数
START_INDEX: int = 3                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC392/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC392/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 392            # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC392 A） =========
VAL_MIN: int = 1
VAL_MAX: int = 100
# ======================================================

@dataclass
class Case:
    A: Tuple[int, int, int]  # (A1, A2, A3)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No'): 3要素のどの並べ替えでも判定OKなように総当り"""
    a1, a2, a3 = case.A
    a = [a1, a2, a3]
    # どの2つの積が残り1つに等しいか（順不同ですべて確認）
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        if a[j] * a[k] == a[i]:
            return "Yes"
    return "No"

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []
    # 公式サンプル
    cs += [Case((3, 15, 5)),   # Yes: 3*5=15
           Case((5, 3, 2)),    # No
           Case((3, 3, 9))]    # Yes

    # 端値/境界
    cs += [Case((1, 1, 1)),          # Yes: 1*1=1
           Case((1, 2, 3)),          # No
           Case((1, 100, 100)),      # Yes: 1*100=100
           Case((10, 10, 100)),      # Yes: 10*10=100
           Case((4, 25, 100)),       # Yes
           Case((4, 25, 99)),        # No
           Case((2, 2, 3)),          # No
           Case((7, 7, 13)),         # No
           Case((2, 4, 8)),          # Yes
           Case((9, 10, 90))]        # Yes

    # 重複/並べ替え耐性の確認
    cs += [Case((42, 1, 42)),        # Yes: 1*42=42
           Case((3, 3, 3))]          # No: 3*3=9

    return cs

# ---------- ランダムケース ----------
def make_yes_case(rng: random.Random) -> Case:
    """必ず Yes になる三つ組を構成（積が 100 以下になる因数を選ぶ）"""
    # 方法1: a*b<=100 を満たす a,b を選んで c=a*b
    while True:
        a = rng.randint(VAL_MIN, VAL_MAX)
        b = rng.randint(VAL_MIN, VAL_MAX)
        c = a * b
        if VAL_MIN <= c <= VAL_MAX:
            triple = [a, b, c]
            rng.shuffle(triple)
            return Case(tuple(triple))  # type: ignore

def make_no_case(rng: random.Random) -> Case:
    """どの2つの積も残り1つに一致しない三つ組をサンプリング"""
    while True:
        triple = [rng.randint(VAL_MIN, VAL_MAX) for _ in range(3)]
        a1, a2, a3 = triple
        ok = False
        for i in range(3):
            j, k = (i + 1) % 3, (i + 2) % 3
            if triple[j] * triple[k] == triple[i]:
                ok = True
                break
        if not ok:
            return Case((a1, a2, a3))

def make_random_case(rng: random.Random) -> Case:
    return make_yes_case(rng) if rng.random() < 0.5 else make_no_case(rng)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    a1, a2, a3 = case.A
    return f"{a1} {a2} {a3}\n"

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

    # ゼロ埋め桁数（少なくとも3桁）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    for offset, c in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert all(VAL_MIN <= x <= VAL_MAX for x in c.A)

        in_path, out_path = write_case_files(abs_idx, width, c, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] A={c.A} -> {solve_case(c)} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print("Inputs :", os.path.abspath(INPUT_DIR))
    print("Outputs:", os.path.abspath(OUTPUT_DIR))
    print("Start  :", START_INDEX, "End:", end_index, "Width:", width)

if __name__ == "__main__":
    main()
