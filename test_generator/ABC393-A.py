from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 98                     # 生成するテストケース総数
START_INDEX: int = 3                     # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC393/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC393/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"                # ファイル名プリフィクス
INPUT_EXT: str = ".txt"                  # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"          # 出力ファイル拡張子
RNG_SEED: int | None = 393               # 乱数シード (再現性が必要な場合は整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC393 A） =========
WORDS = ("sick", "fine")                 # S1, S2 はどちらか一方
# ======================================================

@dataclass
class Case:
    S1: str
    S2: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：毒の牡蠣の番号（1..4）"""
    s1, s2 = case.S1, case.S2
    if s1 == "sick" and s2 == "sick":
        return "1"
    if s1 == "sick" and s2 == "fine":
        return "2"
    if s1 == "fine" and s2 == "sick":
        return "3"
    # s1 == "fine" and s2 == "fine"
    return "4"

# ---------- ハンドクラフト（サンプル＋境界網羅） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []
    # 公式サンプル
    cs.append(Case("sick", "fine"))  # -> 2
    cs.append(Case("fine", "fine"))  # -> 4
    # 残りの組み合わせ
    cs.append(Case("sick", "sick"))  # -> 1
    cs.append(Case("fine", "sick"))  # -> 3
    # 代表例を複製しておく（入出力の動作確認用）
    for _ in range(3):
        cs.extend([Case("sick","fine"), Case("fine","fine"),
                   Case("sick","sick"), Case("fine","sick")])
    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    # 4通りを均等〜やや偏りで出す
    combos = [("sick","sick"), ("sick","fine"), ("fine","sick"), ("fine","fine")]
    S1, S2 = rng.choice(combos)
    return Case(S1, S2)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.S1} {case.S2}\n"

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

    # ゼロ埋め桁数（少なくとも3桁、終了番号で自動決定）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 連番で出力（制約チェック込み）
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        assert cs.S1 in WORDS and cs.S2 in WORDS  # 公式の取りうる値のみ

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] ({cs.S1},{cs.S2}) -> {solve_case(cs)} | {in_path} , {out_path}")

    print(f"\nDone. Generated", len(cases), "cases.")
    print("Inputs :", os.path.abspath(INPUT_DIR))
    print("Outputs:", os.path.abspath(OUTPUT_DIR))
    print("Start  :", START_INDEX, ", End:", end_index, ", Width:", width)

if __name__ == "__main__":
    main()
