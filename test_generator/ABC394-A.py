from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                    # 生成するテストケース総数
START_INDEX: int = 4                    # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC394/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC394/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"               # ファイル名プリフィクス
INPUT_EXT: str = ".txt"                 # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"         # 出力ファイル拡張子
RNG_SEED: int | None = 394              # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC394 A） =========
MIN_LEN: int = 1
MAX_LEN: int = 100
DIGITS: str = "0123456789"
# ======================================================

@dataclass
class Case:
    S: str  # 数字のみ、かつ少なくとも '2' を1つ含む

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力: S から '2' 以外を削除した文字列"""
    return "".join(ch for ch in case.S if ch == "2")

# ---------- ハンドクラフト（サンプル＋境界・代表） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # 公式サンプル
    cs.append(Case("20250222"))             # -> 22222
    cs.append(Case("2"))                    # -> 2
    cs.append(Case("22222000111222222"))    # -> 22222222222

    # 最小/最大・パターン
    cs.append(Case("2"*MIN_LEN))            # 長さ1・全て2
    cs.append(Case("2"*MAX_LEN))            # 長さ100・全て2
    cs.append(Case("2" + "0"*(MAX_LEN-1)))  # 先頭のみ2
    cs.append(Case("0"*(MAX_LEN-1) + "2"))  # 末尾のみ2
    cs.append(Case("0"*30 + "2" + "0"*69))  # 中央のみ2
    cs.append(Case("120120120120"))         # 2が定期的に現れる
    cs.append(Case("90909092"))             # 末尾だけ2

    # 2が多い/少ない
    cs.append(Case("2121212121"))           # 交互に2
    cs.append(Case("0000000002"))           # ほぼ0

    return cs

# ---------- ランダムケース ----------
def rand_digits_with_at_least_one_2(rng: random.Random, L: int) -> str:
    """長さ L の数字列を生成。必ず '2' を少なくとも1つ含む"""
    # まず一様に作る（'2' を少し厚めに）
    probs = [0.10 if d == "2" else 0.10 for d in DIGITS]  # 一様気味
    # たまに '2' を増やしたり減らしたり
    if rng.random() < 0.4:
        probs = [0.05 if d != "2" else 0.55 for d in DIGITS]
    elif rng.random() < 0.2:
        probs = [0.11 if d != "2" else 0.01 for d in DIGITS]  # 2が少なめ（後で挿入で補正）
    # 累積でサンプル
    import itertools
    cdf = list(itertools.accumulate(probs))
    total = cdf[-1]
    def sample_digit() -> str:
        x = rng.random() * total
        for d, t in zip(DIGITS, cdf):
            if x <= t:
                return d
        return DIGITS[-1]

    s_list = [sample_digit() for _ in range(L)]
    if "2" not in s_list:
        # 必ず1箇所を '2' に差し替える
        pos = rng.randrange(L)
        s_list[pos] = "2"
    return "".join(s_list)

def make_random_case(rng: random.Random) -> Case:
    L = rng.choice([MIN_LEN, MAX_LEN] + [rng.randint(2, MAX_LEN-1) for _ in range(3)])
    return Case(rand_digits_with_at_least_one_2(rng, L))

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.S}\n"

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
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数（少なくとも3桁）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 出力（制約チェック込み）
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_LEN <= len(cs.S) <= MAX_LEN
        assert cs.S.isdigit()
        assert "2" in cs.S

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] |S|={len(cs.S)}, out='{solve_case(cs)}' -> {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
