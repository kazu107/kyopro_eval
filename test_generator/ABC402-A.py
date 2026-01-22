from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC402/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC402/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル保存先拡張子
RNG_SEED: int | None = 402            # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC402 A） =========
MIN_LEN: int = 1
MAX_LEN: int = 100
LOWER: str = string.ascii_lowercase   # 'a'..'z'
UPPER: str = string.ascii_uppercase   # 'A'..'Z'
# ======================================================

@dataclass
class Case:
    S: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：S の英大文字のみを順に連結（空でも可）"""
    return "".join(ch for ch in case.S if ch.isupper())

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル（問題ページより）
    cs.append(Case("AtCoderBeginnerContest"))  # -> ACBC
    cs.append(Case("PaymentRequired"))         # -> PR
    cs.append(Case("program"))                 # -> "" (空行)

    # --- 境界長
    cs.append(Case("a"))                       # 最小長、全小文字 -> ""
    cs.append(Case("Z"))                       # 最小長、全大文字 -> "Z"
    cs.append(Case("a"*MAX_LEN))               # 最大長、全小文字 -> ""
    cs.append(Case("Z"*MAX_LEN))               # 最大長、全大文字 -> "Z"*MAX_LEN

    # --- パターン色々
    cs.append(Case("abcDEFghi"))               # -> "DEF"
    cs.append(Case("ABCdefGHIjkl"))            # -> "ABCGHI"
    cs.append(Case("aAaAaAaA"))                # 交互 -> "AAAA"
    cs.append(Case("bbbbbbbbbbC"))             # 末尾のみ大文字 -> "C"
    cs.append(Case("Cbbbbbbbbbb"))             # 先頭のみ大文字 -> "C"

    # --- ランダムでは出にくい：大文字が 1 文字だけ
    cs.append(Case("x"*50 + "Q" + "y"*49))     # 長さ100で中央のみ大文字 -> "Q"

    return cs

# ---------- ランダムケース ----------
def rand_s_with_upper_ratio(rng: random.Random, length: int, p_upper: float) -> str:
    """長さ length。各文字を確率 p_upper で大文字、それ以外は小文字から一様に選ぶ"""
    s_chars = []
    for _ in range(length):
        if rng.random() < p_upper:
            s_chars.append(rng.choice(UPPER))
        else:
            s_chars.append(rng.choice(LOWER))
    return "".join(s_chars)

def make_random_case(rng: random.Random) -> Case:
    """Yes/No の概念は無いが、'大文字ゼロ' と '多め' をバランスよく混ぜる"""
    L = rng.choice([MIN_LEN, MAX_LEN] + [rng.randint(2, MAX_LEN - 1) for _ in range(4)])
    mode = rng.random()
    if mode < 0.2:
        # 全小文字
        S = "".join(rng.choice(LOWER) for _ in range(L))
    elif mode < 0.4:
        # 全大文字
        S = "".join(rng.choice(UPPER) for _ in range(L))
    elif mode < 0.7:
        # 交互やブロック
        block = rng.randint(1, max(1, L // 5))
        parts: List[str] = []
        use_upper = rng.random() < 0.5
        i = 0
        while i < L:
            k = min(block + rng.randint(0, block), L - i)
            if use_upper:
                parts.append("".join(rng.choice(UPPER) for _ in range(k)))
            else:
                parts.append("".join(rng.choice(LOWER) for _ in range(k)))
            use_upper = not use_upper
            i += k
        S = "".join(parts)
    else:
        # 一様に大文字割合 p を設定
        p = rng.uniform(0.05, 0.95)
        S = rand_s_with_upper_ratio(rng, L, p)
    return Case(S)

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
    ans = solve_case(case)  # 空文字の可能性あり
    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(ans + "\n")  # 空でも改行を出力
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

    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）: S は英字のみ、1<=|S|<=100
        assert isinstance(cs.S, str)
        assert MIN_LEN <= len(cs.S) <= MAX_LEN
        assert cs.S.isalpha() and cs.S.isascii()

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        # solve_case(cs) は長さ0もあり得る（空行出力）
        print(f"[{abs_idx:0{width}d}] |S|={len(cs.S)}, upper_out_len={len(solve_case(cs))} -> {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
