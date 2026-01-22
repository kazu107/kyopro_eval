from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 96                 # 生成するテストケース総数
START_INDEX: int = 5                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC418/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC418/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 418           # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
MAX_N: int = 20                      # 問題制約上の最大 N
# ======================================================

LOWER = string.ascii_lowercase  # 'a'..'z'
SUF = "tea"

@dataclass
class Case:
    S: str  # 入力S（Nはlen(S)として出力に書く）

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：Sが'tea'で終わればYes、そうでなければNo"""
    return "Yes" if case.S.endswith(SUF) else "No"

# ---------- 文字列ユーティリティ ----------
def rand_lower(rng: random.Random, L: int) -> str:
    return "".join(rng.choice(LOWER) for _ in range(L))

def rand_len(rng: random.Random, min_len: int = 1, max_len: int = MAX_N) -> int:
    return rng.randint(min_len, max_len)

def rand_with_suffix(rng: random.Random, L: int, suf: str = SUF) -> str:
    """長さLで末尾がsufのランダム文字列（L>=len(suf)前提）"""
    k = len(suf)
    prefix = rand_lower(rng, L - k) if L > k else ""
    return prefix + suf

def rand_without_suffix(rng: random.Random, L: int, suf: str = SUF) -> str:
    """長さLで末尾がsufではないランダム文字列"""
    s = rand_lower(rng, L)
    if L >= len(suf):
        # 末尾がsufだったら最後の1文字だけ回避変更
        if s.endswith(suf):
            last = s[-1]
            alt = 'a' if last != 'a' else 'b'
            s = s[:-1] + alt
    return s

# ---------- ケース作成（網羅 + ランダム） ----------
def make_directed_cases(rng: random.Random) -> List[Case]:
    """サンプル・境界・代表例をハンドクラフト"""
    cases: List[Case] = []

    # --- サンプル (問題文より)
    cases.append(Case("greentea"))  # Yes
    cases.append(Case("coffee"))    # No
    cases.append(Case("tea"))       # Yes
    cases.append(Case("t"))         # No

    # --- 境界長 (N=1,2,3,20)
    cases += [Case("a"), Case("t"), Case("e"), Case("aa")]    # すべて No
    cases += [Case("te"), Case("ea"), Case("ta")]             # No
    cases += [Case("tea")]                                    # Yes (最小でYes)
    cases += [Case("x"*17 + "tea")]                           # N=20, Yes
    cases += [Case("x"*17 + "teb")]                           # N=20, No

    # --- 末尾が'tea'の多様な形
    for pref in ["", "x", "abc", "zzzz", "a"*5, "qwerty"]:
        s = pref + "tea"
        if 1 <= len(s) <= MAX_N:
            cases.append(Case(s))

    # --- 'tea'を含むが末尾ではない
    for s in ["teax", "teateaX".lower(), "ateabc", "teab", "xteax".lower()]:
        if 1 <= len(s) <= MAX_N:
            cases.append(Case(s))  # いずれも No

    # --- 繰り返し・パターン検証
    cases.append(Case("teatea"))    # Yes（…'tea'で終わる）
    cases.append(Case("teateab"))   # No
    cases.append(Case(("abc"*5) + "tea"))  # 長さ18, Yes
    cases.append(Case("a"*MAX_N))   # No（同一文字列最大）

    # --- Yes/Noの近傍（1文字違い）
    cases += [Case("tfa"), Case("tba"), Case("tzz"), Case("bea"), Case("taa")]  # 全て No

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダム: Yes/Noが適度に混ざるよう制御"""
    want_yes = rng.random() < 0.45
    if want_yes:
        L = rand_len(rng, max(3, 3), MAX_N)  # Yesには長さ>=3を確保
        S = rand_with_suffix(rng, L, SUF)
    else:
        L = rand_len(rng, 1, MAX_N)
        S = rand_without_suffix(rng, L, SUF)
    return Case(S)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    N = len(case.S)
    return f"{N}\n{case.S}\n"

def write_case_files(abs_idx: int, width: int, case: Case, in_dir: str, out_dir: str) -> Tuple[str, str]:
    in_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{INPUT_EXT}"
    out_name = f"{FILE_PREFIX}{abs_idx:0{width}d}{OUTPUT_SUFFIX}"
    in_path = os.path.join(in_dir, in_name)
    out_path = os.path.join(out_dir, out_name)

    with open(in_path, "w", encoding="utf-8", newline="\n") as f_in:
        f_in.write(case_to_input_text(case))

    ans = solve_case(case)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f_out:
        f_out.write(ans + "\n")

    return in_path, out_path

def main() -> None:
    rng = random.Random(RNG_SEED)
    safe_mkdir(INPUT_DIR)
    safe_mkdir(OUTPUT_DIR)

    # ハンドクラフト + ランダム充填
    directed = make_directed_cases(rng)
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁。終了番号から決定
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 連番で書き出し（制約チェック付き）
    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（1 <= N <= 20, 英小文字）
        assert 1 <= len(case.S) <= MAX_N
        assert case.S.islower()

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  ({solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
