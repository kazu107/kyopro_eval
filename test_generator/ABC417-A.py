from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                 # 生成するテストケース総数
START_INDEX: int = 4                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC417/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC417/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 417           # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
MAX_N: int = 20                      # 問題制約上の最大 N（1 <= N <= 20）
# ======================================================

LOWER = string.ascii_lowercase  # 'a'..'z'

@dataclass
class Case:
    N: int
    A: int
    B: int
    S: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：先頭からA、末尾からBを削った部分文字列"""
    N, A, B, S = case.N, case.A, case.B, case.S
    # 仕様どおりのスライス
    return S[A: N - B]

# ---------- 文字列ユーティリティ ----------
def rand_lower(rng: random.Random, L: int) -> str:
    return "".join(rng.choice(LOWER) for _ in range(L))

# ---------- 指向（網羅）ケース ----------
def make_directed_cases(rng: random.Random) -> List[Case]:
    """サンプル・境界・代表パターンを網羅するハンドクラフト"""
    C: List[Case] = []

    # --- 問題文のサンプル（出力は solve_case が生成）
    C.append(Case(7, 1, 3, "atcoder"))                 # -> "tco"
    C.append(Case(1, 0, 0, "a"))                       # -> "a"
    C.append(Case(20, 4, 8, "abcdefghijklmnopqrst"))   # -> "efghijkl"

    # --- 境界（N=1,2,MAX_N）/ A=0, B=0 / 片側ギリギリ
    C.append(Case(1, 0, 0, "z"))                       # 長さ1そのまま
    C.append(Case(2, 0, 1, "xy"))                      # 先頭だけ残る
    C.append(Case(2, 1, 0, "xy"))                      # 末尾だけ残る
    C.append(Case(MAX_N, MAX_N-1, 0, rand_lower(rng, MAX_N)))  # 最後の1文字
    C.append(Case(MAX_N, 0, MAX_N-1, rand_lower(rng, MAX_N)))  # 最初の1文字
    C.append(Case(MAX_N, 1, 1, "a"*MAX_N))            # 同一文字列

    # --- パターン検証（繰り返し・ランダム・辞書順っぽい）
    C.append(Case(10, 3, 4, "abbaabbaab"))            # 中央が残る
    C.append(Case(15, 5, 5, "qwertyuiopasdfg"))       # A=B
    C.append(Case(20, 9, 5, "aaaaaaaaaabbbbbbbbbb"))  # 2色ブロック
    C.append(Case(20, 0, 10, "abcdefghijklmnopqrst")) # 後半削除
    C.append(Case(20, 10, 0, "abcdefghijklmnopqrst")) # 前半削除
    C.append(Case(19, 7, 11-0, rand_lower(rng, 19)))  # N-A-B=1 になるよう調整
    # N-A-B = 様々な値（1..N-1）
    for k in [1, 2, 3, 5, 9]:
        N = max(4, k + 1 + 1)  # A,B>=0かつN-A-B=k
        if N < 10: N = 10
        if N > MAX_N: N = MAX_N
        A = min(3, N - k)      # Aを小さめに
        B = N - A - k
        S = rand_lower(rng, N)
        C.append(Case(N, A, B, S))

    return C

# ---------- ランダムケース（制約を必ず満たす生成） ----------
def make_random_case(rng: random.Random) -> Case:
    """
    制約: 1<=N<=MAX_N, 0<=A, 0<=B, A+B<N, Sは小文字で|S|=N
    生成方法: Nと“残す長さk”を先に決め、Aを0..N-kから選び、B=N-A-kで充足。
    """
    N = rng.randint(1, MAX_N)
    k = rng.randint(1, N)            # 残す長さ (N-A-B)
    A = rng.randint(0, N - k)
    B = N - A - k                    # 0以上が保証される
    S = rand_lower(rng, N)
    return Case(N, A, B, S)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.N} {case.A} {case.B}\n{case.S}\n"

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

    # 指向ケース + ランダム補充
    directed = make_directed_cases(rng)
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数（少なくとも3桁）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 書き出し（制約チェック付き）
    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック
        assert 1 <= case.N <= MAX_N
        assert 0 <= case.A and 0 <= case.B and case.A + case.B < case.N
        assert case.S.islower() and len(case.S) == case.N

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (N-A-B={case.N - case.A - case.B})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
