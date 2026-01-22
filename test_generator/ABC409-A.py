from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC409/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC409/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 409            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC409 A） =========
MIN_N: int = 1
MAX_N: int = 100
ALPHABET = ("o", "x")
# ======================================================

@dataclass
class Case:
    N: int
    T: str
    A: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    return "Yes" if any(t == "o" and a == "o" for t, a in zip(case.T, case.A)) else "No"

# ---------- ケース生成ユーティリティ ----------
def make_directed_cases() -> List[Case]:
    """公式サンプル + 境界・代表パターン"""
    cases: List[Case] = []

    # --- 公式サンプル（問題ページ）
    cases.append(Case(4,  "oxoo",       "xoox"))        # Yes
    cases.append(Case(5,  "xxxxx",      "ooooo"))       # No
    cases.append(Case(10, "xoooxoxxxo", "ooxooooxoo"))  # Yes

    # --- N最小/最大、端値
    cases.append(Case(1, "o", "o"))                     # Yes: 最小で一致
    cases.append(Case(1, "o", "x"))                     # No
    cases.append(Case(1, "x", "o"))                     # No
    cases.append(Case(MAX_N, "o"*MAX_N, "o"*MAX_N))     # Yes: 全部一致
    cases.append(Case(MAX_N, "x"*MAX_N, "x"*MAX_N))     # No : 全て x
    cases.append(Case(MAX_N, "o"*MAX_N, "x"*MAX_N))     # No : 互いに排他

    # --- 交互・境界一致の作り
    t = "".join("o" if i % 2 == 0 else "x" for i in range(10))
    a = "".join("x" if i % 2 == 0 else "o" for i in range(10))
    cases.append(Case(10, t, a))                        # No : 完全にずれる
    cases.append(Case(10, t, a[:5] + "o" + a[6:]))      # Yes : 1箇所だけ揃える

    return cases

def rand_ox_str(rng: random.Random, n: int, p_o: float) -> str:
    """長さ n、'o' の確率 p_o のランダム文字列"""
    return "".join("o" if rng.random() < p_o else "x" for _ in range(n))

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: Yes/No 半々、極端な N を混ぜる"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(4)]
    N = rng.choice(n_choices)

    want_yes = rng.random() < 0.5

    if want_yes:
        # ランダムに作ってから、少なくとも1箇所 (o,o) を強制
        T = rand_ox_str(rng, N, rng.uniform(0.2, 0.8))
        A = rand_ox_str(rng, N, rng.uniform(0.2, 0.8))
        pos = rng.randrange(N)
        T = T[:pos] + "o" + T[pos+1:]
        A = A[:pos] + "o" + A[pos+1:]
    else:
        # (o,o) を作らないようペアを選ぶ: (o,x),(x,o),(x,x) のみ
        pairs = [("o", "x"), ("x", "o"), ("x", "x")]
        T_chars, A_chars = [], []
        for _ in range(N):
            t,a = rng.choice(pairs)
            T_chars.append(t); A_chars.append(a)
        T = "".join(T_chars); A = "".join(A_chars)

    return Case(N, T, A)

def case_to_input_text(case: Case) -> str:
    return f"{case.N}\n{case.T}\n{case.A}\n"

def write_case_files(abs_idx: int, width: int, case: Case, in_dir: str, out_dir: str) -> Tuple[str, str]:
    """abs_idx は連番の絶対番号（START_INDEX からの値）"""
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

    directed = make_directed_cases()

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))

    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁。終了番号を考慮して自動決定
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # START_INDEX から連番でファイルを書き出し
    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_N <= case.N <= MAX_N
        assert len(case.T) == case.N and len(case.A) == case.N
        assert set(case.T) <= set(ALPHABET) and set(case.A) <= set(ALPHABET)

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (ans={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
