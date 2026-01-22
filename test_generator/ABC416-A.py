from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 98                 # 生成するテストケース総数
START_INDEX: int = 3                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC416/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC416/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 416           # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)

# ---- 問題特有の設定（デフォルトは公式と同じ） ----
MAX_N: int = 100                     # 1 <= N <= 100
# ======================================================

OX = ("o", "x")

@dataclass
class Case:
    N: int
    L: int
    R: int
    S: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力を計算（L..R がすべて 'o' なら Yes）"""
    sub = case.S[case.L - 1: case.R]
    return "Yes" if all(c == "o" for c in sub) else "No"

def rand_ox_string(rng: random.Random, n: int, p_o: float) -> str:
    """長さ n の 'o'/'x' 文字列を、'o' になる確率 p_o で生成"""
    return "".join("o" if rng.random() < p_o else "x" for _ in range(n))

def make_directed_cases(rng: random.Random) -> List[Case]:
    """網羅用のハンドクラフトケース（サンプル含む＋境界・代表）"""
    cases: List[Case] = []

    # ---- 単要素 N=1：Yes/No
    cases.append(Case(1, 1, 1, "o"))                     # Yes
    cases.append(Case(1, 1, 1, "x"))                     # No

    # ---- 公式サンプル
    cases.append(Case(10, 6, 8, "xoxxooooxo"))           # Yes（Sample 1）
    cases.append(Case(9, 6, 8, "xoxxoxoox"))             # No  （Sample 2）

    # ---- 全範囲 L=1..N
    cases.append(Case(MAX_N, 1, MAX_N, "o" * MAX_N))     # Yes：全 o
    cases.append(Case(MAX_N, 1, MAX_N, "x" * MAX_N))     # No：全 x

    # ---- 交互パターン / 代表
    alt = ("ox" * ((MAX_N + 1) // 2))[:MAX_N]
    cases.append(Case(MAX_N, 2, MAX_N - 1, alt))         # No：区間内に必ず x を含む

    # ---- 端が境界：L=1, R=1 / L=N, R=N
    mid = "ooxxx"
    cases.append(Case(5, 1, 1, mid))                     # Yes（先頭 'o'）
    cases.append(Case(5, 3, 3, mid))                     # No （中央 'x'）
    tail = "oxoo" + "o" * (MAX_N - 5) + "o"
    cases.append(Case(MAX_N, MAX_N, MAX_N, tail[:-1] + "o"))  # Yes（末尾 'o'）
    cases.append(Case(MAX_N, MAX_N, MAX_N, tail[:-1] + "x"))  # No （末尾 'x'）

    # ---- 区間内に 'x' が先頭/末尾/内部にある
    s1 = "xxxoooooxx"
    cases.append(Case(len(s1), 4, 8, s1))                # Yes：区間は o の塊
    s2 = "xoooooxxxx"
    cases.append(Case(len(s2), 1, 5, s2))                # No：区間先頭が x を含む（先頭は x ）
    s3 = "oooooxoooo"
    cases.append(Case(len(s3), 1, 10, s3))               # No：区間末尾付近に x

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース：N/L/R と S をばらけさせ、Yes/No を半々程度で作る"""
    n_choices = [1, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)]
    N = rng.choice(n_choices)
    L = rng.randint(1, N)
    R = rng.randint(L, N)

    # 'o' 割合は極端～中庸を混ぜる
    p_o = rng.choice([0.0, 0.2, 0.5, 0.8, 1.0])
    S = list(rand_ox_string(rng, N, p_o))

    yes_mode = rng.random() < 0.5

    if yes_mode:
        # 区間 L..R をすべて 'o' にして「Yes」を保証
        for i in range(L - 1, R):
            S[i] = "o"
    else:
        # 区間に少なくとも 1 つ 'x' を入れて「No」を保証
        if all(c == "o" for c in S[L - 1:R]):
            pos = rng.randint(L - 1, R - 1)
            S[pos] = "x"

    S_str = "".join(S)
    return Case(N, L, R, S_str)

def case_to_input_text(case: Case) -> str:
    return f"{case.N} {case.L} {case.R}\n{case.S}\n"

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

    directed = make_directed_cases(rng)

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

        # 制約チェック（公式準拠）
        assert 1 <= case.N <= MAX_N
        assert 1 <= case.L <= case.R <= case.N
        assert len(case.S) == case.N
        assert all(c in OX for c in case.S)

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  ({solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
