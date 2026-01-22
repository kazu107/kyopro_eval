"""
AtCoder ABC421 A - Misdelivery 用テストケース自動生成スクリプト
- 入力:  N, S_1..S_N, X Y   (英小文字, 1<=len<=10)
- 出力:  X号室の住人 S_X が Y なら "Yes", それ以外は "No"
- 生成ファイル:
    inputs/case004.txt,  outputs/case004.out.txt
    inputs/case005.txt,  outputs/case005.out.txt
    ... のように連番で作成
"""

from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 96                 # 生成するテストケース総数
START_INDEX: int = 5                # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC421/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC421/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"           # ファイル名プリフィクス
INPUT_EXT: str = ".txt"             # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"     # 出力ファイル拡張子
RNG_SEED: int | None = 421          # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
MAX_N: int = 100                    # 問題制約上の最大 N
# ======================================================

LOWER = string.ascii_lowercase  # 'a'..'z'

@dataclass
class Case:
    N: int
    S: List[str]
    X: int
    Y: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力を計算 ('Yes' / 'No')"""
    return "Yes" if case.S[case.X - 1] == case.Y else "No"

def rand_name(rng: random.Random, min_len: int = 1, max_len: int = 10) -> str:
    """英小文字のみのランダム名。極端長(1,10)も出やすくする。"""
    lens = [1, 10, rng.randint(min_len, max_len), rng.randint(min_len, max_len)]
    L = rng.choice(lens)
    return "".join(rng.choice(LOWER) for _ in range(L))

def rand_names_list(rng: random.Random, n: int, allow_dupes_prob: float = 0.6) -> List[str]:
    """長さ1..10, 英小文字。重複を入れることが多め(allow_dupes_prob)。"""
    names: List[str] = []
    for _ in range(n):
        if names and rng.random() < allow_dupes_prob:
            names.append(rng.choice(names))
        else:
            names.append(rand_name(rng))
    return names

def name_not_in(rng: random.Random, ban: List[str]) -> str:
    """S に含まれない名前を生成"""
    tries = 0
    while True:
        tries += 1
        cand = rand_name(rng)
        if cand not in ban:
            return cand
        if tries > 1000:
            return cand + "a" if len(cand) < 10 else "b"*10

def make_directed_cases(rng: random.Random) -> List[Case]:
    """網羅用のハンドクラフトケース（サンプル相当＋境界・代表パターン）"""
    cases: List[Case] = []

    # --- 単要素 N=1：一致/不一致
    cases.append(Case(1, ["a"], 1, "a"))           # Yes
    cases.append(Case(1, ["a"], 1, "b"))           # No

    # --- 代表例
    cases.append(Case(3, ["sato", "suzuki", "takahashi"], 3, "takahashi"))  # Yes
    cases.append(Case(3, ["sato", "suzuki", "takahashi"], 1, "aoki"))       # No
    cases.append(Case(2, ["smith", "smith"], 1, "smith"))                   # Yes (重名)
    cases.append(Case(2, ["wang", "li"], 2, "wang"))                        # No (位置違い)

    # --- N=max
    S_all_same = ["aaaaa"] * MAX_N
    cases.append(Case(MAX_N, S_all_same, 67, "aaaaa"))                      # Yes

    S_all_same_z = ["z" * 10] * MAX_N
    cases.append(Case(MAX_N, S_all_same_z, MAX_N, "z" * 9))                 # No（長さ9）

    S_rand = rand_names_list(rng, MAX_N, allow_dupes_prob=0.3)
    cases.append(Case(MAX_N, S_rand, 1, S_rand[0]))                         # Yes (X=1)
    cases.append(Case(MAX_N, S_rand, MAX_N, name_not_in(rng, S_rand)))      # No (未登場名)

    # --- 存在はするが部屋違い
    S_mix = ["a", "bb", "ccc", "dddd", "eeeee", "ffffff", "gg", "h", "iiiiiiiii", "j"]
    N = len(S_mix); X = 4; Y = S_mix[0]
    cases.append(Case(N, S_mix, X, Y))                                      # No

    # --- 極端長
    S_len_extreme = ["a", "z"*10, "b", "c"]
    cases.append(Case(4, S_len_extreme, 2, "z"*10))                         # Yes（長さ=10）
    cases.append(Case(4, S_len_extreme, 3, "z"*10))                         # No（部屋違い）

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: 極端な N を混ぜつつ Yes/No をバランスよく作る"""
    n_choices = [1, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)]
    N = rng.choice(n_choices)
    S = rand_names_list(rng, N, allow_dupes_prob=rng.uniform(0.3, 0.8))
    X = rng.randint(1, N)

    mode = rng.random()
    if mode < 0.42:
        Y = S[X - 1]  # Yes
    elif mode < 0.70 and N >= 2 and len(set(S)) >= 1:
        candidates = list(range(1, N + 1))
        candidates.remove(X)
        k = rng.choice(candidates)
        Y = S[k - 1]  # No: S 内にあるが部屋違い
        if Y == S[X - 1]:
            Y = name_not_in(rng, S)
    else:
        Y = name_not_in(rng, S)  # No: 未登場名

    return Case(N, S, X, Y)

def case_to_input_text(case: Case) -> str:
    lines = [str(case.N)]
    lines.extend(case.S)
    lines.append(f"{case.X} {case.Y}")
    return "\n".join(lines) + "\n"

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

        # 制約チェック
        assert 1 <= case.N <= MAX_N
        assert 1 <= case.X <= case.N
        assert all(1 <= len(s) <= 10 and s.islower() for s in case.S)
        assert 1 <= len(case.Y) <= 10 and case.Y.islower()

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  ({solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
