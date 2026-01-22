from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 96                   # 生成するテストケース総数
START_INDEX: int = 5                   # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC399/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC399/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"              # ファイル名プリフィクス
INPUT_EXT: str = ".txt"                # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"        # 出力ファイル拡張子
RNG_SEED: int | None = 399             # 乱数シード (再現性が必要なら整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC399 A） =========
N_MIN: int = 1
N_MAX: int = 100
LOWER: str = string.ascii_lowercase    # 'a'..'z'
# ======================================================

@dataclass
class Case:
    N: int
    S: str
    T: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：S と T のハミング距離"""
    dist = sum(1 for a, b in zip(case.S, case.T) if a != b)
    return str(dist)

# ---------- ユーティリティ ----------
def rand_lower_str(rng: random.Random, L: int) -> str:
    return "".join(rng.choice(LOWER) for _ in range(L))

def mutate_char(rng: random.Random, ch: str) -> str:
    """ch != 返り値 となる別の英小文字を返す"""
    while True:
        c = rng.choice(LOWER)
        if c != ch:
            return c

def make_case_with_k_mismatch(rng: random.Random, N: int, K: int) -> Case:
    """長さ N、ハミング距離ちょうど K の (S,T) を作る"""
    assert 0 <= K <= N
    S = list(rand_lower_str(rng, N))
    T = S[:]  # まず同じにしてから K か所だけ変える
    # 位置を K 個選んで T を別文字に
    idxs = rng.sample(range(N), K)
    for i in idxs:
        T[i] = mutate_char(rng, S[i])
    return Case(N, "".join(S), "".join(T))

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases(rng: random.Random) -> List[Case]:
    cs: List[Case] = []

    # --- 公式サンプル（問題ページ）
    cs.append(Case(6,  "abcarc",    "agcahc"))     # -> 2
    cs.append(Case(7,  "atcoder",   "contest"))    # -> 7
    cs.append(Case(8,  "chokudai",  "chokudai"))   # -> 0
    cs.append(Case(10, "vexknuampx","vzxikuamlx")) # -> 4

    # --- N 最小/最大、K=0/N/境界
    cs.append(make_case_with_k_mismatch(rng, 1, 0))          # N=1, 等しい
    cs.append(make_case_with_k_mismatch(rng, 1, 1))          # N=1, 全不一致
    cs.append(make_case_with_k_mismatch(rng, N_MAX, 0))      # N=100, 全一致
    cs.append(make_case_with_k_mismatch(rng, N_MAX, N_MAX))  # N=100, 全不一致
    cs.append(make_case_with_k_mismatch(rng, 50, 1))         # 1 か所のみ不一致
    cs.append(make_case_with_k_mismatch(rng, 50, 49))        # ほぼ全不一致

    # --- パターン：同一文字列/一文字種のみ/交互
    s = "a"*30 + "b"*20
    t = s
    cs.append(Case(len(s), s, t))                             # 0
    t2 = list(t); t2[0] = "z"; t2[-1] = "z"
    cs.append(Case(len(s), s, "".join(t2)))                   # 2
    s_alt = "".join("ab"[i % 2] for i in range(40))
    t_alt = "".join("ba"[i % 2] for i in range(40))           # すべて不一致
    cs.append(Case(40, s_alt, t_alt))

    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """K（不一致数）を 0..N から広く取り、極端な N も混ぜる"""
    N = rng.choice([N_MIN, N_MAX] + [rng.randint(2, N_MAX - 1) for _ in range(4)])
    # 不一致数 K を幅広く（0,1,N-1,N を厚めに）
    bucket = [0, 1, N//2, max(0, N-1), N]
    if rng.random() < 0.4:
        K = rng.choice(bucket)
        K = min(max(K, 0), N)
    else:
        K = rng.randint(0, N)
    return make_case_with_k_mismatch(rng, N, K)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return f"{case.N}\n{case.S}\n{case.T}\n"

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

    directed = make_directed_cases(rng)

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数（少なくとも3桁、終了番号で自動決定）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 連番でファイル出力
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert N_MIN <= cs.N <= N_MAX
        assert len(cs.S) == cs.N and len(cs.T) == cs.N
        assert cs.S.islower() and cs.S.isalpha() and cs.S.isascii()
        assert cs.T.islower() and cs.T.isalpha() and cs.T.isascii()

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] N={cs.N}, dist={solve_case(cs)} -> {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
