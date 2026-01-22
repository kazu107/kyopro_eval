from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 98                 # 生成するテストケース総数
START_INDEX: int = 3                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC419/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC419/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 419           # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
MAX_LEN: int = 10                    # 問題制約上の最大 |S|
# ======================================================

LOWER = string.ascii_lowercase  # 'a'..'z'
KNOWN = {"red": "SSS", "blue": "FFF", "green": "MMM"}

@dataclass
class Case:
    S: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力：KNOWN にあれば対応語、なければ 'Unknown'"""
    return KNOWN.get(case.S, "Unknown")

# ---------- 文字列生成ユーティリティ ----------
def rand_word(rng: random.Random, min_len: int = 1, max_len: int = MAX_LEN) -> str:
    L = rng.randint(min_len, max_len)
    return "".join(rng.choice(LOWER) for _ in range(L))

def near_misses(base: str) -> List[str]:
    """編集距離1あたりの代表例（置換/挿入/削除）+ 順序入替など"""
    res: set[str] = set()

    # 置換
    for i, _ in enumerate(base):
        for c in "abz":  # 代表的な少文字で抑える
            if c != base[i]:
                res.add(base[:i] + c + base[i+1:])

    # 挿入
    for i in range(len(base)+1):
        for c in "az":
            res.add(base[:i] + c + base[i:])

    # 削除
    for i in range(len(base)):
        res.add(base[:i] + base[i+1:])

    # 隣接swap
    for i in range(len(base)-1):
        s = list(base)
        s[i], s[i+1] = s[i+1], s[i]
        res.add("".join(s))

    # 制約オーバーは除外（長さ1..MAX_LEN & 英小文字のみ）
    res = {s for s in res if 1 <= len(s) <= MAX_LEN and s.islower()}
    # 正解語そのものは near miss から除外
    res.discard(base)
    return sorted(res)

# ---------- ケース作成 ----------
def make_directed_cases(rng: random.Random) -> List[Case]:
    """網羅用のハンドクラフト（サンプル・境界・周辺例）"""
    cases: List[Case] = []

    # 知っている3語（ぴったり一致）
    for w in ["red", "blue", "green"]:
        cases.append(Case(w))

    # サンプル相当の Unknown
    cases.append(Case("atcoder"))

    # 近いが違う（編集距離1 代表）
    for w in ["red", "blue", "green"]:
        nm = near_misses(w)
        # いくつか代表抽出（長さやパターンがばらけるように）
        pick = [nm[0], nm[len(nm)//2], nm[-1]] if len(nm) >= 3 else nm
        for s in pick:
            cases.append(Case(s))

    # 境界長
    cases.append(Case("a"))                  # 長さ1
    cases.append(Case("z"*MAX_LEN))          # 長さMAX
    # 「既知語の前後に1文字」(制約内)
    cases.append(Case("xred"))               # 長さ4
    cases.append(Case("bluey"))              # 長さ5
    # 同長・違語（衝突テスト）
    cases.extend([Case("reed"), Case("bleu"), Case("grean")])

    # 既知語の大文字化は制約外だが一応 Unknown 相当の下限チェック（英小のみで統一）
    # -> 制約に合わせて生成しない

    # 既知語を複数投入（安定性）
    for _ in range(3):
        cases.append(Case("red"))
    for _ in range(2):
        cases.append(Case("blue"))
    cases.append(Case("green"))

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダム: 既知語/未知語をバランスよく"""
    mode = rng.random()
    if mode < 0.35:
        # 既知語をそのまま
        S = rng.choice(list(KNOWN.keys()))
    elif mode < 0.55:
        # 既知語の near miss（Unknown）
        base = rng.choice(list(KNOWN.keys()))
        cand = rng.choice(near_misses(base))
        S = cand
    else:
        # 完全ランダム（Unknown が多い）
        S = rand_word(rng)
        # たまたま既知語になったらそのままでもOK
    return Case(S)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    return case.S + "\n"

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

    directed = make_directed_cases(rng)

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁。終了番号で決定
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # START_INDEX から連番でファイルを書き出し
    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック
        assert 1 <= len(case.S) <= MAX_LEN
        assert case.S.islower()

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  ({solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
