from __future__ import annotations
import os
import random
import string
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC404/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC404/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル保存先拡張子
RNG_SEED: int | None = 404            # 乱数シード (再現性が必要なら整数、ランダムにするなら None)
# ========= 問題固有の既定値（ABC404 A） =========
MIN_LEN: int = 1
MAX_LEN: int = 25
LOWER: str = string.ascii_lowercase   # 'a'..'z'
# ======================================================

@dataclass
class Case:
    S: str

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力: S に含まれない最小(辞書順)の英小文字1文字を出力"""
    used = set(case.S)
    for ch in LOWER:
        if ch not in used:
            return ch
    # 制約上 S の長さは 25 以下なので必ず見つかるはず
    raise AssertionError("No missing letter (should not happen under constraints)")

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases(rng: random.Random) -> List[Case]:
    cases: List[Case] = []

    # --- 公式サンプル（出力は任意可だが、生成器では最小の欠字を出す）
    cases.append(Case("a"))                                       # 欠字: b
    cases.append(Case("abcdfhijklmnopqrstuvwxyz"))                # 欠字: e (g でもよい)
    cases.append(Case("qazplwsxokmedcijnrfvuhbgt"))               # 欠字: y

    # --- 端寄せ・境界
    cases.append(Case("a"*MAX_LEN))                               # 1文字のみ反復 -> b
    cases.append(Case("z"*MAX_LEN))                               # -> a
    cases.append(Case(LOWER.replace("a", "")))                    # 長さ25, 欠字=a
    cases.append(Case(LOWER.replace("m", "")))                    # 欠字=m
    cases.append(Case(LOWER.replace("z", "")))                    # 欠字=z

    # --- ランダムでは出にくいパターン
    cases.append(Case("".join("abcde"[i%5] for i in range(25))))  # 少種類大量重複
    # 交互系
    alt = ("ab"*13)[:25]
    cases.append(Case(alt))                                       # 欠字: c

    # --- 長さ最小/そこそこ/上限
    cases.append(Case("b"))                                       # 欠字: a
    mid = "".join(sorted(set("helloworld")))                      # 'dehlorw' → 欠字: a
    cases.append(Case(mid))
    almost = list(LOWER)
    missing = rng.choice(almost); almost.remove(missing)
    rng.shuffle(almost)
    cases.append(Case("".join(almost)))                           # ランダムな1字欠損, 長さ25

    return cases

# ---------- ランダムケース ----------
def rand_s_with_random_missing(rng: random.Random) -> str:
    """欠字を1つ選び、残り25文字からランダムに長さ L の文字列を作る（重複可）"""
    missing = rng.choice(LOWER)
    allowed = [c for c in LOWER if c != missing]
    L = rng.randint(MIN_LEN, MAX_LEN)
    # ランダムに allowed から重複ありで選ぶ（昇順や重複偏重も混ぜる）
    mode = rng.random()
    if mode < 0.2:
        # allowed のうち少数の文字だけをよく使う
        base = rng.sample(allowed, k=rng.randint(1, 4))
        return "".join(rng.choice(base) for _ in range(L))
    elif mode < 0.4:
        # なるべく多種類を使う
        pool = allowed[:]
        rng.shuffle(pool)
        s = []
        while len(s) < L:
            s.extend(pool)
        return "".join(s[:L])
    else:
        return "".join(rng.choice(allowed) for _ in range(L))

def make_random_case(rng: random.Random) -> Case:
    return Case(rand_s_with_random_missing(rng))

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

    directed = make_directed_cases(rng)

    # 必要数までランダムで補充
    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_LEN <= len(cs.S) <= MAX_LEN
        assert cs.S.islower() and cs.S.isalpha()

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] S(len={len(cs.S)}) -> miss='{solve_case(cs)}' | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
