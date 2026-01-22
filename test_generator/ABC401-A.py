from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import Tuple, List

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                 # 生成するテストケース総数
START_INDEX: int = 4                 # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC401/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC401/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"            # ファイル名プリフィクス
INPUT_EXT: str = ".txt"              # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"      # 出力ファイル拡張子
RNG_SEED: int | None = 401           # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC401 A） =========
S_MIN: int = 100
S_MAX: int = 999
SUCCESS_LO: int = 200
SUCCESS_HI: int = 299
# ======================================================

@dataclass
class Case:
    S: int

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力 ('Success' / 'Failure')"""
    return "Success" if SUCCESS_LO <= case.S <= SUCCESS_HI else "Failure"

# ---------- ハンドクラフト（サンプル＋境界・代表パターン） ----------
def make_directed_cases() -> List[Case]:
    cs: List[Case] = []
    # 公式サンプル
    cs.append(Case(200))  # Success
    cs.append(Case(401))  # Failure
    cs.append(Case(999))  # Failure

    # 境界＆直近
    cs += [Case(100), Case(199), Case(200), Case(299), Case(300), Case(999)]
    cs += [Case(SUCCESS_LO+1), Case(SUCCESS_HI-1)]  # 201,298

    # 代表例いくつか
    cs += [Case(123), Case(250), Case(777), Case(298), Case(455), Case(289)]
    return cs

# ---------- ランダムケース ----------
def make_random_case(rng: random.Random) -> Case:
    """成功区間付近を厚めに、全域からも拾う"""
    r = rng.random()
    if r < 0.45:
        # 成功区間から一様
        S = rng.randint(SUCCESS_LO, SUCCESS_HI)
    elif r < 0.75:
        # 境界付近（±3）を狙う（ただし範囲内に切り詰め）
        base = rng.choice([SUCCESS_LO, SUCCESS_HI])
        S = base + rng.choice([-3, -2, -1, 0, +1, +2, +3])
        S = min(max(S, S_MIN), S_MAX)
    else:
        # 全域から一様
        S = rng.randint(S_MIN, S_MAX)
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
    with open(out_path, "w", encoding="utf-8", newline="\n") as fo:
        fo.write(solve_case(case) + "\n")

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

    # ゼロ埋め桁数（少なくとも3桁、終了番号を考慮）
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    # 連番でファイルを書き出し
    for offset, cs in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert S_MIN <= cs.S <= S_MAX

        in_path, out_path = write_case_files(abs_idx, width, cs, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] S={cs.S} -> {solve_case(cs)} | {in_path} , {out_path}")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
