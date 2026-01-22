from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC413/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC413/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 413            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC413 A） =========
MIN_N: int = 1
MAX_N: int = 100
VAL_MIN: int = 1
VAL_MAX: int = 100
M_MIN: int = 1
M_MAX: int = 10000
# ======================================================

@dataclass
class Case:
    N: int
    M: int
    A: List[int]

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    return "Yes" if sum(case.A) <= case.M else "No"

# ---------- ケース生成ユーティリティ ----------
def make_directed_cases() -> List[Case]:
    """網羅用のハンドクラフトケース（サンプル＋境界・代表パターン）"""
    cases: List[Case] = []

    # --- 公式サンプル
    cases.append(Case(5, 15, [3, 1, 4, 1, 5]))   # Yes
    cases.append(Case(5, 5,  [3, 1, 4, 1, 5]))   # No
    cases.append(Case(1, 10000, [100]))          # Yes

    # --- N, M, A の端値
    cases.append(Case(1, 1, [1]))                # 最小すべて -> Yes
    cases.append(Case(1, 1, [100]))              # 和> M -> No
    cases.append(Case(MAX_N, M_MAX, [VAL_MAX]*MAX_N))   # 100*100=10000 -> ちょうど M_MAX -> Yes
    cases.append(Case(MAX_N, M_MAX-1, [VAL_MAX]*MAX_N)) # 10000>9999 -> No
    cases.append(Case(MAX_N, 100, [1]*MAX_N))    # 和=100 -> ちょうど -> Yes
    cases.append(Case(MAX_N, 99,  [1]*MAX_N))    # 和=100 > 99 -> No

    # --- ランダムだけでは出にくい代表パターン
    cases.append(Case(10, 50, [5]*10))           # ぴったり -> Yes
    cases.append(Case(10, 49, [5]*10))           # 1 だけ足りない -> No
    cases.append(Case(7,  20, [1,2,3,4,5,2,3]))  # ちょうど20 -> Yes
    cases.append(Case(7,  19, [1,2,3,4,5,2,3]))  # 19 -> No

    return cases

def rand_array(rng: random.Random, n: int) -> List[int]:
    """[VAL_MIN, VAL_MAX] の整数をランダム生成。時々偏らせる。"""
    mode = rng.random()
    if mode < 0.15:
        # 全要素同一
        v = rng.randint(VAL_MIN, VAL_MAX)
        return [v]*n
    elif mode < 0.30:
        # 小さめ中心
        return [rng.randint(VAL_MIN, rng.randint(5, 30)) for _ in range(n)]
    elif mode < 0.45:
        # 大きめ中心
        lo = rng.randint(50, 90)
        return [rng.randint(lo, VAL_MAX) for _ in range(n)]
    else:
        # 一様
        return [rng.randint(VAL_MIN, VAL_MAX) for _ in range(n)]

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: Yes/No 半々、極端な N を混ぜる"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(4)]
    N = rng.choice(n_choices)
    A = rand_array(rng, N)
    S = sum(A)

    # Yes/No 制御（ちょうど S、±マージンを混ぜる）
    if rng.random() < 0.5:
        # Yes: M を S..M_MAX の範囲で
        if S > M_MAX:
            # 仕様上 S<=10000 のためここには来ないが保険
            M = M_MAX
        else:
            hi = M_MAX
            lo = max(M_MIN, S)
            M = rng.randint(lo, hi)
    else:
        # No: M を 1..S-1 から（S=1 のときは Yes にフォールバック）
        if S > M_MIN:
            M = rng.randint(M_MIN, S - 1)
        else:
            M = S  # S=1 -> Yes

    # ときどきぴったりにする
    if rng.random() < 0.15:
        M = min(max(S, M_MIN), M_MAX)

    return Case(N, M, A)

def case_to_input_text(case: Case) -> str:
    lines = [f"{case.N} {case.M}"]
    lines.append(" ".join(map(str, case.A)))
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
        assert M_MIN <= case.M <= M_MAX
        assert len(case.A) == case.N
        assert all(VAL_MIN <= a <= VAL_MAX for a in case.A)

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (sum={sum(case.A)}, M={case.M}, ans={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
