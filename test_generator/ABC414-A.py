from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC414/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC414/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 414            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC414 A） =========
MIN_N: int = 1
MAX_N: int = 100
T_MIN: int = 0
T_MAX: int = 23
# ======================================================

@dataclass
class Case:
    N: int
    L: int
    R: int
    XY: List[Tuple[int, int]]  # (X_i, Y_i)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力を計算（配信を最初から最後まで見られる人数）"""
    cnt = sum(1 for x, y in case.XY if x <= case.L and case.R <= y)
    return str(cnt)

# ---------- ケース生成ユーティリティ ----------
def sample_interval(rng: random.Random) -> Tuple[int, int]:
    """0..23 の整数で X<Y を満たす区間をランダム生成"""
    x = rng.randint(T_MIN, T_MAX - 1)
    y = rng.randint(x + 1, T_MAX)
    return x, y

def interval_covering_LR(rng: random.Random, L: int, R: int) -> Tuple[int, int]:
    """[X,Y] が [L,R] を被覆する (X<=L<=R<=Y) 区間を生成"""
    x = rng.randint(T_MIN, L)
    y = rng.randint(R, T_MAX)
    # L<R なので必ず x<y を満たせる
    return x, y

def interval_not_covering_LR(rng: random.Random, L: int, R: int) -> Tuple[int, int]:
    """[X,Y] が [L,R] を被覆しない区間を生成（X>L または Y<R）"""
    # リジェクションで十分高速
    for _ in range(1000):
        x, y = sample_interval(rng)
        if not (x <= L and R <= y):
            return x, y
    # フォールバック（明示的に崩す）
    if L < T_MAX - 1:
        x = min(T_MAX - 1, L + 1)
        y = rng.randint(x + 1, T_MAX)
        return x, y  # X>L
    else:
        x = rng.randint(T_MIN, R - 1)
        y = max(x + 1, R - 1)
        return x, y  # Y<R

def make_directed_cases() -> List[Case]:
    """網羅用のハンドクラフトケース（サンプル相当＋境界・代表パターン）"""
    cases: List[Case] = []

    # --- 公式サンプル
    cases.append(Case(5, 19, 22, [(17,23), (20,23), (19,22), (0,23), (12,20)]))  # 出力3
    cases.append(Case(3, 12, 13, [(0,1), (0,1), (0,1)]))                         # 出力0
    cases.append(Case(10, 8, 14, [(5,20),(14,21),(9,21),(5,23),(8,10),(0,14),(3,8),(2,6),(0,16),(5,20)]))  # 出力5

    # --- N最小/最大、端値
    cases.append(Case(1, 0, 1, [(0,1)]))         # 1人全被覆（最小）
    cases.append(Case(1, 0, 1, [(1,2)]))         # 0人（X>L）
    cases.append(Case(MAX_N, 0, 23, [(0,23)]*MAX_N))   # 全員見られる（最大N、両端）
    cases.append(Case(MAX_N, 22, 23, [(0,22)]*MAX_N))  # 0人（Y<R）

    # --- 境界一致（= を多用）
    cases.append(Case(4, 22, 23, [(22,23), (0,23), (22,23), (0,22)]))  # 3人
    cases.append(Case(6, 0, 23, [(0,23), (0,22), (1,23), (0,23), (5,22), (1,10)]))  # 2人

    # --- 代表的な失敗パターン（開始遅い／終了早い）
    cases.append(Case(5, 10, 15, [(11,23), (0,14), (0,15), (10,14), (5,9)]))  # 0人

    # --- 被覆と非被覆が混在
    mix = [(0,10), (5,20), (10,15), (0,23), (7,14), (1,23), (0,9), (10,23)]
    cases.append(Case(len(mix), 7, 14, mix))  # 4人: (5,20),(0,23),(1,23),(10,23)

    return cases

def make_random_case(rng: random.Random) -> Case:
    """ランダムケース: 極端な N や L,R を混ぜつつ、Yes/No をバランスよく作る"""
    n_choices = [MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)]
    N = rng.choice(n_choices)

    # L,R は 0..23 の整数で L<R
    L = rng.randint(T_MIN, T_MAX - 1)
    R = rng.randint(L + 1, T_MAX)

    XY: List[Tuple[int, int]] = []
    # 被覆/非被覆を半々くらいにミックス
    for _ in range(N):
        if rng.random() < 0.5:
            XY.append(interval_covering_LR(rng, L, R))
        else:
            XY.append(interval_not_covering_LR(rng, L, R))

    return Case(N, L, R, XY)

def case_to_input_text(case: Case) -> str:
    lines = [f"{case.N} {case.L} {case.R}"]
    lines.extend(f"{x} {y}" for x, y in case.XY)
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

        # 制約チェック（公式）: 1 ≤ N ≤ 100, 0 ≤ L < R ≤ 23, 0 ≤ X_i < Y_i ≤ 23
        assert MIN_N <= case.N <= MAX_N
        assert T_MIN <= case.L < case.R <= T_MAX
        assert len(case.XY) == case.N
        assert all(T_MIN <= x < y <= T_MAX for x, y in case.XY)

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (answer={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
