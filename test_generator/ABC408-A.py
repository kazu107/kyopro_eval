from __future__ import annotations
import os
import random
from dataclasses import dataclass
from typing import List, Tuple

# ========= ユーザー編集セクション (Edit here) =========
NUM_CASES: int = 97                  # 生成するテストケース総数
START_INDEX: int = 4                  # 連番の開始番号（例: 4 -> case004 から開始）
INPUT_DIR: str = "../problems_llama/baseline/ABC408/inputs"  # 入力ファイル保存先ディレクトリ
OUTPUT_DIR: str = "../problems_llama/baseline/ABC408/testcases"  # 出力ファイル保存先ディレクトリ
FILE_PREFIX: str = "case"             # ファイル名プリフィクス
INPUT_EXT: str = ".txt"               # 入力ファイル拡張子
OUTPUT_SUFFIX: str = ".out.txt"       # 出力ファイル拡張子
RNG_SEED: int | None = 408            # 乱数シード (再現性が必要な場合は整数、ランダムにする場合は None)
# ========= 問題固有の既定値（ABC408 A） =========
MIN_N: int = 1
MAX_N: int = 100
S_MIN: int = 1
S_MAX: int = 100
T_MIN: int = 1
T_MAX: int = 1000
# ======================================================

@dataclass
class Case:
    N: int
    S: int
    T: List[int]

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def solve_case(case: Case) -> str:
    """期待出力 ('Yes' / 'No')"""
    prev = 0
    for t in case.T:
        if t - prev > case.S:
            return "No"
        prev = t
    return "Yes"

# ---------- ユーティリティ ----------
def prefix_from_increments(incs: List[int]) -> List[int]:
    s = 0
    out = []
    for d in incs:
        s += d
        out.append(s)
    return out

# ---------- ハンドクラフト（網羅） ----------
def make_directed_cases() -> List[Case]:
    cases: List[Case] = []

    # --- 公式サンプル
    cases.append(Case(5, 10, [6, 11, 21, 22, 30]))  # Yes
    cases.append(Case(2, 100, [1, 200]))            # No
    cases.append(Case(10, 22, [47,81,82,95,117,146,165,209,212,215]))  # No

    # --- N最小/最大、端値と等号境界
    cases.append(Case(1, 1, [1]))                   # 1-0=1 -> Yes
    cases.append(Case(1, 1, [2]))                   # 2-0=2 > 1 -> No
    cases.append(Case(3, 5, [5,10,15]))             # すべて =S -> Yes
    cases.append(Case(3, 5, [5,11,12]))             # 2区間目 6>S -> No

    # --- 大きめ N、T_N=上限付近
    incs = [10]*100  # 合計1000
    cases.append(Case(100, 10, prefix_from_increments(incs)))  # すべて =S -> Yes

    # --- 途中で眠る／最後で眠る
    cases.append(Case(6, 7, [3,7,14,15,16,17]))     # 14-7=7 はOK、15-14=1, …, ただし3-0=3<=7 -> 最後まで Yes
    cases.append(Case(6, 7, [3,12,20,21,22,23]))    # 12-3=9>S -> 途中で No
    cases.append(Case(6, 7, [3,9,16,23,30,38]))     # 最後 38-30=8>S -> No

    # --- S と間隔のバリエーション
    cases.append(Case(8, 1, [1,2,3,4,5,6,7,8]))     # すべて 1 -> Yes
    cases.append(Case(8, 1, [1,2,3,4,6,7,8,9]))     # 6-4=2>S -> No

    return cases

# ---------- ランダムケース ----------
def make_yes_case(rng: random.Random) -> Case:
    N = rng.choice([MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)])
    S = rng.randint(S_MIN, S_MAX)
    # 合計が T_MAX を超えにくいように一回の増分を抑える
    inc_upper = min(S, max(1, T_MAX // N))
    incs = [rng.randint(1, inc_upper) for _ in range(N)]
    T = prefix_from_increments(incs)
    # 念のため上限調整（越えたら縮める）
    if T[-1] > T_MAX:
        scale = T_MAX / T[-1]
        # 1 以上を維持しつつ丸め直し
        incs2 = [max(1, int(d * scale)) for d in incs]
        # 最後に不足分を前から足す（S を超えない）
        for i in range(N):
            while sum(incs2) < T_MAX and incs2[i] < S:
                incs2[i] += 1
        T = prefix_from_increments(incs2)
        # 依然超えるなら切り詰め
        while T and T[-1] > T_MAX:
            incs2[-1] -= 1
            T = prefix_from_increments(incs2)
    return Case(N, S, T)

def make_no_case(rng: random.Random) -> Case:
    N = rng.choice([MIN_N, MAX_N] + [rng.randint(2, MAX_N - 1) for _ in range(3)])
    S = rng.randint(S_MIN, S_MAX)

    # まず全て1で初期化（最低増分）
    incs = [1] * N
    # 1つの位置を S+1 にして必ず失敗させる
    pos = rng.randrange(N)
    incs[pos] = S + 1

    # 残りの余裕（合計 ≤ 1000）をランダム配分。ただし各増分は大きくし過ぎない
    slack = T_MAX - (N - 1 + (S + 1))
    for i in range(N):
        if slack <= 0:
            break
        add = rng.randint(0, min(slack, max(0, S)))  # Yes条件は不要だが暴れ過ぎ防止
        incs[i] += add
        slack -= add

    T = prefix_from_increments(incs)
    # 念のため厳密チェック
    assert any(incs[i] > S for i in range(N))
    assert T[-1] <= T_MAX
    return Case(N, S, T)

def make_random_case(rng: random.Random) -> Case:
    return make_yes_case(rng) if rng.random() < 0.5 else make_no_case(rng)

# ---------- I/O ----------
def case_to_input_text(case: Case) -> str:
    lines = [f"{case.N} {case.S}", " ".join(map(str, case.T))]
    return "\n".join(lines) + "\n"

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

    directed = make_directed_cases()

    cases: List[Case] = list(directed)
    while len(cases) < NUM_CASES:
        cases.append(make_random_case(rng))
    cases = cases[:NUM_CASES]

    # ゼロ埋め桁数：少なくとも3桁
    end_index = START_INDEX + len(cases) - 1
    width = max(3, len(str(end_index)))

    for offset, case in enumerate(cases):
        abs_idx = START_INDEX + offset

        # 制約チェック（公式）
        assert MIN_N <= case.N <= MAX_N
        assert S_MIN <= case.S <= S_MAX
        assert len(case.T) == case.N
        assert all(T_MIN <= t <= T_MAX for t in case.T)
        assert all(case.T[i] < case.T[i+1] for i in range(case.N - 1))

        in_path, out_path = write_case_files(abs_idx, width, case, INPUT_DIR, OUTPUT_DIR)
        print(f"[{abs_idx:0{width}d}] -> {in_path} , {out_path}  (ans={solve_case(case)})")

    print(f"\nDone. Generated {len(cases)} cases.")
    print(f"Inputs : {os.path.abspath(INPUT_DIR)}")
    print(f"Outputs: {os.path.abspath(OUTPUT_DIR)}")
    print(f"Start  : {START_INDEX}, End: {end_index}, Width: {width}")

if __name__ == "__main__":
    main()
