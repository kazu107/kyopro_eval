# copy_problem_dirs_ignore_outputs_no_args.py
# -*- coding: utf-8 -*-
"""
problems_llama/ 配下の各手法（baseline / CoT / feedback）にある ABC* ディレクトリを、
「outputs ディレクトリの中身だけコピーしない（空で作る）」ルールで別の場所へ複製します。

- argparse（引数）なし。上部の CONFIG を編集して使います。
- outputs 自体を作りたくない場合は MAKE_EMPTY_OUTPUTS=False にしてください。
"""

from __future__ import annotations
import shutil
from pathlib import Path

# ===================== CONFIG（ここだけ編集） =====================
SRC_ROOT = Path("problems_llama")          # 元
DST_ROOT = Path("problems_llama_cpp")     # 先（別フォルダ推奨）

METHODS = ["baseline", "CoT", "feedback"]  # 存在するものだけ処理
ABC_PATTERN = "ABC*"                       # 対象ディレクトリ名
MAKE_EMPTY_OUTPUTS = True                 # True: outputsは空で作る / False: outputs自体も作らない
FRESH_DST = False                         # True: 先フォルダを最初に削除して作り直す
DRY_RUN = False                           # True: 実行せずログだけ
# ================================================================


def safe_rmtree(p: Path):
    if p.exists():
        shutil.rmtree(p, ignore_errors=True)


def ensure_dir(p: Path):
    if DRY_RUN:
        print(f"[DRY] mkdir: {p}")
        return
    p.mkdir(parents=True, exist_ok=True)


def is_outputs_dir(path: Path) -> bool:
    return path.is_dir() and path.name == "outputs"


def copy_abc_dir(src_abc: Path, dst_abc: Path):
    """
    1つの ABC* ディレクトリをコピー。
    - MAKE_EMPTY_OUTPUTS=True なら outputs は空で作り、配下はコピーしない
    - MAKE_EMPTY_OUTPUTS=False なら outputs 自体をコピーしない
    """
    # 親階層(ABC*直下)で outputs を除外するかどうか
    def parent_ignore(cur_dir: str, names: list[str]) -> set[str]:
        if not MAKE_EMPTY_OUTPUTS and "outputs" in names:
            return {"outputs"}  # outputs自体を除外
        return set()

    if DRY_RUN:
        print(f"[DRY] copytree: {src_abc} -> {dst_abc} (empty_outputs={MAKE_EMPTY_OUTPUTS})")
        return

    shutil.copytree(
        src_abc,
        dst_abc,
        symlinks=False,
        dirs_exist_ok=True,
        ignore=parent_ignore
    )

    # outputs を空にする（元に outputs がある場合だけ）
    if MAKE_EMPTY_OUTPUTS and (src_abc / "outputs").exists():
        out = dst_abc / "outputs"
        out.mkdir(parents=True, exist_ok=True)
        # 念のため中身があれば削除
        for child in out.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                try:
                    child.unlink()
                except FileNotFoundError:
                    pass


def main():
    src_root = SRC_ROOT.resolve()
    dst_root = DST_ROOT.resolve()

    if not src_root.exists():
        raise FileNotFoundError(f"SRC_ROOT not found: {src_root}")

    if FRESH_DST:
        if DRY_RUN:
            print(f"[DRY] rmtree: {dst_root}")
        else:
            safe_rmtree(dst_root)

    ensure_dir(dst_root)

    total = 0
    for method in METHODS:
        src_method = src_root / method
        if not src_method.exists():
            continue

        dst_method = dst_root / method
        ensure_dir(dst_method)

        targets = sorted([p for p in src_method.glob(ABC_PATTERN) if p.is_dir()])
        print(f"[INFO] method={method} targets={len(targets)} pattern={ABC_PATTERN}")

        for src_abc in targets:
            dst_abc = dst_method / src_abc.name
            copy_abc_dir(src_abc, dst_abc)
            total += 1

    print(f"[DONE] processed ABC* folders: {total} (dry_run={DRY_RUN})")


if __name__ == "__main__":
    main()
