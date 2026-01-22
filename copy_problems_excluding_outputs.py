# copy_problems_excluding_outputs.py
# -*- coding: utf-8 -*-
"""
problems_llama フォルダ全体をコピーするが、以下は除外する:
  - problems_llama/baseline/ABC*/outputs
  - problems_llama/CoT/ABC*/outputs
  - problems_llama/feedback/ABC*/outputs

引数は使わず、コード内の設定を書き換えて使う。
"""

from __future__ import annotations
import re
from pathlib import Path
import shutil
from datetime import datetime

# ========= 設定（必要に応じて変更） ============================================
SRC_ROOT = Path("problems_llama")                     # コピー元
DST_BASE = Path("problems_qwen")              # コピー先のベース
APPEND_TIMESTAMP = False                         # True だと末尾にタイムスタンプを付与
CATEGORIES = {"baseline", "cot", "feedback"}    # 判定対象のカテゴリ名（大文字小文字無視）
ABC_DIR_PATTERN = re.compile(r"^abc\d{3}$", re.IGNORECASE)  # ABCxxx ディレクトリの判定
# ============================================================================

# 進捗メモ: 無視した outputs の一覧を格納
SKIPPED_OUTPUTS: list[Path] = []


def _make_dst_dir() -> Path:
    if APPEND_TIMESTAMP:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return DST_BASE.with_name(f"{DST_BASE.name}_{stamp}")
    return DST_BASE


def _is_abc_dir_name(name: str) -> bool:
    return bool(ABC_DIR_PATTERN.match(name))


def _should_ignore_outputs(cur_dir: Path, entry_names: list[str]) -> bool:
    """
    今いるディレクトリ（cur_dir）が「problems_llama/<cat>/<ABCxxx>」に当たっていて、
    かつ entries に 'outputs' が含まれていれば、除外対象。
    """
    try:
        rel = cur_dir.resolve().relative_to(SRC_ROOT.resolve())
    except Exception:
        return False

    parts = [p.lower() for p in rel.parts]  # ['baseline', 'abc420'] など
    if len(parts) >= 2 and parts[0] in CATEGORIES and _is_abc_dir_name(parts[1]):
        return "outputs" in entry_names
    return False


def _ignore_func(cur_dir: str, entry_names: list[str]) -> set[str]:
    """
    shutil.copytree の ignore 引数に渡す関数。
    cur_dir に対して、コピーしたくない entry 名の集合を返す。
    """
    p = Path(cur_dir)
    # 対象ディレクトリなら 'outputs' を無視する
    if _should_ignore_outputs(p, entry_names):
        SKIPPED_OUTPUTS.append(p / "outputs")
        return {"outputs"}
    # それ以外は何も無視しない
    return set()


def main():
    src = SRC_ROOT.resolve()
    if not src.exists() or not src.is_dir():
        raise FileNotFoundError(f"コピー元が見つかりません: {src}")

    dst = _make_dst_dir().resolve()
    print(f"[info] copy from: {src}")
    print(f"[info] copy to  : {dst}")

    # copytree: Python 3.8+ なら dirs_exist_ok が使える
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        src,
        dst,
        symlinks=False,
        ignore=_ignore_func,
        dirs_exist_ok=False  # 既存ならエラーにする（安全側）。上書きしたいなら True に変更
    )

    # 結果の要約
    if SKIPPED_OUTPUTS:
        print("[skip] excluded outputs directories:")
        for p in SKIPPED_OUTPUTS:
            print("  -", p.relative_to(src))
    else:
        print("[skip] no outputs directory matched to exclude.")

    print("[done] copy completed.")


if __name__ == "__main__":
    main()
