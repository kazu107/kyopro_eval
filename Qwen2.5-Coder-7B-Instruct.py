# solve_pipeline.py
# -*- coding: utf-8 -*-
"""
[ベース版パイプライン / 複数問題順次処理対応 / Qwen2.5-Coder-7B-Instruct(ローカル)]

問題フォルダ（例: problems_llama/baseline/ABC421）にある
  - prompt.txt                    : LLMに渡すプロンプト
  - inputs/*.txt                  : 各入力
  - testcases/<name>.out.txt      : 期待出力（正規表現は先頭行に 'regex:'）
を用いて、

  1) ローカルに保存した Qwen/Qwen2.5-Coder-7B-Instruct を 4bit 量子化でロード（1回）
  2) 各問題ごとに N_TRIES 回 生成 → ```python``` 抽出 → Docker で各入力を並列実行
     - timeout / time 出力で 1ms 精度の時間、Max RSS を取得（TLE は時間平均から除外）
     - exit code 137 を MLE として判定
  3) 期待出力と照合（数値は誤差許容、regex対応）、verdict.json 保存
  4) iterごと/全体の集計、best_solution.py を保存
  5) overall_summary.json に pass@k 用の tries 配列と平均統計（time/mem/code_bytes）を保存

★ 完全オフライン運用したい場合は、環境変数と MODEL_LOCAL_DIR をローカルフォルダに設定し、
  transformers の local_files_only=True を有効化しています。
"""

import os
# ---- HF キャッシュとオフライン設定（必要に応じて編集）----
os.environ.setdefault("HF_HOME", r"D:\hf")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf\transformers")
os.environ.setdefault("HF_HUB_CACHE", r"D:\hf\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf\datasets")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


import re
import json
import math
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import time as pytime
import random as pyrandom
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from concurrent.futures import ThreadPoolExecutor, as_completed  # 並列化

# ★ 事前にダウンロードしておいた Qwen2.5-Coder-7B-Instruct のローカルフォルダを指定
MODEL_LOCAL_DIR = Path(r"D:\models\Qwen2.5-Coder-7B-Instruct")

# ===== ユーザー設定 ==================================================
# 複数問題を順番に処理：必要な分だけ列挙
PROBLEM_DIRS: List[Path] = [
    Path("problems_qwen/baseline/ABC405"),
    Path("problems_qwen/baseline/ABC404"),
    Path("problems_qwen/baseline/ABC403"),
    Path("problems_qwen/baseline/ABC402"),
    Path("problems_qwen/baseline/ABC401"),
    Path("problems_qwen/baseline/ABC400"),
    Path("problems_qwen/baseline/ABC399"),
    Path("problems_qwen/baseline/ABC398"),
    Path("problems_qwen/baseline/ABC397"),
]

# メタ情報（ログ用）
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct (local)"

# 生成（LLM）設定
N_TRIES = 100
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.7
TOP_P = 0.95
SEED_BASE = 42  # 再現性担保用（各iterで SEED_BASE+i を設定）

# Docker 実行設定
DOCKER_IMAGE = "extracted-exec:py312"     # 事前に Dockerfile でビルド（python3.12想定）
DOCKER_MEMORY_LIMIT = "1g"                # MLE 判定用メモリ上限（例: "1g", "2g"）
DOCKER_CPUS = 1                           # None なら未指定
RUN_TIMEOUT_SEC = 10                      # TLE 判定用（秒）
PYTHON_IN_CONTAINER = "python"            # コンテナ内の python コマンド

# 並列数（各 iter 内で同時に起動するコンテナ数）
PARALLEL_JOBS = os.cpu_count() or 4

# 採点設定
EPS_ABS = 1e-6                            # 数値比較の絶対誤差
EPS_REL = 1e-6                            # 数値比較の相対誤差
NORMALIZE_WHITESPACE = True               # 文字列比較時に空白正規化

# 4bit 計算 dtype（RTX 3060 は fp16 が安定）
FOURBIT_COMPUTE_DTYPE = torch.float16

# 生成安定化 system プロンプト（Qwen の chat-template 利用）
SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Your answer should be a single block of code enclosed in ```python."
    "The program must read from standard input and write to standard output. "
)
# ====================================================================


# ---------- ユーティリティ（problem_dir 引数付き） ----------
def jst_now_iso() -> str:
    jst = timezone(timedelta(hours=9))
    return datetime.now(jst).isoformat(timespec="seconds")


def ensure_dirs(problem_dir: Path):
    (problem_dir / "outputs").mkdir(parents=True, exist_ok=True)


def load_prompt(problem_dir: Path) -> str:
    p = problem_dir / "prompt.txt"
    if not p.exists():
        raise FileNotFoundError(f"prompt.txt が見つかりません: {p}")
    return p.read_text(encoding="utf-8")


def list_inputs(problem_dir: Path) -> List[Path]:
    d = problem_dir / "inputs"
    if not d.exists():
        return []
    return sorted([p for p in d.glob("*.txt") if p.is_file()])


def testcase_path_for_input(problem_dir: Path, inp: Path) -> Path:
    base = inp.stem
    return problem_dir / "testcases" / f"{base}.out.txt"


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def normalize_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    if NORMALIZE_WHITESPACE:
        s = "\n".join(" ".join(line.split()) for line in s.split("\n"))
    if s and not s.endswith("\n"):
        s += "\n"
    return s


def is_float_token(x: str) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


def compare_as_numbers_linewise(expected: str, actual: str) -> bool:
    elines = normalize_text(expected).splitlines()
    alines = normalize_text(actual).splitlines()
    if len(elines) != len(alines):
        return False
    for e, a in zip(elines, alines):
        et = e.split()
        at = a.split()
        if len(et) != len(at):
            return False
        if all(is_float_token(t) for t in et) and all(is_float_token(t) for t in at):
            for u, v in zip(et, at):
                if not math.isclose(float(u), float(v), rel_tol=EPS_REL, abs_tol=EPS_ABS):
                    return False
        else:
            if e != a:
                return False
    return True


def compare_with_regex(expected: str, actual: str) -> bool:
    pat = expected.strip().split("\n", 1)[0][len("regex:"):].strip()
    flags = re.MULTILINE | re.DOTALL
    try:
        return re.fullmatch(pat, actual, flags) is not None
    except re.error:
        return False


def first_mismatch(expected: str, actual: str) -> Optional[Dict[str, Any]]:
    elines = normalize_text(expected).splitlines()
    alines = normalize_text(actual).splitlines()
    from itertools import zip_longest
    for i, (e, a) in enumerate(zip_longest(elines, alines, fillvalue="")):
        if e != a:
            pos = 0
            for j, (ce, ca) in enumerate(zip_longest(e, a, fillvalue="")):
                if ce != ca:
                    pos = j
                    break
            return {
                "line": i + 1,
                "col": pos + 1,
                "expected_line": e[:200],
                "actual_line": a[:200],
            }
    return None


def judge_one(expected_path: Path, actual_text: str) -> Dict[str, Any]:
    if not expected_path.exists():
        return {"status": "no_expected", "pass": False, "mode": "missing_expected"}
    exp_text = read_text(expected_path)
    if exp_text.lstrip().startswith("regex:"):
        ok = compare_with_regex(exp_text, actual_text)
        return {"status": "AC" if ok else "WA", "pass": bool(ok), "mode": "regex"}
    ok = compare_as_numbers_linewise(exp_text, actual_text)
    return {"status": "AC" if ok else "WA", "pass": bool(ok), "mode": "num_or_str"}


# ---------- LLM 生成（Qwen2.5-Coder ローカル） ----------
CODE_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)


def resolve_model_dir(p: Path) -> Path:
    p = Path(p)
    if p.is_file():
        p = p.parent
    # すでに model ファイルが揃っている（config.json が直下）ならそのまま
    if (p / "config.json").exists():
        return p
    # HF Hub のキャッシュ形式 .../models--org--name/snapshots/<sha>/ を探す
    snap_root = p / "snapshots"
    if snap_root.exists() and snap_root.is_dir():
        # いちばん新しいスナップショットを選ぶ
        cand = sorted([d for d in snap_root.iterdir() if d.is_dir()],
                      key=lambda d: d.stat().st_mtime, reverse=True)
        for d in cand:
            if (d / "config.json").exists():
                return d
    # 1階層下にそのまま格納しているケースも救う
    for d in p.iterdir() if p.exists() and p.is_dir() else []:
        if d.is_dir() and (d / "config.json").exists():
            return d
    raise FileNotFoundError(
        f"モデルフォルダが見つかりません。'config.json' を含むディレクトリを指してください: {p}"
    )

def get_qwen_eos_id(tok) -> int:
    # Qwen2.5 系は通常 <|endoftext|> が eos。なければ tok.eos_token_id を使用
    cand_tokens = ["<|endoftext|>", "<|im_end|>"]
    ids = []
    for t in cand_tokens:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    # tokenizer に登録済みの eos_token_id を最優先
    if isinstance(getattr(tok, "eos_token_id", None), int) and tok.eos_token_id >= 0:
        return tok.eos_token_id
    if ids:
        return ids[0]
    raise RuntimeError("Could not resolve a valid eos_token_id for Qwen tokenizer.")

def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_DIR,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,
    )
    if tok.pad_token_id is None:
        # Qwen は pad なしの場合があるので eos を pad に流用
        tok.pad_token = tok.eos_token if getattr(tok, "eos_token", None) else "</s>"
    return tok

def build_model_4bit_full_gpu():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが無効です。")
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        trust_remote_code=True,
        quantization_config=qconf,
        device_map={'': 0},
        # ★ Windows+4bit では sdpa の方が安定することが多い
        attn_implementation="sdpa",
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model

def generate_once(tok: AutoTokenizer, model: AutoModelForCausalLM, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    prompt_text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    # 事前に入力が空でないかチェック
    enc = tok(prompt_text, return_tensors="pt")
    if enc["input_ids"].numel() == 0 or enc["input_ids"].shape[1] == 0:
        raise RuntimeError("Tokenized prompt is empty. Check chat template / SYSTEM_PROMPT.")

    # pad/eos を明示・一致させる
    eos_id = get_qwen_eos_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    inputs = {k: v.to(model.device) for k, v in enc.items()}
    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=pad_id,
            eos_token_id=eos_id,   # ← ★ 単一の int を渡す（リストにしない）
        )
    gen_only = gen_ids[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_only, skip_special_tokens=False)
    return text.strip()

def set_seed_for_iter(i: int):
    torch.manual_seed(SEED_BASE + i)
    torch.cuda.manual_seed_all(SEED_BASE + i)
    pyrandom.seed(SEED_BASE + i)


def get_terminators(tok) -> List[int]:
    """
    Qwen 系でよく使われる終端トークンを複数登録（存在するものだけ）。
    """
    ids = []
    if tok.eos_token_id is not None:
        ids.append(tok.eos_token_id)
    for t in ["<|im_end|>", "<|endoftext|>", "<|eot_id|>"]:
        try:
            tid = tok.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid != tok.eos_token_id and tid not in ids:
                ids.append(tid)
        except Exception:
            pass
    return ids or [tok.eos_token_id]


def apply_chat_template(tok, user_prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def generate_once(tok, model, user_prompt: str) -> str:
    prompt_text = apply_chat_template(tok, user_prompt)
    inputs = tok(prompt_text, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=tok.eos_token_id,
            eos_token_id=get_terminators(tok),
        )
    gen_only = gen_ids[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_only, skip_special_tokens=False)

    # Qwen 系の終了記号や特殊トークンを軽く掃除
    for tkn in ("<|im_end|>", "<|endoftext|>", "<|eot_id|>"):
        text = text.replace(tkn, "")
    return text.strip()


def extract_python_code(text: str) -> Optional[str]:
    pref = {"python", "py", "python3"}
    candidates = []
    for m in CODE_FENCE_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower()
        body = m.group(2)
        candidates.append((lang, body))
    for lang, body in candidates:
        if lang in pref:
            return body.strip()
    for lang, body in candidates:
        if lang == "":
            return body.strip()
    return None


def write_iter_files(iter_dir: Path, response_text: str, code: Optional[str], meta: Dict[str, Any]):
    iter_dir.mkdir(parents=True, exist_ok=True)
    (iter_dir / "response.txt").write_text(response_text, encoding="utf-8")
    if code is not None:
        (iter_dir / "extracted.py").write_text(code, encoding="utf-8")
    else:
        (iter_dir / "extracted.py").write_text("# extraction failed\n", encoding="utf-8")
    (iter_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------- Docker 実行 ----------
def docker_available() -> bool:
    try:
        subprocess.run(["docker", "version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False


def parse_time_file(path: Path) -> Dict[str, Optional[float]]:
    """
    runs/<case>/time.txt を多形式で解析:
    - key=value: elapsed_msec=.., elapsed_sec=.., user_sec=.., sys_sec=.., maxrss_kb=..
    - GNU time -v の英語行（フォールバック）
    - time -p の real/user/sys（フォールバック）
    """
    out = {"wall_time_sec": None, "max_rss_kb": None, "user_time_sec": None, "sys_time_sec": None}
    if not path.exists():
        return out
    txt = path.read_text(encoding="utf-8", errors="ignore")

    # 1) key=value 形式
    kv = {}
    for line in txt.splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k.strip()] = v.strip()

    def f2(x):
        try:
            return float(x)
        except Exception:
            return None

    def i2(x):
        try:
            return int(x)
        except Exception:
            return None

    if kv:
        if "elapsed_msec" in kv:
            try:
                out["wall_time_sec"] = int(kv["elapsed_msec"]) / 1000.0
            except Exception:
                pass
        if out["wall_time_sec"] is None and "elapsed_sec" in kv:
            out["wall_time_sec"] = f2(kv.get("elapsed_sec"))
        out["user_time_sec"] = f2(kv.get("user_sec"))
        out["sys_time_sec"] = f2(kv.get("sys_sec"))
        out["max_rss_kb"] = i2(kv.get("maxrss_kb"))
        if any(v is not None for v in out.values()):
            return out

    # 2) GNU time -v
    def to_seconds(s: str):
        s = s.strip()
        parts = s.split(":")
        try:
            if len(parts) == 3:
                h, m, sec = int(parts[0]), int(parts[1]), float(parts[2])
                return h * 3600 + m * 60 + sec
            if len(parts) == 2:
                m, sec = int(parts[0]), float(parts[1])
                return m * 60 + sec
            return float(s)
        except Exception:
            return None

    any_hit = False
    for line in txt.splitlines():
        if "Elapsed (wall clock) time" in line:
            out["wall_time_sec"] = to_seconds(line.split(":", 1)[-1])
            any_hit = True
        elif "Maximum resident set size" in line:
            try:
                out["max_rss_kb"] = int(line.split(":")[-1].strip().split()[0])
                any_hit = True
            except Exception:
                pass
        elif "User time (seconds)" in line:
            try:
                out["user_time_sec"] = float(line.split(":")[-1].strip().split()[0])
                any_hit = True
            except Exception:
                pass
        elif "System time (seconds)" in line:
            try:
                out["sys_time_sec"] = float(line.split(":")[-1].strip().split()[0])
                any_hit = True
            except Exception:
                pass
    if any_hit:
        return out

    # 3) time -p (real/user/sys)
    real = user = sys = None
    for line in txt.splitlines():
        if line.startswith("real"):
            try:
                real = float(line.split()[1])
            except Exception:
                pass
        elif line.startswith("user"):
            try:
                user = float(line.split()[1])
            except Exception:
                pass
        elif line.startswith("sys"):
            try:
                sys = float(line.split()[1])
            except Exception:
                pass
    if (real is not None) or (user is not None) or (sys is not None):
        out["wall_time_sec"] = real
        out["user_time_sec"] = user
        out["sys_time_sec"] = sys
    return out


def run_in_docker(iter_dir: Path, run_dir: Path, input_path: Path, timeout_sec: int) -> Tuple[int, str]:
    """
    iter_dir を /work にマウントして、extracted.py を実行。
    - stdin: runs/<case>/input.txt を < リダイレクトで供給
    - argv[1]: "input.txt" を渡す
    - 環境変数: INPUT_FILE, INPUT_BASENAME
    - time の出力 → runs/<case>/time.txt（elapsed_msec/elapsed_sec/maxrss 等）
    - 戻り値: (exit_code, docker_stderr_tail)
    """
    abs_iter = str(iter_dir.resolve())
    case_name = input_path.stem
    in_container_case_dir = f"runs/{case_name}"

    inner_cmd = (
        'if [ -x /usr/bin/time ]; then '
        '  __START_NS=$(date +%s%N); '
        f'  timeout -k 1s {timeout_sec}s '
        f'  /usr/bin/time -f "maxrss_kb=%M\nuser_sec=%U\nsys_sec=%S" '
        f'  -o "{in_container_case_dir}/time_extra.txt" '
        f'  {PYTHON_IN_CONTAINER} "extracted.py" "input.txt" '
        f'  < "{in_container_case_dir}/input.txt" '
        f'  1>"{in_container_case_dir}/stdout.txt" '
        f'  2>"{in_container_case_dir}/prog_stderr.txt"; '
        '  __RC=$?; '
        '  __END_NS=$(date +%s%N); '
        '  __ELAPSED_MS=$(( (__END_NS - __START_NS) / 1000000  )); '
        f'  {{ echo "elapsed_msec=$__ELAPSED_MS"; '
        f'     awk -v ms="$__ELAPSED_MS" \'BEGIN{{printf("elapsed_sec=%.3f\\n", ms/1000.0)}}\'; }} '
        f'     > "{in_container_case_dir}/time.txt"; '
        f'  cat "{in_container_case_dir}/time_extra.txt" >> "{in_container_case_dir}/time.txt" 2>/dev/null || true; '
        '  exit $__RC; '
        'else '
        '  __START_NS=$(date +%s%N); '
        f'  TIMEFORMAT=$\'user_sec=%U\\nsys_sec=%S\'; '
        f'  {{ time -p {PYTHON_IN_CONTAINER} "extracted.py" "input.txt" '
        f'     < "{in_container_case_dir}/input.txt" '
        f'     1>"{in_container_case_dir}/stdout.txt" '
        f'     2>"{in_container_case_dir}/prog_stderr.txt"; }} '
        f'  2>"{in_container_case_dir}/time_extra.txt"; '
        '  __RC=$?; '
        '  __END_NS=$(date +%s%N); '
        '  __ELAPSED_MS=$(( (__END_NS - __START_NS) / 1000000 )); '
        f'  {{ echo "elapsed_msec=$__ELAPSED_MS"; '
        f'     awk -v ms="$__ELAPSED_MS" \'BEGIN{{printf("elapsed_sec=%.3f\\n", ms/1000.0)}}\'; }} '
        f'     > "{in_container_case_dir}/time.txt"; '
        f'  cat "{in_container_case_dir}/time_extra.txt" >> "{in_container_case_dir}/time.txt" 2>/dev/null || true; '
        '  exit $__RC; '
        'fi'
    )

    cmd = [
        "docker", "run", "--rm",
        "-v", f"{abs_iter}:/work",
        "-w", "/work",
        "--memory", DOCKER_MEMORY_LIMIT,
        "--memory-swap", DOCKER_MEMORY_LIMIT,
    ]
    if DOCKER_CPUS:
        cmd += ["--cpus", str(DOCKER_CPUS)]
    cmd += [
        "-e", f"INPUT_FILE=/work/{in_container_case_dir}/input.txt",
        "-e", f"INPUT_BASENAME={input_path.name}",
        DOCKER_IMAGE,
        "/bin/bash", "-lc", inner_cmd,
    ]

    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        rc = proc.returncode
        tail = (proc.stderr or "")[-1000:]
        return rc, tail
    except Exception as e:
        (run_dir / "prog_stderr.txt").write_text(f"[runner] docker run failed: {e}\n", encoding="utf-8")
        return -1, str(e)


# ---------- 集計 ----------
def summarize_iter(iter_dir: Path, results: List[Dict[str, Any]]):
    counts = {"AC": 0, "WA": 0, "TLE": 0, "RE": 0, "MLE": 0, "NOEXP": 0}
    passed_cases = 0
    for r in results:
        st = r.get("status")
        if st in counts:
            counts[st] += 1
        if r.get("verdict", {}).get("pass"):
            passed_cases += 1
    total_cases = len(results)
    iter_pass_all = (counts["AC"] == total_cases and total_cases > 0)

    summary = {
        "timestamp_jst": jst_now_iso(),
        "iter_dir": iter_dir.name,
        "total_cases": total_cases,
        "passed": passed_cases,
        "failed": total_cases - passed_cases,
        "counts": counts,
        "iter_pass_all": iter_pass_all,
        "cases": results,
    }
    (iter_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return iter_pass_all, passed_cases, total_cases, counts


# ---------- 1ケース実行（並列用ワーカー） ----------
def execute_case(index: int, problem_dir: Path, iter_dir: Path, inp: Path, code_bytes: int) -> Tuple[int, Dict[str, Any]]:
    run_dir = iter_dir / "runs" / inp.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # 入力スナップショット（CRLF→LF、BOM除去で安定化）
    inp_text = read_text(inp)
    write_input_snapshot(run_dir / "input.txt", inp_text)

    # 実行（Docker）
    rc, docker_err_tail = run_in_docker(iter_dir, run_dir, inp, RUN_TIMEOUT_SEC)

    # time の結果
    time_info = parse_time_file(run_dir / "time.txt")
    wall = time_info.get("wall_time_sec")
    maxrss = time_info.get("max_rss_kb")

    # stdout / stderr
    stdout_txt = (run_dir / "stdout.txt").read_text(encoding="utf-8", errors="ignore") if (run_dir / "stdout.txt").exists() else ""
    stderr_prog = (run_dir / "prog_stderr.txt").read_text(encoding="utf-8", errors="ignore") if (run_dir / "prog_stderr.txt").exists() else ""
    stderr_tail = (stderr_prog or docker_err_tail)[-2000:]

    # ステータス分類
    status = "RE"
    detail = None
    verdict_core = {"status": "RE", "pass": False, "mode": None}

    if rc == 124:        # timeout
        status = "TLE"
        detail = "timeout"
    elif rc == 137:      # OOM
        status = "MLE"
        detail = "oom_killed"
    elif rc == 0:
        expected_p = testcase_path_for_input(problem_dir, inp)
        j = judge_one(expected_p, stdout_txt)
        status = "AC" if j["pass"] else "WA"
        if not j["pass"] and j.get("mode") != "regex":
            mm = first_mismatch(read_text(expected_p) if expected_p.exists() else "", stdout_txt)
            detail = {"mismatch": mm} if mm else None
        verdict_core = j
    else:
        status = "RE"
        detail = f"nonzero_exit({rc})"

    expected_file_name = testcase_path_for_input(problem_dir, inp).name if testcase_path_for_input(problem_dir, inp).exists() else None
    verdict = {
        "input_file": inp.name,
        "expected_file": expected_file_name,
        "returncode": rc,
        "status": status,                # "AC" / "WA" / "TLE" / "RE" / "MLE"
        "detail": detail,
        "time_sec": wall,                # 1ms精度（秒）
        "peak_memory_kb": maxrss,
        "code_bytes": code_bytes,
        "stderr_tail": stderr_tail,
        "verdict": verdict_core,
    }
    (run_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return index, verdict


# ---------- 1問題フォルダを処理 ----------
def process_problem(problem_dir: Path, tok: AutoTokenizer, model: AutoModelForCausalLM):
    print(f"\n===== Processing problem: {problem_dir} =====")
    ensure_dirs(problem_dir)
    if not docker_available():
        raise RuntimeError("docker が見つかりません。Docker Desktop などを起動し、パスが通っているか確認してください。")

    inputs = list_inputs(problem_dir)
    if not inputs:
        raise RuntimeError(f"inputs/*.txt が見つかりません: {problem_dir / 'inputs'}")

    user_prompt = load_prompt(problem_dir)

    all_iter_stats: List[Dict[str, Any]] = []

    # 全iter平均用の集計（time は TLE 除外）
    sum_time_sec = 0.0
    cnt_time = 0
    sum_peak_kb = 0
    cnt_peak = 0
    sum_code_bytes = 0
    cnt_code_bytes = 0

    for i in range(1, N_TRIES + 1):
        set_seed_for_iter(i)
        iter_dir = problem_dir / "outputs" / f"iter_{i:03d}"
        print(f"[gen] iter {i}/{N_TRIES} …")

        # 1) 生成（高精度タイマ）
        t0 = pytime.perf_counter()
        response = generate_once(tok, model, user_prompt)
        gen_dt = round(pytime.perf_counter() - t0, 3)  # 秒（小数3桁=1ms）

        code = extract_python_code(response)
        code_bytes = len(code.encode("utf-8")) if code is not None else 0

        meta = {
            "timestamp_jst": jst_now_iso(),
            "iteration": i,
            "model": MODEL_ID,
            "quantization": "bnb_4bit_nf4",
            "compute_dtype": str(FOURBIT_COMPUTE_DTYPE).replace("torch.", ""),
            "gen_params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P},
            "gen_time_sec": gen_dt,
            "extraction": {"found": code is not None, "code_bytes": code_bytes},
        }
        write_iter_files(iter_dir, response, code, meta)

        # iter単位の code_bytes を平均用に計上
        sum_code_bytes += code_bytes
        cnt_code_bytes += 1

        per_case_results: List[Dict[str, Any]] = []

        # 2) 抽出失敗 → 全ケース RE 扱いで記録
        if code is None:
            for inp in inputs:
                run_dir = iter_dir / "runs" / inp.stem
                run_dir.mkdir(parents=True, exist_ok=True)
                write_input_snapshot(run_dir / "input.txt", read_text(inp))
                expected_p = testcase_path_for_input(problem_dir, inp)
                verdict = {
                    "input_file": inp.name,
                    "expected_file": expected_p.name if expected_p.exists() else None,
                    "returncode": -2,
                    "status": "RE",
                    "detail": "code_extraction_failed",
                    "time_sec": None,
                    "peak_memory_kb": None,
                    "code_bytes": 0,
                    "stderr_tail": "",
                    "verdict": {"status": "RE", "pass": False, "mode": None},
                }
                (run_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
                per_case_results.append(verdict)

            # 平均用（TLE除外は不要: time=None）
            for v in per_case_results:
                ts = v.get("time_sec")
                pm = v.get("peak_memory_kb")
                if ts is not None and v.get("status") != "TLE":
                    sum_time_sec += float(ts)
                    cnt_time += 1
                if pm is not None:
                    sum_peak_kb += int(pm)
                    cnt_peak += 1

            iter_pass_all, passed_cases, total_cases, counts = summarize_iter(iter_dir, per_case_results)
            all_iter_stats.append({
                "iter": iter_dir.name,
                "iter_pass_all": iter_pass_all,
                "passed": passed_cases,
                "total": total_cases,
                "counts": counts,
            })
            continue

        # 3) 各入力を Docker で実行（並列）
        (iter_dir / "runs").mkdir(parents=True, exist_ok=True)

        futures = []
        with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as ex:
            for idx, inp in enumerate(inputs):
                futures.append(ex.submit(execute_case, idx, problem_dir, iter_dir, inp, code_bytes))

            results_by_idx: Dict[int, Dict[str, Any]] = {}
            for fut in as_completed(futures):
                idx, verdict = fut.result()
                results_by_idx[idx] = verdict

        # 入力順に復元
        per_case_results = [results_by_idx[i] for i in range(len(inputs))]

        # 平均用：ケース単位の time/mem を加算（TLE の time は除外）
        for v in per_case_results:
            ts = v.get("time_sec")
            pm = v.get("peak_memory_kb")
            if ts is not None and v.get("status") != "TLE":
                sum_time_sec += float(ts)
                cnt_time += 1
            if pm is not None:
                sum_peak_kb += int(pm)
                cnt_peak += 1

        # 4) iter まとめ
        iter_pass_all, passed_cases, total_cases, counts = summarize_iter(iter_dir, per_case_results)
        all_iter_stats.append({
            "iter": iter_dir.name,
            "iter_pass_all": iter_pass_all,
            "passed": passed_cases,
            "total": total_cases,
            "counts": counts,
        })

    # 5) ベスト解選択と overall
    def iter_key(x):
        try:
            idx = int(x["iter"].split("_")[1])
        except Exception:
            idx = 0
        return (x["iter_pass_all"], x["passed"], -idx)

    best = max(all_iter_stats, key=iter_key) if all_iter_stats else {}

    avg_time_sec = (sum_time_sec / cnt_time) if cnt_time else None
    avg_peak_memory_kb = (sum_peak_kb / cnt_peak) if cnt_peak else None
    avg_code_bytes = (sum_code_bytes / cnt_code_bytes) if cnt_code_bytes else None

    overall = {
        "timestamp_jst": jst_now_iso(),
        "problem_dir": str(problem_dir),
        "tries": all_iter_stats,            # iter_*ごとの集計（pass@k 用の元データ）
        "best": best,
        "docker": {"image": DOCKER_IMAGE, "memory_limit": DOCKER_MEMORY_LIMIT, "cpus": DOCKER_CPUS},
        "gen": {"n_tries": N_TRIES, "model": MODEL_ID, "temp": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS},
        "parallel": {"jobs": PARALLEL_JOBS},
        # 全iter平均（time は TLE 除外、欠損は除外）
        "averages": {
            "time_sec": avg_time_sec,
            "peak_memory_kb": avg_peak_memory_kb,
            "code_bytes": avg_code_bytes,
            "samples": {
                "time": cnt_time,          # time の有効サンプル数（TLE除外）
                "peak_memory": cnt_peak,   # peak mem の有効サンプル数
                "iters": cnt_code_bytes    # code_bytes のサンプル数（=iter数）
            }
        }
    }
    (problem_dir / "outputs" / "overall_summary.json").write_text(
        json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # best_solution.py
    if best:
        best_dir = problem_dir / "outputs" / best["iter"]
        src = best_dir / "extracted.py"
        dst = problem_dir / "outputs" / "best_solution.py"
        if src.exists():
            shutil.copy2(src, dst)
        print(f"[done] best={best.get('iter')}  iter_pass_all={best.get('iter_pass_all')}  passed_cases={best.get('passed')}/{best.get('total')}")
    print(f"[path] overall_summary.json → {problem_dir / 'outputs' / 'overall_summary.json'}")


# 追加：LF固定で書き出し、BOMも除去
def write_input_snapshot(path: Path, txt: str):
    norm = txt.replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(norm)


# ---------- メイン（問題配列を順次処理） ----------
def main():
    if not PROBLEM_DIRS:
        raise RuntimeError("PROBLEM_DIRS が空です。問題フォルダを設定してください。")

    # LLM は 1 回だけロードして使い回す（高速化）
    tok = build_tokenizer()
    model = build_model_4bit_full_gpu()

    for prob in PROBLEM_DIRS:
        process_problem(prob, tok, model)


if __name__ == "__main__":
    main()
