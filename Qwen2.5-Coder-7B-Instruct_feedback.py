# solve_pipeline_feedback_qwen.py
# -*- coding: utf-8 -*-
"""
[ローカル実行・フィードバック付きパイプライン / Qwen2.5-Coder-7B-Instruct / 複数問題順次処理]

完全ローカル（オフライン）で Qwen/Qwen2.5-Coder-7B-Instruct を使い、
「初回生成 + 最大N回のフィードバック改善」を1 iterとし、各 iter の“最後のラウンド”だけを
overall_summary.json（pass@k互換）に記録します。TLE の time は平均から除外します。

前提:
  - モデル一式をローカルへ事前ダウンロード済み（例: D:\models\Qwen\Qwen2.5-Coder-7B-Instruct）
  - Python 3.10+ / CUDA 対応 PyTorch / transformers / accelerate / bitsandbytes
  - Docker イメージ "extracted-exec:py312" が利用可能（実行環境は Python 3.12 相当）

入出力:
  問題フォルダ（例: problems_llama/baseline/ABC407）配下に
    prompt.txt
    inputs/*.txt
    testcases/<name>.out.txt
  を置く。結果は outputs/ 以下に round_*/summary.json, verdict.json 群、
  iter_* 単位のメタや best_solution.py、overall_summary.json を出力。
"""

import os
from pathlib import Path
import re
import json
import math
import shutil
import subprocess
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional, Tuple

import time as pytime
import random as pyrandom
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from concurrent.futures import ThreadPoolExecutor, as_completed

# ========= 完全ローカル設定（必要に応じて編集） ==========================
# 事前ダウンロードしたモデルフォルダ（必ず実在パスを指定）
MODEL_LOCAL_DIR = Path(r"D:\models\Qwen2.5-Coder-7B-Instruct")

# オフライン強制（環境により未設定でもOK）
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", r"D:\hf")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf\transformers")
os.environ.setdefault("HF_HUB_CACHE", r"D:\hf\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf\datasets")
# =====================================================================

# ===== ユーザー設定（問題セットを順番に処理） ===========================
PROBLEM_DIRS: List[Path] = [
    Path("problems_llama/feedback/ABC407"),
    # 追加OK
]

# 生成（LLM）設定
N_TRIES = 1                 # iter（試行）回数
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.7
TOP_P = 0.95
SEED_BASE = 42
MAX_FEEDBACK_ROUNDS = 3      # 追加フィードバック回数（初回 + MAX = 最大 4 ラウンド）

# Docker 実行設定
DOCKER_IMAGE = "extracted-exec:py312"     # 事前に docker build 済み想定
DOCKER_MEMORY_LIMIT = "1g"                # 例: "1g", "2g"
DOCKER_CPUS = None                        # 例: 2（None なら未指定）
RUN_TIMEOUT_SEC = 10
PYTHON_IN_CONTAINER = "python"

# 並列数（ケース並列）
PARALLEL_JOBS = os.cpu_count() or 4

# 採点設定
EPS_ABS = 1e-6
EPS_REL = 1e-6
NORMALIZE_WHITESPACE = True

# 4bit dtype
FOURBIT_COMPUTE_DTYPE = torch.float16

# Qwen 用の system プロンプト
SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Your answer should be a single block of code enclosed in ```python."
    "The program must read from standard input and write to standard output. "
)
# =====================================================================


# ---------- ユーティリティ ----------
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


def judge_one(problem_dir: Path, inp: Path, actual_text: str) -> Dict[str, Any]:
    expected_path = testcase_path_for_input(problem_dir, inp)
    if not expected_path.exists():
        return {"status": "no_expected", "pass": False, "mode": "missing_expected"}
    exp_text = read_text(expected_path)
    if exp_text.lstrip().startswith("regex:"):
        ok = compare_with_regex(exp_text, actual_text)
        return {"status": "AC" if ok else "WA", "pass": bool(ok), "mode": "regex"}
    ok = compare_as_numbers_linewise(exp_text, actual_text)
    return {"status": "AC" if ok else "WA", "pass": bool(ok), "mode": "num_or_str"}


# ---------- LLM 生成 ----------
CODE_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)


def build_tokenizer() -> AutoTokenizer:
    tok = AutoTokenizer.from_pretrained(
        MODEL_LOCAL_DIR,
        use_fast=True,
        trust_remote_code=True,
        local_files_only=True,   # ネットに出ない
    )
    # pad 未設定なら eos を pad に
    if tok.pad_token_id is None:
        if getattr(tok, "eos_token", None) is not None:
            tok.pad_token = tok.eos_token
        else:
            tok.add_special_tokens({"pad_token": "</s>"})
    return tok


def build_model_4bit_full_gpu() -> AutoModelForCausalLM:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDAが無効です。GPU/ドライバ/Torch を確認してください。")
    qconf = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=FOURBIT_COMPUTE_DTYPE,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR,
        trust_remote_code=True,
        quantization_config=qconf,
        device_map={'': 0},
        attn_implementation="sdpa",   # Windows+4bit で安定しやすい
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    return model


def set_seed_for_iter(i: int):
    torch.manual_seed(SEED_BASE + i)
    torch.cuda.manual_seed_all(SEED_BASE + i)
    pyrandom.seed(SEED_BASE + i)


def get_safe_eos_id(tok: AutoTokenizer) -> int:
    # ★CUDA device assert 予防：単一 int を使う
    if isinstance(getattr(tok, "eos_token_id", None), int) and tok.eos_token_id >= 0:
        return tok.eos_token_id
    for t in ["<|endoftext|>", "</s>"]:
        tid = tok.convert_tokens_to_ids(t)
        if isinstance(tid, int) and tid >= 0:
            return tid
    if isinstance(tok.pad_token_id, int) and tok.pad_token_id >= 0:
        return tok.pad_token_id
    raise RuntimeError("Valid eos_token_id not found.")


def apply_messages(tok: AutoTokenizer, messages: List[Dict[str, str]]) -> str:
    # Qwen2.5 は chat_template を持つ。失敗時は素朴フォールバック。
    try:
        return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    except Exception:
        merged = []
        for m in messages:
            if m["role"] == "system":
                merged.append(f"<<SYS>>\n{m['content']}\n<</SYS>>")
            elif m["role"] == "user":
                merged.append(f"User: {m['content']}")
            else:
                merged.append(f"Assistant: {m['content']}")
        merged.append("Assistant:")
        return "\n\n".join(merged)


def generate_with_messages(tok: AutoTokenizer, model: AutoModelForCausalLM, messages: List[Dict[str, str]]) -> str:
    prompt_text = apply_messages(tok, messages)
    enc = tok(prompt_text, return_tensors="pt")
    if enc["input_ids"].numel() == 0:
        raise RuntimeError("Tokenized prompt is empty.")
    inputs = {k: v.to(model.device) for k, v in enc.items()}

    eos_id = get_safe_eos_id(tok)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else eos_id

    with torch.inference_mode():
        gen_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            pad_token_id=pad_id,
            eos_token_id=eos_id,  # 単一 int
        )
    gen_only = gen_ids[0, inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_only, skip_special_tokens=False)
    for tkn in ("<|eot_id|>", "<|eom_id|>"):
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


def write_round_files(round_dir: Path, response_text: str, code: Optional[str], meta: Dict[str, Any]):
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "response.txt").write_text(response_text, encoding="utf-8")
    if code is not None:
        (round_dir / "extracted.py").write_text(code, encoding="utf-8", newline="\n")
    else:
        (round_dir / "extracted.py").write_text("# extraction failed\n", encoding="utf-8", newline="\n")
    (round_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


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
    - GNU time -v / time -p をフォールバック
    """
    out = {"wall_time_sec": None, "max_rss_kb": None, "user_time_sec": None, "sys_time_sec": None}
    if not path.exists():
        return out
    txt = path.read_text(encoding="utf-8", errors="ignore")

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

    # GNU time -v
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

    # time -p
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


def run_in_docker(round_dir: Path, run_dir: Path, input_path: Path, timeout_sec: int) -> Tuple[int, str]:
    """
    round_dir を /work にマウントして、extracted.py を実行。
    - stdin: runs/<case>/input.txt を < リダイレクトで供給
    - argv[1]: "input.txt" を渡す
    - 環境変数: INPUT_FILE, INPUT_BASENAME
    - time の出力 → runs/<case>/time.txt（elapsed_msec/elapsed_sec/maxrss 等）
    - 戻り値: (exit_code, docker_stderr_tail)
    """
    abs_round = str(round_dir.resolve())
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
        '  __ELAPSED_MS=$(( (__END_NS - __START_NS) / 1000000 )); '
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
        "-v", f"{abs_round}:/work",
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
def summarize_round(round_dir: Path, results: List[Dict[str, Any]]):
    counts = {"AC": 0, "WA": 0, "TLE": 0, "RE": 0, "MLE": 0, "NOEXP": 0}
    passed_cases = 0
    for r in results:
        st = r.get("status")
        if st in counts:
            counts[st] += 1
        if r.get("verdict", {}).get("pass"):
            passed_cases += 1
    total_cases = len(results)
    pass_all = (counts["AC"] == total_cases and total_cases > 0)

    summary = {
        "timestamp_jst": jst_now_iso(),
        "round_dir": round_dir.name,
        "total_cases": total_cases,
        "passed": passed_cases,
        "failed": total_cases - passed_cases,
        "counts": counts,
        "pass_all": pass_all,
        "cases": results,
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return pass_all, passed_cases, total_cases, counts


# ---------- ケース実行ワーカー ----------
def execute_case(index: int, problem_dir: Path, round_dir: Path, inp: Path, code_bytes: int) -> Tuple[int, Dict[str, Any]]:
    run_dir = round_dir / "runs" / inp.stem
    run_dir.mkdir(parents=True, exist_ok=True)

    # 入力スナップショット（LF固定・BOM除去）
    inp_text = read_text(inp).replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    (run_dir / "input.txt").write_text(inp_text, encoding="utf-8", newline="\n")

    # 実行（Docker）
    rc, docker_err_tail = run_in_docker(round_dir, run_dir, inp, RUN_TIMEOUT_SEC)

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

    if rc == 124:
        status = "TLE"
        detail = "timeout"
    elif rc == 137:
        status = "MLE"
        detail = "oom_killed"
    elif rc == 0:
        j = judge_one(problem_dir, inp, stdout_txt)
        status = "AC" if j["pass"] else "WA"
        if not j["pass"] and j.get("mode") != "regex":
            mm = first_mismatch(read_text(testcase_path_for_input(problem_dir, inp)) if testcase_path_for_input(problem_dir, inp).exists() else "", stdout_txt)
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


# ---------- フィードバック文（英語） ----------
def build_feedback_message(total_cases: int, passed: int, failed: int, counts: Dict[str, int], round_idx: int) -> str:
    return (
        "Your previous Python solution did not pass all tests.\n"
        f"Test summary (round {round_idx}): total_cases={total_cases}, passed={passed}, failed={failed}, "
        f"counts={{AC:{counts.get('AC',0)}, WA:{counts.get('WA',0)}, TLE:{counts.get('TLE',0)}, "
        f"RE:{counts.get('RE',0)}, MLE:{counts.get('MLE',0)}}}.\n"
        "Please fix the bugs, handle edge cases, and ensure the algorithm is correct and efficient. "
        "Keep the exact same I/O format: read from STDIN and write to STDOUT. "
        "Return only the final Python program enclosed in a ```python code block, with no explanations."
    )


# ---------- 1問題フォルダ処理 ----------
def process_problem(problem_dir: Path, tok: AutoTokenizer, model: AutoModelForCausalLM):
    print(f"\n===== Processing problem: {problem_dir} =====")
    ensure_dirs(problem_dir)
    if not docker_available():
        raise RuntimeError("docker が見つかりません。Docker Desktop などを起動し、パスが通っているか確認してください。")

    inputs = list_inputs(problem_dir)
    if not inputs:
        raise RuntimeError(f"inputs/*.txt が見つかりません: {problem_dir / 'inputs'}")

    initial_user_prompt = load_prompt(problem_dir)

    tries_for_overall: List[Dict[str, Any]] = []

    # 全iter平均用（最終ラウンドのみ集計、time は TLE 除外）
    sum_time_sec = 0.0
    cnt_time = 0
    sum_peak_kb = 0
    cnt_peak = 0
    sum_code_bytes = 0
    cnt_code_bytes = 0

    for i in range(1, N_TRIES + 1):
        set_seed_for_iter(i)
        iter_dir = problem_dir / "outputs" / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"[gen] iter {i}/{N_TRIES} (feedback up to {MAX_FEEDBACK_ROUNDS} rounds)…")

        # ---- 同一会話で継続 ----
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_prompt},
        ]

        final_round_idx = 0
        final_per_case_results: List[Dict[str, Any]] = []
        final_counts: Dict[str, int] = {}
        final_passed = 0
        final_total = 0
        final_pass_all = False
        final_code_bytes = 0

        for r in range(0, MAX_FEEDBACK_ROUNDS + 1):
            round_dir = iter_dir / f"round_{r:03d}"
            print(f"  [round {r}] generating…")
            t0 = pytime.perf_counter()
            response_text = generate_with_messages(tok, model, messages)
            gen_dt = round(pytime.perf_counter() - t0, 3)

            code = extract_python_code(response_text)
            code_bytes = len(code.encode("utf-8")) if code is not None else 0

            meta = {
                "timestamp_jst": jst_now_iso(),
                "iteration": i,
                "round": r,
                "model_local_dir": str(MODEL_LOCAL_DIR),
                "quantization": "bnb_4bit_nf4",
                "compute_dtype": str(FOURBIT_COMPUTE_DTYPE).replace("torch.", ""),
                "gen_params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P},
                "gen_time_sec": gen_dt,
                "extraction": {"found": code is not None, "code_bytes": code_bytes},
            }
            write_round_files(round_dir, response_text, code, meta)

            # assistant 応答を会話に追加
            messages.append({"role": "assistant", "content": response_text})

            # ---- テスト実行 ----
            per_case_results: List[Dict[str, Any]] = []
            if code is None:
                # 抽出失敗 → 全ケース RE
                for inp in inputs:
                    run_dir = round_dir / "runs" / inp.stem
                    run_dir.mkdir(parents=True, exist_ok=True)
                    (run_dir / "input.txt").write_text(read_text(inp), encoding="utf-8")
                    expected_name = testcase_path_for_input(problem_dir, inp).name if testcase_path_for_input(problem_dir, inp).exists() else None
                    verdict = {
                        "input_file": inp.name,
                        "expected_file": expected_name,
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
            else:
                # 並列実行
                (round_dir / "runs").mkdir(parents=True, exist_ok=True)
                futures = []
                with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as ex:
                    for idx, inp in enumerate(inputs):
                        futures.append(ex.submit(execute_case, idx, problem_dir, round_dir, inp, code_bytes))
                    results_by_idx: Dict[int, Dict[str, Any]] = {}
                    for fut in as_completed(futures):
                        idx, verdict = fut.result()
                        results_by_idx[idx] = verdict
                per_case_results = [results_by_idx[idx] for idx in range(len(inputs))]

            # ---- ラウンドまとめ ----
            pass_all, passed_cases, total_cases, counts = summarize_round(round_dir, per_case_results)

            # “最後のラウンド”情報を更新（途中でも上書き）
            final_round_idx = r
            final_per_case_results = per_case_results
            final_counts = counts
            final_passed = passed_cases
            final_total = total_cases
            final_pass_all = pass_all
            final_code_bytes = code_bytes

            if pass_all:
                print(f"  [round {r}] all tests passed ✔")
                break

            if r < MAX_FEEDBACK_ROUNDS:
                fb_msg = build_feedback_message(total_cases, passed_cases, total_cases - passed_cases, counts, r)
                messages.append({"role": "user", "content": fb_msg})
                print(f"  [round {r}] feedback queued, continue…")
            else:
                print(f"  [round {r}] reached max feedback rounds ✖")

        # ---- iter の“最後のラウンド”結果を overall 用に記録 ----
        tries_for_overall.append({
            "iter": iter_dir.name,
            "iter_pass_all": final_pass_all,
            "passed": final_passed,
            "total": final_total,
            "counts": final_counts,
            "rounds_used": final_round_idx + 1,
        })

        # ---- 平均用（最終ラウンドのみ加算。time は TLE 除外）----
        sum_code_bytes += final_code_bytes
        cnt_code_bytes += 1
        for v in final_per_case_results:
            ts = v.get("time_sec")
            pm = v.get("peak_memory_kb")
            if ts is not None and v.get("status") != "TLE":
                sum_time_sec += float(ts)
                cnt_time += 1
            if pm is not None:
                sum_peak_kb += int(pm)
                cnt_peak += 1

    # ---- ベスト選択（最後のラウンド結果に基づく）----
    def iter_key(x):
        # 優先: 全AC → passed 多 → rounds 少 → 生成順（新しい優先）
        try:
            idx = int(x["iter"].split("_")[1])
        except Exception:
            idx = 0
        return (x["iter_pass_all"], x["passed"], -(x.get("rounds_used", 0)), -idx)

    best = max(tries_for_overall, key=iter_key) if tries_for_overall else {}

    # best_solution.py は best iter の「最後の round_*」からコピー
    if best:
        best_iter_dir = problem_dir / "outputs" / best["iter"]
        rounds = sorted([p for p in best_iter_dir.glob("round_*") if p.is_dir()])
        if rounds:
            best_round_dir = rounds[-1]
            src = best_round_dir / "extracted.py"
            dst = problem_dir / "outputs" / "best_solution.py"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[best] {best['iter']} ({best_round_dir.name}) → {dst}")

    # ---- overall_summary.json（pass@k互換）----
    avg_time_sec = (sum_time_sec / cnt_time) if cnt_time else None
    avg_peak_memory_kb = (sum_peak_kb / cnt_peak) if cnt_peak else None
    avg_code_bytes = (sum_code_bytes / cnt_code_bytes) if cnt_code_bytes else None

    overall = {
        "timestamp_jst": jst_now_iso(),
        "problem_dir": str(problem_dir),
        "tries": tries_for_overall,   # 各 iter の「最後のラウンドのみ」
        "best": best,
        "docker": {"image": DOCKER_IMAGE, "memory_limit": DOCKER_MEMORY_LIMIT, "cpus": DOCKER_CPUS},
        "gen": {
            "n_tries": N_TRIES,
            "model_local_dir": str(MODEL_LOCAL_DIR),
            "temp": TEMPERATURE,
            "top_p": TOP_P,
            "max_new_tokens": MAX_NEW_TOKENS,
            "feedback_rounds_max": MAX_FEEDBACK_ROUNDS,
        },
        "parallel": {"jobs": PARALLEL_JOBS},
        "averages": {
            "time_sec": avg_time_sec,
            "peak_memory_kb": avg_peak_memory_kb,
            "code_bytes": avg_code_bytes,
            "samples": {"time": cnt_time, "peak_memory": cnt_peak, "iters": cnt_code_bytes}
        }
    }
    out_json = problem_dir / "outputs" / "overall_summary.json"
    out_json.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] tries={len(tries_for_overall)}  best={best.get('iter')}")
    print(f"[path] overall_summary.json → {out_json}")


# ---------- メイン（問題配列を順次処理） ----------
def main():
    if not MODEL_LOCAL_DIR.exists():
        raise FileNotFoundError(f"MODEL_LOCAL_DIR が存在しません: {MODEL_LOCAL_DIR}")
    if not PROBLEM_DIRS:
        raise RuntimeError("PROBLEM_DIRS が空です。問題フォルダを設定してください。")

    tok = build_tokenizer()
    model = build_model_4bit_full_gpu()

    for prob in PROBLEM_DIRS:
        process_problem(prob, tok, model)


if __name__ == "__main__":
    main()
