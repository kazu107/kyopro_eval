# solve_pipeline_feedback_cpp.py
# -*- coding: utf-8 -*-
"""
[フィードバック版パイプライン(C++) / 複数問題順次処理対応]

1 iter = 「初回生成 + 最大 MAX_FEEDBACK_ROUNDS 回の改善」
各ラウンドで C++ コード生成 → Dockerビルド(g++) → 全テスト並列実行 → 採点。
全ACで打ち切り。overall_summary.json には各 iter の「最後のラウンド」の集計のみを保存。
best_solution.cpp はベストiterの「最後のラウンド」のソースを保存。
"""

import os
os.environ.setdefault("HF_HOME", r"D:\hf")
os.environ.setdefault("TRANSFORMERS_CACHE", r"D:\hf\transformers")
os.environ.setdefault("HF_HUB_CACHE", r"D:\hf\hub")
os.environ.setdefault("HF_DATASETS_CACHE", r"D:\hf\datasets")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

MODEL_LOCAL_DIR = r"D:\hf_models\llama-3.1-8b-instruct"

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

from concurrent.futures import ThreadPoolExecutor, as_completed

# ===== ユーザー設定 ==================================================
PROBLEM_DIRS: List[Path] = [
    Path("problems_llama_cpp/feedback/ABC408"),
    Path("problems_llama_cpp/feedback/ABC407"),
    Path("problems_llama_cpp/feedback/ABC406"),
    Path("problems_llama_cpp/feedback/ABC405"),
    Path("problems_llama_cpp/feedback/ABC404"),
    Path("problems_llama_cpp/feedback/ABC403"),
    Path("problems_llama_cpp/feedback/ABC402"),
    Path("problems_llama_cpp/feedback/ABC401"),
    Path("problems_llama_cpp/feedback/ABC400"),
    Path("problems_llama_cpp/feedback/ABC399"),
    Path("problems_llama_cpp/feedback/ABC398"),
    Path("problems_llama_cpp/feedback/ABC397"),
    Path("problems_llama_cpp/feedback/ABC396"),
    Path("problems_llama_cpp/feedback/ABC395"),
    Path("problems_llama_cpp/feedback/ABC394"),
    Path("problems_llama_cpp/feedback/ABC393"),
    Path("problems_llama_cpp/feedback/ABC392"),
]

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

N_TRIES = 100
MAX_NEW_TOKENS = 800
TEMPERATURE = 0.7
TOP_P = 0.95
SEED_BASE = 42
MAX_FEEDBACK_ROUNDS = 3

DOCKER_IMAGE = "extracted-exec:cpp"
DOCKER_MEMORY_LIMIT = "1g"
DOCKER_CPUS = 1
RUN_TIMEOUT_SEC = 10

CPP_STD = "gnu++20"
CPP_COMPILE_FLAGS = "-O2 -pipe -s"
CPP_EXE_REL = "build/main"
CPP_SOURCE_NAME = "extracted.cpp"

PARALLEL_JOBS = os.cpu_count() or 4

EPS_ABS = 1e-6
EPS_REL = 1e-6
NORMALIZE_WHITESPACE = True

HF_TOKEN = os.environ.get("HF_TOKEN", "")
FOURBIT_COMPUTE_DTYPE = torch.float16

SYSTEM_PROMPT = (
    "You are an expert competitive programmer. "
    "Return ONLY a single C++17/20 program enclosed in ```cpp code fences. "
    "The program must read from STDIN and write to STDOUT. No explanations."
)
# ====================================================================

# ---------- 共通ユーティリティ ----------
CODE_FENCE_RE = re.compile(r"```([^\n]*)\n(.*?)```", re.DOTALL)

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
    return problem_dir / "testcases" / f"{inp.stem}.out.txt"

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def normalize_text(s: str) -> str:
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = "\n".join(line.rstrip() for line in s.split("\n"))
    if NORMALIZE_WHITESPACE:
        s = "\n".join(" ".join(line.split()) for line in s.split("\n"))
    if s and not s.endswith("\n"):
        s += "\n"
    return s

def is_float_token(x: str) -> bool:
    try:
        float(x); return True
    except Exception:
        return False

def compare_as_numbers_linewise(expected: str, actual: str) -> bool:
    elines = normalize_text(expected).splitlines()
    alines = normalize_text(actual).splitlines()
    if len(elines) != len(alines):
        return False
    for e, a in zip(elines, alines):
        et, at = e.split(), a.split()
        if len(et) != len(at): return False
        if all(is_float_token(t) for t in et) and all(is_float_token(t) for t in at):
            for u, v in zip(et, at):
                if not math.isclose(float(u), float(v), rel_tol=EPS_REL, abs_tol=EPS_ABS):
                    return False
        else:
            if e != a: return False
    return True

def compare_with_regex(expected: str, actual: str) -> bool:
    pat = expected.strip().split("\n", 1)[0][len("regex:"):].strip()
    try:
        return re.fullmatch(pat, actual, re.MULTILINE | re.DOTALL) is not None
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
                    pos = j; break
            return {"line": i+1, "col": pos+1, "expected_line": e[:200], "actual_line": a[:200]}
    return None

def judge_one(problem_dir: Path, expected_from_inp: Path, actual_text: str) -> Dict[str, Any]:
    expected_path = testcase_path_for_input(problem_dir, expected_from_inp)
    if not expected_path.exists():
        return {"status":"no_expected","pass":False,"mode":"missing_expected"}
    exp_text = read_text(expected_path)
    if exp_text.lstrip().startswith("regex:"):
        ok = compare_with_regex(exp_text, actual_text)
        return {"status":"AC" if ok else "WA", "pass":bool(ok), "mode":"regex"}
    ok = compare_as_numbers_linewise(exp_text, actual_text)
    return {"status":"AC" if ok else "WA", "pass":bool(ok), "mode":"num_or_str"}

# ---------- LLM ----------
def build_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_LOCAL_DIR, use_fast=False, trust_remote_code=True, local_files_only=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    return tok

def build_model_4bit_full_gpu():
    if not torch.cuda.is_available(): raise RuntimeError("CUDAが無効です。")
    qconf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_LOCAL_DIR, trust_remote_code=True, quantization_config=qconf, device_map={'':0},
        attn_implementation="eager", local_files_only=True, low_cpu_mem_usage=True)
    model.eval(); return model

def set_seed_for_iter(i: int):
    torch.manual_seed(SEED_BASE+i); torch.cuda.manual_seed_all(SEED_BASE+i); pyrandom.seed(SEED_BASE+i)

def get_terminators(tok: AutoTokenizer) -> List[int]:
    ids=[tok.eos_token_id]; eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if isinstance(eot,int) and eot != tok.eos_token_id: ids.append(eot)
    return ids

def apply_messages(tok: AutoTokenizer, messages: List[Dict[str,str]]) -> str:
    return tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

def generate_with_messages(tok: AutoTokenizer, model: AutoModelForCausalLM, messages: List[Dict[str,str]]) -> str:
    prompt_text = apply_messages(tok, messages)
    inputs = tok(prompt_text, return_tensors="pt", padding=True)
    inputs = {k:v.to(model.device) for k,v in inputs.items()}
    with torch.no_grad():
        gen_ids = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=True,
            temperature=TEMPERATURE, top_p=TOP_P, pad_token_id=tok.eos_token_id, eos_token_id=get_terminators(tok))
    gen_only = gen_ids[0][inputs["input_ids"].shape[1]:]
    text = tok.decode(gen_only, skip_special_tokens=False)
    for tkn in ("<|eot_id|>", "<|eom_id|>"): text = text.replace(tkn,"")
    return text.strip()

def extract_cpp_code(text: str) -> Optional[str]:
    prefs = {"cpp","c++","cc","cxx","cpp17","cpp20"}
    cands = []
    for m in CODE_FENCE_RE.finditer(text):
        lang = (m.group(1) or "").strip().lower(); body = m.group(2); cands.append((lang, body))
    for lang, body in cands:
        if lang in prefs: return body.strip()
    for lang, body in cands:
        if lang == "": return body.strip()
    return None

def write_round_files(round_dir: Path, response_text: str, code: Optional[str], meta: Dict[str, Any]):
    round_dir.mkdir(parents=True, exist_ok=True)
    (round_dir / "response.txt").write_text(response_text, encoding="utf-8")
    if code is not None:
        (round_dir / CPP_SOURCE_NAME).write_text(code, encoding="utf-8")
    else:
        (round_dir / CPP_SOURCE_NAME).write_text("// extraction failed\n", encoding="utf-8")
    (round_dir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

# ---------- Docker ----------
def docker_available() -> bool:
    try:
        subprocess.run(["docker","version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def parse_time_file(path: Path) -> Dict[str, Optional[float]]:
    out = {"wall_time_sec":None,"max_rss_kb":None,"user_time_sec":None,"sys_time_sec":None}
    if not path.exists(): return out
    txt = path.read_text(encoding="utf-8", errors="ignore")
    kv={}
    for line in txt.splitlines():
        if "=" in line:
            k,v=line.split("=",1); kv[k.strip()]=v.strip()
    def f2(x): 
        try: return float(x)
        except Exception: return None
    def i2(x):
        try: return int(x)
        except Exception: return None
    if kv:
        if "elapsed_msec" in kv:
            try: out["wall_time_sec"]=int(kv["elapsed_msec"])/1000.0
            except Exception: pass
        if out["wall_time_sec"] is None and "elapsed_sec" in kv:
            out["wall_time_sec"]=f2(kv.get("elapsed_sec"))
        out["user_time_sec"]=f2(kv.get("user_sec"))
        out["sys_time_sec"]=f2(kv.get("sys_sec"))
        out["max_rss_kb"]=i2(kv.get("maxrss_kb"))
        if any(v is not None for v in out.values()): return out

    any_hit=False
    def to_seconds(s: str):
        s=s.strip(); parts=s.split(":")
        try:
            if len(parts)==3: h,m,sec=int(parts[0]),int(parts[1]),float(parts[2]); return h*3600+m*60+sec
            if len(parts)==2: m,sec=int(parts[0]),float(parts[1]); return m*60+sec
            return float(s)
        except Exception: return None
    for line in txt.splitlines():
        if "Elapsed (wall clock) time" in line:
            out["wall_time_sec"]=to_seconds(line.split(":",1)[-1]); any_hit=True
        elif "Maximum resident set size" in line:
            try: out["max_rss_kb"]=int(line.split(":")[-1].strip().split()[0]); any_hit=True
            except Exception: pass
        elif "User time (seconds)" in line:
            try: out["user_time_sec"]=float(line.split(":")[-1].strip().split()[0]); any_hit=True
            except Exception: pass
        elif "System time (seconds)" in line:
            try: out["sys_time_sec"]=float(line.split(":")[-1].strip().split()[0]); any_hit=True
            except Exception: pass
    if any_hit: return out

    real=user=sys=None
    for line in txt.splitlines():
        if line.startswith("real"):
            try: real=float(line.split()[1])
            except Exception: pass
        elif line.startswith("user"):
            try: user=float(line.split()[1])
            except Exception: pass
        elif line.startswith("sys"):
            try: sys=float(line.split()[1])
            except Exception: pass
    if (real is not None) or (user is not None) or (sys is not None):
        out["wall_time_sec"]=real; out["user_time_sec"]=user; out["sys_time_sec"]=sys
    return out

def compile_cpp_in_docker(work_dir: Path) -> Tuple[int, str]:
    abs_dir = str(work_dir.resolve())
    inner = (
        f'mkdir -p "$(dirname {CPP_EXE_REL})" && '
        f'g++ -std={CPP_STD} {CPP_COMPILE_FLAGS} -o "{CPP_EXE_REL}" "{CPP_SOURCE_NAME}" '
        f'2> "build/compile_stderr.txt"'
    )
    cmd = ["docker","run","--rm","-v",f"{abs_dir}:/work","-w","/work", DOCKER_IMAGE, "/bin/bash","-lc", inner]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode, (proc.stderr or "")[-1000:]
    except Exception as e:
        (work_dir / "build").mkdir(parents=True, exist_ok=True)
        (work_dir / "build" / "compile_stderr.txt").write_text(f"[runner] docker build failed: {e}\n", encoding="utf-8")
        return -1, str(e)

def run_in_docker(work_dir: Path, run_dir: Path, input_path: Path, timeout_sec: int) -> Tuple[int, str]:
    abs_dir = str(work_dir.resolve())
    case = input_path.stem
    in_container_case_dir = f"runs/{case}"
    exe = f"/work/{CPP_EXE_REL}"

    inner = (
        'if [ -x /usr/bin/time ]; then '
        '  __START_NS=$(date +%s%N); '
        f'  timeout -k 1s {timeout_sec}s '
        f'  /usr/bin/time -f "maxrss_kb=%M\nuser_sec=%U\nsys_sec=%S" '
        f'  -o "{in_container_case_dir}/time_extra.txt" '
        f'  "{exe}" '
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
        f'  {{ time -p "{exe}" '
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
    cmd = ["docker","run","--rm","-v",f"{abs_dir}:/work","-w","/work",
           "--memory",DOCKER_MEMORY_LIMIT,"--memory-swap",DOCKER_MEMORY_LIMIT]
    if DOCKER_CPUS: cmd += ["--cpus", str(DOCKER_CPUS)]
    cmd += [DOCKER_IMAGE, "/bin/bash","-lc", inner]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return proc.returncode, (proc.stderr or "")[-1000:]
    except Exception as e:
        (run_dir / "prog_stderr.txt").write_text(f"[runner] docker run failed: {e}\n", encoding="utf-8")
        return -1, str(e)

# ---------- 集計 ----------
def summarize_round(round_dir: Path, results: List[Dict[str, Any]]):
    counts = {"AC":0,"WA":0,"TLE":0,"RE":0,"MLE":0,"NOEXP":0}
    passed_cases = 0
    for r in results:
        st = r.get("status")
        if st in counts: counts[st]+=1
        if r.get("verdict", {}).get("pass"): passed_cases += 1
    total = len(results)
    pass_all = (counts["AC"] == total and total>0)
    summary = {
        "timestamp_jst": jst_now_iso(),
        "round_dir": round_dir.name,
        "total_cases": total,
        "passed": passed_cases,
        "failed": total - passed_cases,
        "counts": counts,
        "pass_all": pass_all,
        "cases": results,
    }
    (round_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return pass_all, passed_cases, total, counts

# ---------- 1ケース実行 ----------
def execute_case(index: int, problem_dir: Path, round_dir: Path, inp: Path, code_bytes: int) -> Tuple[int, Dict[str, Any]]:
    run_dir = round_dir / "runs" / inp.stem
    run_dir.mkdir(parents=True, exist_ok=True)
    write_input_snapshot(run_dir / "input.txt", read_text(inp))
    rc, docker_err_tail = run_in_docker(round_dir, run_dir, inp, RUN_TIMEOUT_SEC)

    time_info = parse_time_file(run_dir / "time.txt")
    wall = time_info.get("wall_time_sec"); maxrss = time_info.get("max_rss_kb")
    stdout_txt = (run_dir / "stdout.txt").read_text(encoding="utf-8", errors="ignore") if (run_dir / "stdout.txt").exists() else ""
    stderr_prog = (run_dir / "prog_stderr.txt").read_text(encoding="utf-8", errors="ignore") if (run_dir / "prog_stderr.txt").exists() else ""
    stderr_tail = (stderr_prog or docker_err_tail)[-2000:]

    status="RE"; detail=None; verdict_core={"status":"RE","pass":False,"mode":None}
    if rc==124: status="TLE"; detail="timeout"
    elif rc==137: status="MLE"; detail="oom_killed"
    elif rc==0:
        j = judge_one(problem_dir, inp, stdout_txt)
        status = "AC" if j["pass"] else "WA"
        if not j["pass"] and j.get("mode")!="regex":
            mm = first_mismatch(read_text(testcase_path_for_input(problem_dir, inp)) if testcase_path_for_input(problem_dir, inp).exists() else "", stdout_txt)
            detail = {"mismatch": mm} if mm else None
        verdict_core = j
    else:
        status="RE"; detail=f"nonzero_exit({rc})"

    expected_file_name = testcase_path_for_input(problem_dir, inp).name if testcase_path_for_input(problem_dir, inp).exists() else None
    verdict = {
        "input_file": inp.name, "expected_file": expected_file_name,
        "returncode": rc, "status": status, "detail": detail,
        "time_sec": wall, "peak_memory_kb": maxrss, "code_bytes": code_bytes,
        "stderr_tail": stderr_tail, "verdict": verdict_core,
    }
    (run_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    return index, verdict

# ---------- FBメッセージ ----------
def build_feedback_message(total_cases: int, passed: int, failed: int, counts: Dict[str,int], round_idx: int) -> str:
    return (
        "Your previous C++ solution did not pass all tests.\n"
        f"Test summary (round {round_idx}): total_cases={total_cases}, passed={passed}, failed={failed}, "
        f"counts={{AC:{counts.get('AC',0)}, WA:{counts.get('WA',0)}, TLE:{counts.get('TLE',0)}, RE:{counts.get('RE',0)}, MLE:{counts.get('MLE',0)}}}.\n"
        "Fix bugs and edge cases. Keep exactly the same I/O (STDIN/STDOUT). "
        "Return only the final C++ program enclosed in a ```cpp code block, with no explanations."
    )

# ---------- ラン ----------
def write_input_snapshot(path: Path, txt: str):
    norm = txt.replace("\r\n","\n").replace("\r","\n").lstrip("\ufeff")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(norm)

def process_problem(problem_dir: Path, tok: AutoTokenizer, model: AutoModelForCausalLM):
    print(f"\n===== Processing problem: {problem_dir} =====")
    ensure_dirs(problem_dir)
    if not docker_available():
        raise RuntimeError("docker が見つかりません。Docker Desktop を起動し、パスを確認してください。")
    inputs = list_inputs(problem_dir)
    if not inputs:
        raise RuntimeError(f"inputs/*.txt が見つかりません: {problem_dir / 'inputs'}")

    initial_user_prompt = load_prompt(problem_dir)
    tries_for_overall: List[Dict[str, Any]] = []

    sum_time_sec=0.0; cnt_time=0
    sum_peak_kb=0;   cnt_peak=0
    sum_code_bytes=0; cnt_code_bytes=0

    for i in range(1, N_TRIES+1):
        set_seed_for_iter(i)
        iter_dir = problem_dir / "outputs" / f"iter_{i:03d}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        print(f"[gen] iter {i}/{N_TRIES} (feedback up to {MAX_FEEDBACK_ROUNDS}) …")

        messages: List[Dict[str,str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": initial_user_prompt},
        ]

        final_round_idx=0
        final_per_case_results: List[Dict[str,Any]]=[]
        final_counts: Dict[str,int]={}
        final_passed=0
        final_total=0
        final_pass_all=False
        final_code_bytes=0

        for r in range(0, MAX_FEEDBACK_ROUNDS+1):
            round_dir = iter_dir / f"round_{r:03d}"
            print(f"  [round {r}] generating…")
            t0 = pytime.perf_counter()
            response_text = generate_with_messages(tok, model, messages)
            gen_dt = round(pytime.perf_counter()-t0, 3)

            code = extract_cpp_code(response_text)
            code_bytes = len(code.encode("utf-8")) if code is not None else 0
            meta = {
                "timestamp_jst": jst_now_iso(), "iteration": i, "round": r,
                "model": MODEL_ID, "quantization": "bnb_4bit_nf4",
                "compute_dtype": str(FOURBIT_COMPUTE_DTYPE).replace("torch.",""),
                "gen_params": {"max_new_tokens": MAX_NEW_TOKENS, "temperature": TEMPERATURE, "top_p": TOP_P},
                "gen_time_sec": gen_dt, "extraction": {"found": code is not None, "code_bytes": code_bytes},
            }
            write_round_files(round_dir, response_text, code, meta)
            messages.append({"role":"assistant","content":response_text})

            per_case_results: List[Dict[str,Any]] = []

            if code is None:
                # 抽出失敗 → 全ケース RE
                for inp in inputs:
                    run_dir = round_dir / "runs" / inp.stem
                    run_dir.mkdir(parents=True, exist_ok=True)
                    verdict = {
                        "input_file": inp.name, "expected_file": testcase_path_for_input(problem_dir, inp).name if testcase_path_for_input(problem_dir, inp).exists() else None,
                        "returncode": -2, "status":"RE", "detail":"code_extraction_failed",
                        "time_sec": None, "peak_memory_kb": None, "code_bytes": 0, "stderr_tail": "",
                        "verdict": {"status":"RE","pass":False,"mode":None},
                    }
                    (run_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
                    per_case_results.append(verdict)
            else:
                # ビルド
                rc_build, _ = compile_cpp_in_docker(round_dir)
                if rc_build != 0 or not (round_dir / CPP_EXE_REL).exists():
                    compile_stderr = (round_dir / "build" / "compile_stderr.txt").read_text(encoding="utf-8", errors="ignore") if (round_dir / "build" / "compile_stderr.txt").exists() else ""
                    for inp in inputs:
                        run_dir = round_dir / "runs" / inp.stem
                        run_dir.mkdir(parents=True, exist_ok=True)
                        verdict = {
                            "input_file": inp.name, "expected_file": testcase_path_for_input(problem_dir, inp).name if testcase_path_for_input(problem_dir, inp).exists() else None,
                            "returncode": -3, "status":"RE", "detail":{"compile_failed": True},
                            "time_sec": None, "peak_memory_kb": None, "code_bytes": code_bytes,
                            "stderr_tail": compile_stderr[-2000:], "verdict": {"status":"RE","pass":False,"mode":None},
                        }
                        (run_dir / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
                        per_case_results.append(verdict)
                else:
                    # 実行
                    (round_dir / "runs").mkdir(parents=True, exist_ok=True)
                    futures=[]
                    with ThreadPoolExecutor(max_workers=PARALLEL_JOBS) as ex:
                        for idx, inp in enumerate(inputs):
                            futures.append(ex.submit(execute_case, idx, problem_dir, round_dir, inp, code_bytes))
                        results_by_idx: Dict[int, Dict[str,Any]] = {}
                        for fut in as_completed(futures):
                            idx, verdict = fut.result()
                            results_by_idx[idx] = verdict
                    per_case_results = [results_by_idx[idx] for idx in range(len(inputs))]

            pass_all, passed_cases, total_cases, counts = summarize_round(round_dir, per_case_results)

            # 最終（暫定）値を更新
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
                fb_msg = build_feedback_message(total_cases, passed_cases, total_cases-passed_cases, counts, r)
                messages.append({"role":"user","content": fb_msg})
                print(f"  [round {r}] feedback queued, continue…")
            else:
                print(f"  [round {r}] reached max feedback rounds ✖")

        # overall 用（最後のラウンドのみ）
        tries_for_overall.append({
            "iter": iter_dir.name,
            "iter_pass_all": final_pass_all,
            "passed": final_passed,
            "total": final_total,
            "counts": final_counts,
            "rounds_used": final_round_idx + 1
        })

        # 平均（最後ラウンド, TLE除外）
        sum_code_bytes += final_code_bytes; cnt_code_bytes += 1
        for v in final_per_case_results:
            ts=v.get("time_sec"); pm=v.get("peak_memory_kb")
            if ts is not None and v.get("status")!="TLE":
                sum_time_sec += float(ts); cnt_time += 1
            if pm is not None:
                sum_peak_kb += int(pm); cnt_peak += 1

    # ベスト選定
    def iter_key(x):
        try: idx=int(x["iter"].split("_")[1])
        except Exception: idx=0
        return (x["iter_pass_all"], x["passed"], -(x.get("rounds_used",0)), -idx)
    best = max(tries_for_overall, key=iter_key) if tries_for_overall else {}

    # best_solution.cpp
    if best:
        best_iter_dir = problem_dir / "outputs" / best["iter"]
        rounds = sorted([p for p in best_iter_dir.glob("round_*") if p.is_dir()])
        if rounds:
            best_round_dir = rounds[-1]
            src = best_round_dir / CPP_SOURCE_NAME
            dst = problem_dir / "outputs" / "best_solution.cpp"
            if src.exists():
                shutil.copy2(src, dst)
                print(f"[best] {best['iter']} ({best_round_dir.name}) → {dst}")

    avg_time_sec = (sum_time_sec / cnt_time) if cnt_time else None
    avg_peak_memory_kb = (sum_peak_kb / cnt_peak) if cnt_peak else None
    avg_code_bytes = (sum_code_bytes / cnt_code_bytes) if cnt_code_bytes else None

    overall = {
        "timestamp_jst": jst_now_iso(),
        "problem_dir": str(problem_dir),
        "tries": tries_for_overall,
        "best": best,
        "docker": {"image": DOCKER_IMAGE, "memory_limit": DOCKER_MEMORY_LIMIT, "cpus": DOCKER_CPUS},
        "gen": {"n_tries": N_TRIES, "model": MODEL_ID, "temp": TEMPERATURE, "top_p": TOP_P, "max_new_tokens": MAX_NEW_TOKENS, "feedback_rounds_max": MAX_FEEDBACK_ROUNDS},
        "parallel": {"jobs": PARALLEL_JOBS},
        "averages": {"time_sec": avg_time_sec, "peak_memory_kb": avg_peak_memory_kb, "code_bytes": avg_code_bytes,
                     "samples": {"time": cnt_time, "peak_memory": cnt_peak, "iters": cnt_code_bytes}}
    }
    out_path = problem_dir / "outputs" / "overall_summary.json"
    out_path.write_text(json.dumps(overall, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] tries={len(tries_for_overall)}  best={best.get('iter')}")
    print(f"[path] overall_summary.json → {out_path}")

def main():
    if not PROBLEM_DIRS:
        raise RuntimeError("PROBLEM_DIRS が空です。問題フォルダを設定してください。")
    tok = build_tokenizer()
    model = build_model_4bit_full_gpu()
    for prob in PROBLEM_DIRS:
        process_problem(prob, tok, model)

if __name__ == "__main__":
    main()
