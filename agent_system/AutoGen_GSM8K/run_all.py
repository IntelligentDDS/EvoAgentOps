# run_all.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import sys
import json
import shutil
import subprocess
import time
import argparse
import threading
import runpy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
import numpy as np
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from datasets import load_dataset

AGENT_PROMPT_MAP = {
    "MathSolverA": "MATHSOLVERA_SYSTEM",
    "MathSolverB": "MATHSOLVERB_SYSTEM",
    "MathSolverC": "MATHSOLVERC_SYSTEM",
    "MathSolverD": "MATHSOLVERD_SYSTEM",
}


def setup_workdir(workdir):
    """Set up working directory structure"""
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "monitor_data"), exist_ok=True)


def extract_task_id(folder_name: str) -> str:
    """Extract task_id from folder name by removing timestamp suffix"""
    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 14:
        return parts[0]
    return folder_name


def load_tasks_from_dataset() -> dict:
    """Load tasks from gsm8k dataset, return dict keyed by task_id"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    tasks = {}
    for i, example in enumerate(dataset):
        task_id = f"gsm8k-{i:04d}"
        tasks[task_id] = {
            "task_id": task_id,
            "question": example["question"],
            "answer": example["answer"],
        }
    return tasks


def collect_task_ids_from_test_dir(test_dir: str) -> list:
    """Collect task_ids from test/success and test/fail directories"""
    task_ids = []
    test_path = Path(test_dir)
    for status_dir in ["success", "fail"]:
        status_path = test_path / status_dir
        if not status_path.exists():
            continue
        for case_dir in sorted(status_path.iterdir()):
            if case_dir.is_dir():
                task_ids.append(extract_task_id(case_dir.name))
    return task_ids


def create_modified_prompt(query: str, prompt_path: str, bank_path: str, topkg: int = 3, topka: int = 3):
    """Modify prompt.py in-place with retrieved principles appended"""
    bank_file = Path(bank_path) / "principlebank" / "principlebank.jsonl"
    if (topkg <= 0 and topka <= 0) or not bank_file.exists():
        return
    principles = [json.loads(line) for line in bank_file.read_text().splitlines() if line.strip()]
    if not principles:
        return
    content = Path(prompt_path).read_text(encoding="utf-8")
    mod = {}
    exec(content, mod)
    client = OpenAI(base_url=os.environ["EMBEDDING_BASE_URL"], api_key=os.environ["EMBEDDING_API_KEY"])
    query_emb = np.array(
        client.embeddings.create(input=[query], encoding_format="float", model=os.environ["EMBEDDING_MODEL"])
        .data[0]
        .embedding
    )

    def get_top_principles(candidates, k):
        if not candidates or k <= 0:
            return []

        def cosine_sim(p):
            emb = np.array(p["embedding"])
            return np.dot(query_emb, emb) / (np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8)

        scored = sorted(
            ((cosine_sim(p), p) for p in candidates if p.get("embedding")), reverse=True, key=lambda x: x[0]
        )
        return [p for _, p in scored[:k]]

    def format_principles(agent_ps, global_ps):
        if not agent_ps and not global_ps:
            return ""
        lines = ["\n<principles>", "Principles are advisory, not mandatory."]
        if agent_ps:
            lines.append("# agent principle: agent specific principle")
            for p in agent_ps:
                lines.append(f"Title: {p['title']}\nContent: {p['content']}")
        if global_ps:
            lines.append("# global principle: whole system principle")
            for p in global_ps:
                lines.append(f"Title: {p['title']}\nContent: {p['content']}")
        lines.append("</principles>")
        return "\n".join(lines)

    global_ps, agent_ps = [], defaultdict(list)
    for p in principles:
        (global_ps if p.get("type") == "global" else agent_ps[p.get("agent_name", "")]).append(p)
    global_top = get_top_principles(global_ps, topkg)
    overrides = {}
    for agent_name, var_name in AGENT_PROMPT_MAP.items():
        orig = mod.get(var_name)
        if orig is None:
            continue
        agent_top = get_top_principles(agent_ps.get(agent_name, []), topka)
        text = format_principles(agent_top, global_top)
        if text:
            overrides[var_name] = orig + text
    if overrides:
        content += "\n\n# === Principles ===\n"
        content += "\n".join(f"{k} = {repr(v)}" for k, v in overrides.items())
        Path(prompt_path).write_text(content, encoding="utf-8")


def prepare_task_workdir(task: dict, workdir: str) -> str:
    """Prepare workdir: create dirs, write qa.json. Return question."""
    setup_workdir(workdir)
    qa_path = os.path.join(workdir, "qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump([task], f, indent=2, ensure_ascii=False)
    return task.get("question", "")


def load_completed_result(workdir):
    """Load result from output.json if exists, return (success, correct) or None"""
    output_file = Path(workdir) / "output.json"
    if not output_file.exists():
        return None
    try:
        data = json.loads(output_file.read_text())
        return (True, 1 if data.get("status") == "success" else 0)
    except:
        return None


def find_existing_workdir(workdir_root: str, task_id: str) -> str | None:
    """Find existing workdir for task_id (with timestamp suffix), return path or None"""
    root = Path(workdir_root)
    if not root.exists():
        return None
    for d in root.iterdir():
        if d.is_dir() and extract_task_id(d.name) == task_id:
            return str(d)
    return None


def run_single_task(task, args, workdir_root, main_script, source_prompt, backup_prompt, env, prompt_lock):
    """Run a single task, return (task_id, success, correct)"""
    task_id = task["task_id"]
    existing_workdir = find_existing_workdir(workdir_root, task_id)
    if existing_workdir:
        completed = load_completed_result(existing_workdir)
        if completed:
            print(f"[{task_id}] Skipped (already completed)", flush=True)
            return task_id, completed[0], completed[1]
        shutil.rmtree(existing_workdir)
        print(f"[{task_id}] Removed incomplete workdir", flush=True)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    workdir = os.path.join(workdir_root, f"{task_id}_{timestamp}")
    query = prepare_task_workdir(task, workdir)
    for attempt in range(1, 11):
        print(f"[{task_id}] Attempt {attempt}/10", flush=True)
        proc = None
        try:
            with prompt_lock:
                shutil.copy(backup_prompt, source_prompt)
                create_modified_prompt(
                    query, source_prompt, bank_path=args.bank_path, topkg=args.topkg, topka=args.topka
                )
                shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))
                cmd = [sys.executable, main_script, "--workdir", workdir]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
                )
                time.sleep(2)
                shutil.copy(backup_prompt, source_prompt)
            timer = threading.Timer(3600, proc.kill)
            timer.start()
            stdout_lines = []
            try:
                for line in proc.stdout:
                    print(f"[{task_id}] {line.rstrip()}", flush=True)
                    stdout_lines.append(line)
            finally:
                timer.cancel()
            proc.wait()
            if proc.returncode == -9:
                raise subprocess.TimeoutExpired(cmd, 3600)
            if proc.returncode == 0:
                corr = 0
                for line in stdout_lines:
                    if line.startswith("RESULT:"):
                        try:
                            corr = int(json.loads(line.split("RESULT:", 1)[1].strip()).get("_this_corr", 0))
                        except:
                            pass
                print(f"[{task_id}] Done, correct={corr}", flush=True)
                return task_id, True, corr
            else:
                print(f"[{task_id}] Failed rc={proc.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[{task_id}] Timeout", flush=True)
        except Exception as e:
            print(f"[{task_id}] Error: {e}", flush=True)
        finally:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
            if os.path.exists(backup_prompt):
                shutil.copy(backup_prompt, source_prompt)
    return task_id, False, 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run agent on batch tasks")
    parser.add_argument("--test_dir", type=str, default="../../datasets/raw_split/autogen_math/test")
    parser.add_argument("--main_script", type=str, default="./main.py")
    parser.add_argument("--output_base", type=str, default="../../results/test_mas/autogen_math")
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "test_batch"])
    parser.add_argument("--task_id", type=str, default="gsm8k-0013")
    parser.add_argument("--topkg", type=int, default=3)
    parser.add_argument("--topka", type=int, default=5)
    parser.add_argument("--bank_path", type=str, default="../../results/train_bank/autogen_math")
    load_dotenv()
    args = parser.parse_args()
    env = os.environ.copy()
    main_script = os.path.abspath(args.main_script)
    source_prompt = os.path.join(os.path.dirname(main_script), "prompt.py")
    backup_prompt = os.path.join(os.path.dirname(main_script), "prompt.py.bak")
    if not os.path.exists(backup_prompt):
        shutil.copy(source_prompt, backup_prompt)
    tasks = load_tasks_from_dataset()
    print(f"Loaded {len(tasks)} tasks from gsm8k dataset")
    workdir_root = os.path.join(args.output_base, f"g{args.topkg}_a{args.topka}")
    os.makedirs(workdir_root, exist_ok=True)
    prompt_lock = threading.Lock()
    if args.mode == "debug":
        task = tasks.get(args.task_id) if args.task_id else next(iter(tasks.values()), None)
        if not task:
            print("No task found")
            sys.exit(1)
        existing_workdir = find_existing_workdir(workdir_root, task["task_id"])
        if existing_workdir:
            if load_completed_result(existing_workdir):
                print(f"Task already completed: {existing_workdir}")
                sys.exit(0)
            shutil.rmtree(existing_workdir)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        workdir = os.path.join(workdir_root, f"{task['task_id']}_{timestamp}")
        query = prepare_task_workdir(task, workdir)
        shutil.copy(backup_prompt, source_prompt)
        create_modified_prompt(query, source_prompt, bank_path=args.bank_path, topkg=args.topkg, topka=args.topka)
        shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))
        sys.argv = [main_script, "--workdir", workdir]
        try:
            runpy.run_path(main_script, run_name="__main__")
        finally:
            shutil.copy(backup_prompt, source_prompt)
    elif args.mode == "test_batch":
        test_task_ids = collect_task_ids_from_test_dir(args.test_dir)
        print(f"Found {len(test_task_ids)} task_ids from {args.test_dir}")
        batch_tasks = [tasks[tid] for tid in test_task_ids if tid in tasks]
        print(f"Matched {len(batch_tasks)} tasks")
        results = {}
        pending = []
        for task in batch_tasks:
            existing_workdir = find_existing_workdir(workdir_root, task["task_id"])
            completed = load_completed_result(existing_workdir) if existing_workdir else None
            if completed:
                results[task["task_id"]] = completed
                print(f"[{task['task_id']}] Skipped (already completed)", flush=True)
            else:
                pending.append(task)
        print(f"Skipped {len(results)} completed, {len(pending)} pending")
        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(
                    run_single_task,
                    task,
                    args,
                    workdir_root,
                    main_script,
                    source_prompt,
                    backup_prompt,
                    env,
                    prompt_lock,
                ): task["task_id"]
                for task in pending
            }
            for f in as_completed(futures):
                tid, ok, corr = f.result()
                results[tid] = (ok, corr)
        total_corr = sum(corr for _, corr in results.values())
        print(f"\n{'='*60}")
        print(f"Results: {sum(1 for ok, _ in results.values() if ok)}/{len(results)} success")
        print(f"Correct: {total_corr}/{len(results)}")
