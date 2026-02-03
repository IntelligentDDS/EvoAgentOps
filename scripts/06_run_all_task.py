# run_all.py - Unified runner for multiple agent tasks
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
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

import numpy as np
import pandas as pd
from openai import OpenAI
from datasets import load_dataset

# ==================== Task Configurations ====================
TASK_CONFIGS = {
    "autogen_math": {
        "agent_prompt_map": {
            "MathSolverA": "MATHSOLVERA_SYSTEM",
            "MathSolverB": "MATHSOLVERB_SYSTEM",
            "MathSolverC": "MATHSOLVERC_SYSTEM",
            "MathSolverD": "MATHSOLVERD_SYSTEM",
        },
        "main_script": "../AutoGen_GSM8K/main.py",
        "test_dir": "../../datasets/raw_split/autogen_math/test",
        "output_base": "../../results/test_mas/autogen_math",
        "bank_path": "../../results/train_bank/autogen_math",
        "timeout": 3600,
        "result_key": "RESULT:",
        "query_field": "question",
    },
    "veadk_gaia": {
        "agent_prompt_map": {
            "ck_plan_agent": "_CK_PLAN_SYS",
            "ck_action_agent": "_CK_ACTION_SYS",
            "ck_end_agent": "_CK_END_SYS",
            "file_plan_agent": "_FILE_PLAN_SYS",
            "file_action_agent": "_FILE_ACTION_SYS",
            "file_end_agent": "_FILE_END_SYS",
            "web_plan_agent": "_WEB_PLAN_SYS",
            "web_action_agent": "_WEB_ACTION_SYS",
            "web_end_agent": "_WEB_END_SYS",
        },
        "main_script": "../CogKernelPro/main.py",
        "metadata_file": "../CogKernelPro/_test/validation/metadata_filtered.jsonl",
        "test_dir": "../../datasets/raw_split/veadk_gaia/test",
        "output_base": "../../results/test_mas/veadk_gaia",
        "bank_path": "../../results/train_bank/veadk_gaia",
        "timeout": 3600,
        "result_key": "CK_RESULT:",
        "query_field": "question",
    },
    "langgraph_sql": {
        "agent_prompt_map": {
            "write_query": "WRITE_QUERY_SYSTEM",
            "check_query": "CHECK_QUERY_SYSTEM",
            "rewrite_query": "REWRITE_QUERY_SYSTEM",
        },
        "main_script": "../LangGraph_Spider/main.py",
        "task_file": "../LangGraph_Spider/data/test.json",
        "test_dir": "../../datasets/raw_split/langgraph_sql/test",
        "output_base": "../../results/test_mas/langgraph_sql",
        "bank_path": "../../results/train_bank/langgraph_sql",
        "timeout": 3600,
        "result_key": "RESULT:",
        "query_field": "question",
    },
    "agno_rca": {
        "agent_prompt_map": {
            "reasoning_agent": "reasoning_agent_system",
            "execution_agent": "execution_agent_system",
        },
        "main_script": "../OpenRCA/main.py",
        "csv_file": "../OpenRCA/dataset/Telecom/query.csv",
        "test_dir": "../../datasets/raw_split/agno_rca/test",
        "output_base": "../../results/test_mas/agno_rca",
        "bank_path": "../../results/train_bank/agno_rca",
        "timeout": 3600,
        "result_key": "RESULT:",
        "query_field": "instruction",
    },
    "adk_swe": {
        "agent_prompt_map": {
            "swe_agent": "SYSTEM_PROMPT",
        },
        "main_script": "../SWE/main.py",
        "test_dir": "../../datasets/raw_split/adk_swe/test",
        "output_base": "../../results/test_mas/adk_swe",
        "bank_path": "../../results/train_bank/adk_swe",
        "timeout": 7200,
        "result_key": "RESULT:",
        "query_field": "problem_statement",
    },
}


# ==================== Data Loading Functions ====================
def load_tasks_gsm8k() -> dict:
    """Load tasks from gsm8k dataset"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)
    tasks = {}
    for i, example in enumerate(dataset):
        task_id = f"gsm8k-{i:04d}"
        tasks[task_id] = {"task_id": task_id, "question": example["question"], "answer": example["answer"]}
    return tasks


def load_tasks_swebench() -> dict:
    """Load tasks from SWE-bench dataset"""
    cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", cache_dir=cache_dir)
    tasks = {}
    for example in dataset:
        task_id = example["instance_id"]
        tasks[task_id] = {
            "task_id": task_id,
            "problem_statement": example["problem_statement"],
            "patch": example["patch"],
        }
    return tasks


def load_tasks_from_metadata(metadata_file: str) -> dict:
    """Load tasks from jsonl metadata file"""
    metadata = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                tid = item.get("task_id")
                if tid:
                    metadata[tid] = item
    return metadata


def load_tasks_from_json(task_file: str) -> dict:
    """Load tasks from json file (Spider format)"""
    with open(task_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    tasks = {}
    for i, sample in enumerate(samples):
        task_id = f"spider-{i:04d}"
        tasks[task_id] = {"task_id": task_id, **sample}
    return tasks


def load_tasks_from_csv(csv_path: str) -> dict:
    """Load tasks from CSV file (RCA format)"""
    df = pd.read_csv(csv_path)
    tasks = {}
    for i, row in df.iterrows():
        task_id = f"{row['task_index']}-{i:04d}"
        tasks[task_id] = {
            "task_id": task_id,
            "instruction": row["instruction"],
            "scoring_points": row["scoring_points"],
        }
    return tasks


def load_tasks_for_task_type(task_type: str, config: dict) -> tuple[dict, str | None]:
    """Load tasks based on task type, return (tasks_dict, extra_dir)"""
    if task_type == "autogen_math":
        return load_tasks_gsm8k(), None
    elif task_type == "adk_swe":
        return load_tasks_swebench(), None
    elif task_type == "veadk_gaia":
        metadata_file = os.path.abspath(config["metadata_file"])
        return load_tasks_from_metadata(metadata_file), os.path.dirname(metadata_file)
    elif task_type == "langgraph_sql":
        task_file = os.path.abspath(config["task_file"])
        return load_tasks_from_json(task_file), os.path.dirname(task_file)
    elif task_type == "agno_rca":
        csv_file = os.path.abspath(config["csv_file"])
        return load_tasks_from_csv(csv_file), None
    else:
        raise ValueError(f"Unknown task type: {task_type}")


# ==================== Common Utilities ====================
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
    """Find existing workdir for task_id (with timestamp suffix)"""
    root = Path(workdir_root)
    if not root.exists():
        return None
    for d in root.iterdir():
        if d.is_dir() and extract_task_id(d.name) == task_id:
            return str(d)
    return None


# ==================== Prompt Modification ====================
def create_modified_prompt(
    query: str, prompt_path: str, bank_path: str, agent_prompt_map: dict, topkg: int = 3, topka: int = 3
):
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
        client.embeddings.create(input=[query[:8000]], encoding_format="float", model=os.environ["EMBEDDING_MODEL"])
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
    for agent_name, var_name in agent_prompt_map.items():
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


# ==================== Task Preparation ====================
def prepare_task_workdir(task: dict, workdir: str, task_type: str, extra_dir: str = None) -> str:
    """Prepare workdir: create dirs, write qa.json, copy extra files. Return query string."""
    setup_workdir(workdir)

    # Write qa.json
    qa_path = os.path.join(workdir, "qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump([task], f, indent=2, ensure_ascii=False)

    # Copy extra files based on task type
    if task_type == "veadk_gaia" and extra_dir:
        info = task.get("info", task)
        file_name = info.get("file_name")
        if file_name:
            src_path = os.path.join(extra_dir, file_name)
            if os.path.exists(src_path):
                shutil.copy(src_path, os.path.join(workdir, os.path.basename(file_name)))
        # Extract query from info
        for key in ["question", "Question", "task", "Task", "query", "Query"]:
            if key in info and info[key]:
                return info[key]
        return ""

    elif task_type == "langgraph_sql" and extra_dir:
        db_id = task.get("db_id")
        if db_id:
            for sub in ["database", "test_database", ""]:
                db_dir = os.path.join(extra_dir, sub, db_id) if sub else os.path.join(extra_dir, db_id)
                db_path = os.path.join(db_dir, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    shutil.copy(db_path, os.path.join(workdir, f"{db_id}.sqlite"))
                    schema_path = os.path.join(db_dir, "schema.sql")
                    if os.path.exists(schema_path):
                        shutil.copy(schema_path, os.path.join(workdir, "schema.sql"))
                    break

    # Return query field based on config
    config = TASK_CONFIGS[task_type]
    return task.get(config["query_field"], "")


# ==================== Task Execution ====================
def run_single_task(
    task,
    task_type,
    config,
    workdir_root,
    main_script,
    source_prompt,
    backup_prompt,
    env,
    prompt_lock,
    extra_dir,
    topkg,
    topka,
):
    """Run a single task, return (task_id, success, correct)"""
    task_id = task["task_id"]

    existing_workdir = find_existing_workdir(workdir_root, task_id)
    if existing_workdir:
        completed = load_completed_result(existing_workdir)
        if completed:
            print(f"[{task_type}][{task_id}] Skipped (already completed)", flush=True)
            return task_id, completed[0], completed[1]
        shutil.rmtree(existing_workdir, ignore_errors=True)
        print(f"[{task_type}][{task_id}] Removed incomplete workdir", flush=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    workdir = os.path.abspath(os.path.join(workdir_root, f"{task_id}_{timestamp}"))
    query = prepare_task_workdir(task, workdir, task_type, extra_dir)

    for attempt in range(1, 11):
        print(f"[{task_type}][{task_id}] Attempt {attempt}/10", flush=True)
        proc = None
        try:
            with prompt_lock:
                shutil.copy(backup_prompt, source_prompt)
                create_modified_prompt(
                    query, source_prompt, config["bank_path"], config["agent_prompt_map"], topkg, topka
                )
                shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))
                cmd = [sys.executable, main_script, "--workdir", workdir]
                proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env
                )
                time.sleep(2)
                shutil.copy(backup_prompt, source_prompt)

            timer = threading.Timer(config["timeout"], proc.kill)
            timer.start()
            stdout_lines = []
            try:
                for line in proc.stdout:
                    print(f"[{task_type}][{task_id}] {line.rstrip()}", flush=True)
                    stdout_lines.append(line)
            finally:
                timer.cancel()

            proc.wait()
            if proc.returncode == -9:
                raise subprocess.TimeoutExpired(cmd, config["timeout"])

            if proc.returncode == 0:
                corr = 0
                result_key = config["result_key"]
                for line in stdout_lines:
                    if line.startswith(result_key):
                        try:
                            corr = int(json.loads(line.split(result_key, 1)[1].strip()).get("_this_corr", 0))
                        except:
                            pass
                print(f"[{task_type}][{task_id}] Done, correct={corr}", flush=True)
                return task_id, True, corr
            else:
                print(f"[{task_type}][{task_id}] Failed rc={proc.returncode}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"[{task_type}][{task_id}] Timeout", flush=True)
        except Exception as e:
            print(f"[{task_type}][{task_id}] Error: {e}", flush=True)
        finally:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
            if os.path.exists(backup_prompt):
                shutil.copy(backup_prompt, source_prompt)

    return task_id, False, 0


def run_task_type(task_type: str, args, env, prompt_lock):
    """Run all tasks for a specific task type"""
    config = TASK_CONFIGS[task_type]
    print(f"\n{'='*60}\nRunning task type: {task_type}\n{'='*60}")

    # Load tasks
    tasks, extra_dir = load_tasks_for_task_type(task_type, config)
    print(f"Loaded {len(tasks)} tasks for {task_type}")

    # Setup paths
    main_script = os.path.abspath(config["main_script"])
    source_prompt = os.path.join(os.path.dirname(main_script), "prompt.py")
    backup_prompt = os.path.join(os.path.dirname(main_script), "prompt.py.bak")
    if not os.path.exists(backup_prompt):
        shutil.copy(source_prompt, backup_prompt)

    workdir_root = os.path.join(config["output_base"], f"g{args.topkg}_a{args.topka}")
    os.makedirs(workdir_root, exist_ok=True)

    if args.mode == "debug":
        # Debug mode: run single task
        task = tasks.get(args.task_id) if args.task_id else next(iter(tasks.values()), None)
        if not task:
            print(f"[{task_type}] No task found")
            return {}

        existing_workdir = find_existing_workdir(workdir_root, task["task_id"])
        if existing_workdir:
            if load_completed_result(existing_workdir):
                print(f"[{task_type}] Task already completed: {existing_workdir}")
                return {task["task_id"]: (True, 1)}
            shutil.rmtree(existing_workdir, ignore_errors=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        workdir = os.path.abspath(os.path.join(workdir_root, f"{task['task_id']}_{timestamp}"))
        query = prepare_task_workdir(task, workdir, task_type, extra_dir)

        shutil.copy(backup_prompt, source_prompt)
        create_modified_prompt(
            query, source_prompt, config["bank_path"], config["agent_prompt_map"], args.topkg, args.topka
        )
        shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))

        sys.argv = [main_script, "--workdir", workdir]
        try:
            runpy.run_path(main_script, run_name="__main__")
        except SystemExit:
            pass
        finally:
            shutil.copy(backup_prompt, source_prompt)

        result = load_completed_result(workdir)
        return {task["task_id"]: result if result else (False, 0)}

    elif args.mode == "test_batch":
        # Batch mode: run tasks from test_dir
        test_task_ids = collect_task_ids_from_test_dir(config["test_dir"])
        print(f"[{task_type}] Found {len(test_task_ids)} task_ids from {config['test_dir']}")

        batch_tasks = [tasks[tid] for tid in test_task_ids if tid in tasks]
        print(f"[{task_type}] Matched {len(batch_tasks)} tasks")

        results = {}
        pending = []
        for task in batch_tasks:
            existing_workdir = find_existing_workdir(workdir_root, task["task_id"])
            completed = load_completed_result(existing_workdir) if existing_workdir else None
            if completed:
                results[task["task_id"]] = completed
                print(f"[{task_type}][{task['task_id']}] Skipped (already completed)", flush=True)
            else:
                pending.append(task)

        print(f"[{task_type}] Skipped {len(results)} completed, {len(pending)} pending")

        with ThreadPoolExecutor(max_workers=1) as executor:
            futures = {
                executor.submit(
                    run_single_task,
                    task,
                    task_type,
                    config,
                    workdir_root,
                    main_script,
                    source_prompt,
                    backup_prompt,
                    env,
                    prompt_lock,
                    extra_dir,
                    args.topkg,
                    args.topka,
                ): task["task_id"]
                for task in pending
            }
            for f in as_completed(futures):
                tid, ok, corr = f.result()
                results[tid] = (ok, corr)

        return results


# ==================== Main ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified runner for multiple agent tasks")
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["autogen_math"],
        choices=list(TASK_CONFIGS.keys()),
        help="Task types to run",
    )
    parser.add_argument("--mode", type=str, default="debug", choices=["debug", "test_batch"])
    parser.add_argument("--task_id", type=str, default="", help="Specific task id for debug mode")
    parser.add_argument("--topkg", type=int, default=3, help="Top-k global principles")
    parser.add_argument("--topka", type=int, default=5, help="Top-k agent principles")

    load_dotenv()
    args = parser.parse_args()

    env = os.environ.copy()
    if "OPENAI_MODEL_NAME" not in env:
        env["OPENAI_MODEL_NAME"] = f"openai/{os.getenv('OPENAI_MODEL', 'doubao-seed-1.6-250615')}"
    os.environ["OPENAI_MODEL_NAME"] = env["OPENAI_MODEL_NAME"]

    prompt_lock = threading.Lock()
    all_results = {}

    for task_type in args.tasks:
        results = run_task_type(task_type, args, env, prompt_lock)
        all_results[task_type] = results

    # Print summary
    print(f"\n{'='*60}\nFinal Summary\n{'='*60}")
    for task_type, results in all_results.items():
        if results:
            total = len(results)
            success = sum(1 for ok, _ in results.values() if ok)
            correct = sum(corr for _, corr in results.values())
            print(f"[{task_type}] Success: {success}/{total}, Correct: {correct}/{total}")
