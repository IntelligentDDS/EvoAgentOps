# main.py
import os, sys, json, shutil, subprocess, time, argparse, threading, runpy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from pathlib import Path
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
import contextlib
from pydantic import BaseModel, Field
from typing import List
from evoagentops.config import Config
from evoagentops.util import call_embedding, acall_llm
import asyncio
from evoagentops.prompt import get_keyword_system_prompt, get_rerank_system_prompt

# === Environment variable keys for single mode ===
ENV_MODEL = "EXEC_MODEL"
ENV_DATASET = "EXEC_DATASET"
ENV_TASK_ID = "EXEC_TASK_ID"
ENV_TOPKG = "EXEC_TOPKG"
ENV_TOPKA = "EXEC_TOPKA"
ENV_PRINCIPLE_SCOPE = "EXEC_PRINCIPLE_SCOPE"

# Used to match bank_model name to corresponding env vars
MODEL_PREFIX_MAP = {
    "deepseek": "DEEPSEEK",
    "gpt": "GPT",
    "seed": "SEED",
    "doubao": "SEED",
    "kimi": "KIMI",
    "claude": "CLAUDE",
}


def get_model_config(prefix: str) -> tuple:
    """Get (base_url, api_key, model) for a given env prefix"""
    return (
        os.getenv(f"{prefix}_OPENAI_BASE_URL"),
        os.getenv(f"{prefix}_OPENAI_API_KEY"),
        os.getenv(f"{prefix}_OPENAI_MODEL"),
    )


def get_rerank_config_from_model(model_name: str) -> tuple:
    """Get (base_url, api_key, model) for rerank/keyword LLM based on model_name keyword match.

    Maps bank_model names like 'deepseek-v3-2-251201' to DEEPSEEK_OPENAI_* env vars.
    Agent execution uses SEED_OPENAI_* separately (set in run_single_task).
    """
    model_lower = model_name.lower() if model_name else ""
    for keyword, prefix in MODEL_PREFIX_MAP.items():
        if keyword in model_lower:
            return get_model_config(prefix)
    return get_model_config("SEED")  # Fallback to SEED instead of generic OPENAI_*


class KeywordsOutput(BaseModel):
    keywords: List[str] = Field(description="3-5 keywords for execute principles")


class RerankOutput(BaseModel):
    sorted_indices: List[int] = Field(description="ALL indices sorted by relevance")
    reason: str = Field(description="Brief rationale")


# === Agent System Configurations ===
AGENT_CONFIGS = {
    "autogen_math": {
        "dir": "AutoGen_GSM8K",
        "prompt_map": {
            "MathSolverA": "MATHSOLVERA_SYSTEM",
            "MathSolverB": "MATHSOLVERB_SYSTEM",
            "MathSolverC": "MATHSOLVERC_SYSTEM",
            "MathSolverD": "MATHSOLVERD_SYSTEM",
        },
        "result_prefix": "RESULT:",
    },
    "veadk_gaia": {
        "dir": "CogKernelPro",
        "prompt_map": {
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
        "result_prefix": "CK_RESULT:",
        "metadata_file": "_test/validation/metadata_filtered.jsonl",
    },
    "langgraph_sql": {
        "dir": "LangGraph_Spider",
        "prompt_map": {
            "write_query": "WRITE_QUERY_SYSTEM",
            "check_query": "CHECK_QUERY_SYSTEM",
            "rewrite_query": "REWRITE_QUERY_SYSTEM",
        },
        "result_prefix": "RESULT:",
        "task_file": "data/test.json",
    },
    "agno_rca": {
        "dir": "OpenRCA",
        "prompt_map": {
            "reasoning_agent": "reasoning_agent_system",
            "execution_agent": "execution_agent_system",
        },
        "result_prefix": "RESULT:",
        "csv_file": "dataset/Market/cloudbed-1/query.csv",
    },
    "adk_swe": {
        "dir": "SWE",
        "prompt_map": {
            "swe_agent": "SYSTEM_PROMPT",
        },
        "result_prefix": "RESULT:",
    },
}


class RetrievalCache:
    """Cache retrieval results for reuse across different top_k values"""

    def __init__(self, cache_file: str):
        self.cache_file = Path(cache_file)
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        self._cache = self._load_cache()
        print(f"[RetrievalCache] Loaded {len(self._cache)} cases from {cache_file}", flush=True)

    def _load_cache(self) -> dict:
        cache = {}
        if self.cache_file.exists():
            with open(self.cache_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if tid := data.get("task_id"):
                            cache[tid] = data
                    except:
                        pass
        return cache

    def get(self, task_id: str) -> dict | None:
        return self._cache.get(task_id)

    def save(self, task_id: str, data: dict):
        data["task_id"] = task_id
        is_new = task_id not in self._cache
        self._cache[task_id] = data
        with open(self.cache_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
        if is_new:
            print(f"[RetrievalCache] NEW: {task_id} (total={len(self._cache)})", flush=True)


# === Utility Functions ===
def get_project_root():
    """Get project root (parent of scripts dir)"""
    return Path(__file__).resolve().parent.parent


def setup_workdir(workdir):
    """Create working directory structure"""
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(os.path.join(workdir, "monitor_data"), exist_ok=True)
    os.makedirs(os.path.join(workdir, "metrics"), exist_ok=True)


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


def find_existing_workdir(workdir_root: str, task_id: str) -> str | None:
    """Find existing workdir for task_id"""
    root = Path(workdir_root)
    if not root.exists():
        return None
    for d in root.iterdir():
        if d.is_dir() and extract_task_id(d.name) == task_id:
            return str(d)
    return None


def load_completed_result(workdir):
    output_file = Path(workdir) / "output.json"
    if not output_file.exists():
        return None
    try:
        data = json.loads(output_file.read_text())
        return (True, 1 if data.get("status") == "success" else 0)
    except:
        return None


# === Task Loading Functions ===
def load_tasks_gsm8k() -> dict:
    """Load tasks from gsm8k dataset"""
    from datasets import load_dataset

    cache_dir = get_project_root() / "agent_system" / "AutoGen_GSM8K" / ".cache"
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=str(cache_dir))
    return {
        f"gsm8k-{i:04d}": {"task_id": f"gsm8k-{i:04d}", "question": ex["question"], "answer": ex["answer"]}
        for i, ex in enumerate(dataset)
    }


def load_tasks_swe() -> dict:
    """Load tasks from SWE-bench dataset"""
    from datasets import load_dataset

    cache_dir = get_project_root() / "agent_system" / "SWE" / ".cache"
    dataset = load_dataset("princeton-nlp/SWE-bench_Verified", split="test", cache_dir=str(cache_dir))
    return {
        ex["instance_id"]: {
            "task_id": ex["instance_id"],
            "problem_statement": ex["problem_statement"],
            "patch": ex["patch"],
        }
        for ex in dataset
    }


def load_tasks_metadata(metadata_file: str) -> dict:
    """Load tasks from jsonl metadata file"""
    tasks = {}
    with open(metadata_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line)
                if tid := item.get("task_id"):
                    tasks[tid] = item
    return tasks


def load_tasks_json(task_file: str) -> dict:
    """Load tasks from json file"""
    with open(task_file, "r", encoding="utf-8") as f:
        samples = json.load(f)
    return {f"spider-{i:04d}": {"task_id": f"spider-{i:04d}", **s} for i, s in enumerate(samples)}


def load_tasks_csv(csv_file: str) -> dict:
    """Load tasks from csv file"""
    import pandas as pd

    df = pd.read_csv(csv_file)
    return {
        f"{row['task_index']}-{i:04d}": {
            "task_id": f"{row['task_index']}-{i:04d}",
            "instruction": row["instruction"],
            "scoring_points": row["scoring_points"],
        }
        for i, row in df.iterrows()
    }


def create_modified_prompt(
    prompt_path: str,
    prompt_map: dict,  # {agent_name: prompt_var_name}
    global_top: list,  # top-k global execute_principles
    agent_dict: dict,  # {agent_name: top-k agent execute_principles}
):
    """Modify prompt.py with pre-retrieved principles"""
    if not global_top and not agent_dict:
        return

    content = Path(prompt_path).read_text(encoding="utf-8")
    mod = {}
    exec(content, mod)

    def format_principles(agent_name: str, agent_ps: list, global_ps: list) -> str:
        """Format principles for agent prompt injection."""
        if not agent_ps and not global_ps:
            return ""
        lines = ["\n<principles>", "Principles are advisory, not mandatory."]
        if global_ps:
            lines.append("# global level")
            for p in global_ps:
                lines.append(f"title: {p['title']}")
                lines.append(f"content: {p['content']}")
        if agent_ps:
            lines.append(f"# agent level: {agent_name}")
            for p in agent_ps:
                lines.append(f"title: {p['title']}")
                lines.append(f"content: {p['content']}")
        lines.append("</principles>")
        return "\n".join(lines)

    overrides = {}
    for agent_name, var_name in prompt_map.items():
        if (orig := mod.get(var_name)) is None:
            continue
        agent_ps = agent_dict.get(agent_name, [])
        if text := format_principles(agent_name, agent_ps, global_top):
            overrides[var_name] = orig + text

    if overrides:
        content += "\n\n# === Principles ===\n"
        content += "\n".join(f"{k} = {repr(v)}" for k, v in overrides.items())
        Path(prompt_path).write_text(content, encoding="utf-8")
        print(f"[Principles injected: {len(overrides)} agents, global={len(global_top)}]", flush=True)


# === Task Workdir Preparation ===
def prepare_task_workdir(task: dict, workdir: str, agent_name: str, extra_dir: str = None) -> str:
    """Prepare workdir and return query string"""
    setup_workdir(workdir)
    qa_path = os.path.join(workdir, "qa.json")
    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(task, f, indent=2, ensure_ascii=False)

    info = task.get("info", task)

    # Agent-specific file copying
    if agent_name == "veadk_gaia" and extra_dir:
        if file_name := info.get("file_name"):
            src = os.path.join(extra_dir, file_name)
            if os.path.exists(src):
                shutil.copy(src, os.path.join(workdir, os.path.basename(file_name)))
    elif agent_name == "langgraph_sql" and extra_dir:
        if db_id := task.get("db_id"):
            for sub in ["database", "test_database", ""]:
                db_dir = os.path.join(extra_dir, sub, db_id) if sub else os.path.join(extra_dir, db_id)
                db_path = os.path.join(db_dir, f"{db_id}.sqlite")
                if os.path.exists(db_path):
                    shutil.copy(db_path, os.path.join(workdir, f"{db_id}.sqlite"))
                    schema = os.path.join(db_dir, "schema.sql")
                    if os.path.exists(schema):
                        shutil.copy(schema, os.path.join(workdir, "schema.sql"))
                    break

    # Extract query
    for key in ["question", "Question", "task", "Task", "query", "Query", "instruction", "problem_statement"]:
        if key in info and info[key]:
            return info[key]
    return ""


def retrieve_principles_once(
    query: str,
    task_id: str,
    bank_path: str,
    prompt_map: dict,
    retrieval_cache: RetrievalCache = None,
    use_rerank: bool = True,
    top_e: int = 20,
    top_l: int = 20,
    min_sim: float = 0.3,
    rerank_model_name: str = None,
) -> dict:
    """Retrieve and cache full sorted results. Returns cached data."""
    # Check cache first
    if retrieval_cache and (cached := retrieval_cache.get(task_id)):
        print(f"[Retrieve] Cache hit: {task_id}", flush=True)
        return cached

    bank_file = Path(bank_path) / "principlebank" / "principlebank.jsonl"
    if not bank_file.exists():
        return {}

    # Load execute principles only
    principles = [
        p
        for line in bank_file.read_text().splitlines()
        if line.strip() and (p := json.loads(line)).get("function") == "execute" and p.get("embedding")
    ]
    if not principles:
        return {}
    # Config for embedding
    emb_config = Config(output_dir=str(Path(bank_path).parent))
    query_emb = np.array(call_embedding(query[:8000], emb_config))
    # Config for keyword/rerank LLM (use model-specific config)
    rerank_base_url, rerank_api_key, rerank_model = get_rerank_config_from_model(rerank_model_name)
    seed_base_url, seed_api_key, seed_model = get_model_config("SEED")
    llm_model = rerank_model or seed_model or "doubao-seed-1-8-251228"
    llm_config = Config(output_dir=str(Path(bank_path).parent))
    llm_config.openai_base_url = rerank_base_url or seed_base_url
    llm_config.openai_api_key = rerank_api_key or seed_api_key
    llm_config.openai_model = llm_model
    print(f"[Retrieve] Using rerank model: {llm_model}", flush=True)

    def rank_by_similarity(candidates: list) -> list:
        scored = []
        for p in candidates:
            if emb := p.get("embedding"):
                doc = np.array(emb)
                sim = float(np.dot(query_emb, doc) / (np.linalg.norm(query_emb) * np.linalg.norm(doc) + 1e-8))
                if sim >= min_sim:
                    scored.append((sim, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_e]]

    def expand_keywords(task_text: str) -> list:
        """LLM generates search keywords for execute_principle retrieval"""

        messages = [
            {"role": "system", "content": get_keyword_system_prompt("execute")},
            {"role": "user", "content": f"<task>{task_text[:500]}</task>"},
        ]
        try:
            resp = asyncio.run(acall_llm(messages, llm_config, output_schema=KeywordsOutput))
            return json.loads(resp).get("keywords", [])
        except Exception as e:
            print(f"[Keyword expand failed: {e}]", flush=True)
            return []

    def search_by_keywords(keywords: list, candidates: list) -> list:
        if not keywords:
            return []
        term_weights = {}
        for kw in keywords:
            kw_lower = kw.lower().strip()
            if len(kw_lower) >= 2:
                term_weights[kw_lower] = max(term_weights.get(kw_lower, 0), 2)
            for w in kw_lower.split():
                if len(w) >= 3 and w != kw_lower:
                    term_weights[w] = max(term_weights.get(w, 0), 1)
        scored = []
        for p in candidates:
            text = f"{p.get('title','')} {p.get('content','')}".lower()
            score = sum(wt for t, wt in term_weights.items() if t in text)
            if score > 0:
                scored.append((score, p))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored[:top_l]]

    def rerank_by_llm(task_text: str, candidates: list) -> list:

        n = len(candidates)
        if n == 0:
            return []
        cand_lines = [f"[{i}] {p['title']}: {p['content']}" for i, p in enumerate(candidates)]
        messages = [
            {"role": "system", "content": get_rerank_system_prompt("execute")},
            {
                "role": "user",
                "content": f"<task>{task_text[:800]}</task>\n<candidates>\n{chr(10).join(cand_lines)}\n</candidates>\nSort all {n} indices by execution relevance.",
            },
        ]
        try:
            resp = asyncio.run(acall_llm(messages, llm_config, output_schema=RerankOutput))
            result = json.loads(resp)
            indices = result.get("sorted_indices", [])
            seen = set()
            valid = [i for i in indices if 0 <= i < n and i not in seen and not seen.add(i)]
            for i in range(n):
                if i not in seen:
                    valid.append(i)
            print(f"[Rerank: n={n}, top5={valid[:5]}]", flush=True)
            return [candidates[i] for i in valid]
        except Exception as e:
            print(f"[Rerank failed: {e}]", flush=True)
            return candidates

    # Separate global and agent principles
    global_ps, agent_ps = [], defaultdict(list)
    for p in principles:
        (global_ps if p.get("type") == "global" else agent_ps[p.get("agent_name", "")]).append(p)

    # Global retrieval
    emb_cands = rank_by_similarity(global_ps)
    keywords = expand_keywords(query) if use_rerank else []
    kw_cands = search_by_keywords(keywords, global_ps)
    seen = {p["title"] for p in emb_cands}
    merged = emb_cands + [p for p in kw_cands if p["title"] not in seen]
    global_sorted = rerank_by_llm(query, merged) if use_rerank and merged else merged

    # Agent retrieval
    agent_sorted = {}
    for agent_name in prompt_map.keys():
        agent_filtered = agent_ps.get(agent_name, [])
        if not agent_filtered:
            continue
        emb_cands = rank_by_similarity(agent_filtered)
        kw_cands = search_by_keywords(keywords, agent_filtered)
        seen = {p["title"] for p in emb_cands}
        merged = emb_cands + [p for p in kw_cands if p["title"] not in seen]
        sorted_list = rerank_by_llm(query, merged) if use_rerank and merged else merged
        agent_sorted[agent_name] = [{"title": p["title"], "content": p["content"]} for p in sorted_list]

    # Build cache data
    cache_data = {
        "task_id": task_id,
        "query": query[:500],
        "keywords": keywords,
        "global_sorted": [{"title": p["title"], "content": p["content"]} for p in global_sorted],
        "agent_sorted": agent_sorted,
        "config": {"top_e": top_e, "top_l": top_l, "use_rerank": use_rerank},
    }

    if retrieval_cache:
        retrieval_cache.save(task_id, cache_data)

    print(f"[Retrieved] {task_id}: global={len(global_sorted)}, agents={list(agent_sorted.keys())}", flush=True)
    return cache_data


def extract_top_principles(cache_data: dict, topkg: int, topka: int) -> tuple:
    """Extract top-k principles from cached data. Returns (global_top, agent_dict)"""
    if not cache_data:
        return [], {}
    global_top = cache_data.get("global_sorted", [])[:topkg]
    agent_dict = {name: sorted_list[:topka] for name, sorted_list in cache_data.get("agent_sorted", {}).items()}
    return global_top, agent_dict


# === Task Runner ===
def run_single_task(
    task,
    config,
    agent_name,
    workdir_root,
    agent_dir,
    prompt_lock,
    extra_dir,
    topkg,
    topka,
    bank_path,
    retrieval_cache_data: dict = None,
    output_base: str = None,
):
    """Run a single task"""
    task_id = task["task_id"]
    main_script = os.path.join(agent_dir, "main.py")
    source_prompt = os.path.join(agent_dir, "prompt.py")
    backup_prompt = os.path.join(agent_dir, "prompt.py.bak")
    # Extract bank_model from bank_path for logging
    bank_model = Path(bank_path).parent.name if bank_path else "baseline"
    topk_tag = f"g{topkg}_a{topka}"

    # Check existing
    if existing := find_existing_workdir(workdir_root, task_id):
        if completed := load_completed_result(existing):
            print(f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] SKIP (completed)", flush=True)
            return task_id, completed[0], completed[1]
        shutil.rmtree(existing, ignore_errors=True)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    workdir = os.path.abspath(os.path.join(workdir_root, f"{task_id}_{timestamp}"))
    query = prepare_task_workdir(task, workdir, agent_name, extra_dir)
    print(f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] START workdir={workdir}", flush=True)

    env = os.environ.copy()
    # Agent execution always uses SEED model
    env["OPENAI_BASE_URL"] = os.getenv("SEED_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
    env["OPENAI_API_KEY"] = os.getenv("SEED_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    env["OPENAI_MODEL"] = os.getenv("SEED_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
    if agent_name in ("agno_rca", "adk_swe") and "OPENAI_MODEL_NAME" not in env:
        env["OPENAI_MODEL_NAME"] = f"openai/{env.get('OPENAI_MODEL', 'doubao-seed-1-8-251228')}"

    for attempt in range(1, 11):
        proc = None
        try:
            # Lock is optional (None when running serially within agent)
            lock_ctx = prompt_lock if prompt_lock else contextlib.nullcontext()
            with lock_ctx:
                # Restore original prompt first
                if os.path.exists(backup_prompt):
                    shutil.copy(backup_prompt, source_prompt)
                # Modify prompt if using principles
                if topkg > 0 or topka > 0:
                    global_top, agent_dict = extract_top_principles(retrieval_cache_data, topkg, topka)
                    create_modified_prompt(source_prompt, config["prompt_map"], global_top, agent_dict)
                # Copy to workdir (subprocess will use this copy)
                shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))
                # Restore source immediately after copy
                if os.path.exists(backup_prompt):
                    shutil.copy(backup_prompt, source_prompt)

            # Start subprocess outside lock (uses copied prompt.py in workdir)
            metrics_dir = os.path.join(workdir, "metrics")
            os.makedirs(metrics_dir, exist_ok=True)
            cmd = [sys.executable, main_script, "--workdir", workdir]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, cwd=agent_dir
            )

            timer = threading.Timer(7200, proc.kill)
            timer.start()
            stdout_lines = []
            try:
                for line in proc.stdout:
                    print(f"[{agent_name}:{task_id}] {line.rstrip()}", flush=True)
                    stdout_lines.append(line)
            finally:
                timer.cancel()
            proc.wait()

            if proc.returncode == -9:
                raise subprocess.TimeoutExpired(cmd, 7200)
            if proc.returncode == 0:
                corr = 0
                prefix = config["result_prefix"]
                for line in stdout_lines:
                    if line.startswith(prefix):
                        try:
                            corr = int(json.loads(line.split(prefix, 1)[1].strip()).get("_this_corr", 0))
                        except:
                            pass
                print(f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] SUCCESS corr={corr}", flush=True)
                # Append to unified result file
                unified_file = Path(output_base) / "all_results.jsonl"
                with open(unified_file, "a", encoding="utf-8") as uf:
                    uf.write(
                        json.dumps(
                            {
                                "bank_model": bank_model,
                                "topk": topk_tag,
                                "agent": agent_name,
                                "task_id": task_id,
                                "success": True,
                                "correct": corr,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                return task_id, True, corr
            print(
                f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] RETRY {attempt}/10 rc={proc.returncode}", flush=True
            )
        except subprocess.TimeoutExpired:
            print(f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] RETRY {attempt}/10 timeout", flush=True)
        except Exception as e:
            print(f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] RETRY {attempt}/10 error: {e}", flush=True)
        finally:
            if proc and proc.poll() is None:
                proc.kill()
                proc.wait()
    # Mark as failed to prevent re-execution
    print(f"[{bank_model}/{topk_tag}/{agent_name}:{task_id}] FAILED after 10 attempts", flush=True)
    # Append to unified result file
    unified_file = Path(output_base) / "all_results.jsonl"
    with open(unified_file, "a", encoding="utf-8") as uf:
        uf.write(
            json.dumps(
                {
                    "bank_model": bank_model,
                    "topk": topk_tag,
                    "agent": agent_name,
                    "task_id": task_id,
                    "success": False,
                    "correct": 0,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    return task_id, False, 0


def load_agent_tasks(agent_name: str, config: dict, agent_dir: Path) -> tuple[dict, str | None]:
    """Load tasks for an agent, return (tasks_dict, extra_dir)"""
    extra_dir = None
    if agent_name == "autogen_math":
        tasks = load_tasks_gsm8k()
    elif agent_name == "veadk_gaia":
        metadata_file = agent_dir / config["metadata_file"]
        tasks = load_tasks_metadata(str(metadata_file))
        extra_dir = str(metadata_file.parent)
    elif agent_name == "langgraph_sql":
        task_file = agent_dir / config["task_file"]
        tasks = load_tasks_json(str(task_file))
        extra_dir = str(task_file.parent)
    elif agent_name == "agno_rca":
        csv_file = agent_dir / config["csv_file"]
        tasks = load_tasks_csv(str(csv_file))
    elif agent_name == "adk_swe":
        tasks = load_tasks_swe()
    else:
        tasks = {}
    return tasks, extra_dir


def get_default_paths(agent_name: str, project_root: Path) -> dict:
    """Generate default paths for agent"""
    return {
        "test_dir": project_root / f"datasets/raw_split/{agent_name}/test",
        "output_base": project_root / "results/principle_run",
        "bank_path": project_root / "results/train_bank",
    }


def run_single_task_from_env(args) -> int:
    """Run single (model, dataset, task_id, topkg, topka) from env vars"""
    model_name = os.getenv(ENV_MODEL)
    dataset_name = os.getenv(ENV_DATASET)
    task_id = os.getenv(ENV_TASK_ID)
    topkg = int(os.getenv(ENV_TOPKG, "0"))
    topka = int(os.getenv(ENV_TOPKA, "0"))
    principle_scope = os.getenv(ENV_PRINCIPLE_SCOPE, "both")

    # Apply principle_scope: override topkg/topka based on scope
    if principle_scope == "global":
        topka = 0  # No agent level principles
    elif principle_scope == "agent":
        topkg = 0  # No global level principles

    if not model_name or not dataset_name or not task_id:
        print(f"[ERROR] Missing env: MODEL={model_name}, DATASET={dataset_name}, TASK_ID={task_id}", flush=True)
        return 1

    if dataset_name not in AGENT_CONFIGS:
        print(f"[ERROR] Unknown dataset: {dataset_name}", flush=True)
        return 1

    project_root = get_project_root()
    config = AGENT_CONFIGS[dataset_name]
    agent_dir = project_root / "agent_system" / config["dir"]
    tasks, extra_dir = load_agent_tasks(dataset_name, config, agent_dir)

    if task_id not in tasks:
        print(f"[ERROR] Task not found: {task_id}", flush=True)
        return 1

    task = tasks[task_id]
    bank_base = args.bank_path or str(project_root / "results/train_bank")
    bank_path = str(Path(bank_base) / model_name / dataset_name)
    output_base = args.output_base or str(project_root / "results/principle_run")

    # Determine workdir_root
    topk_name = f"g{topkg}_a{topka}"
    if topkg == 0 and topka == 0:
        workdir_root = os.path.join(output_base, "baseline", topk_name, dataset_name)
    else:
        workdir_root = os.path.join(output_base, model_name, topk_name, dataset_name)
    os.makedirs(workdir_root, exist_ok=True)

    # Ensure prompt backup
    source_prompt = agent_dir / "prompt.py"
    backup_prompt = agent_dir / "prompt.py.bak"
    if not backup_prompt.exists():
        shutil.copy(source_prompt, backup_prompt)

    # Retrieve principles
    cache_dir = Path(output_base) / model_name / "retrieval_cache" / dataset_name
    cache_file = cache_dir / "retrieval.jsonl"
    cache = RetrievalCache(str(cache_file))

    # Extract query
    info = task.get("info", task)
    query = ""
    for key in ["question", "Question", "task", "Task", "query", "Query", "instruction", "problem_statement"]:
        if key in info and info[key]:
            query = info[key]
            break

    cache_data = retrieve_principles_once(
        query,
        task_id,
        bank_path,
        config["prompt_map"],
        retrieval_cache=cache,
        use_rerank=True,
        top_e=20,
        top_l=20,
        rerank_model_name=model_name,
    )

    # Run task
    prompt_lock = threading.Lock()
    tid, ok, corr = run_single_task(
        task,
        config,
        dataset_name,
        workdir_root,
        str(agent_dir),
        prompt_lock,
        extra_dir,
        topkg,
        topka,
        bank_path,
        cache_data,
        output_base,
    )

    print(f"[SINGLE] {model_name}/{topk_name}/{dataset_name}:{task_id} -> ok={ok}, corr={corr}", flush=True)
    return 0 if ok else 1


# === Run Configurations (customize here) ===
# "autogen_math", "langgraph_sql", "veadk_gaia", "agno_rca", "adk_swe"
RUN_SYSTEMS = ["autogen_math", "langgraph_sql", "veadk_gaia"]  # [] = all
RUN_TOPK_LIST = [(0, 0), (5, 5), (10, 10)]  # (topkg, topka) pairs
RUN_BANK_MODELS = [
    "claude-sonnet-4.5",
    "deepseek-v3-2-251201",
    "doubao-seed-1-8-251228",
    "kimi-k2-250905",
    "gpt-5.2-chat",
]

# RUN_SYSTEMS = ["autogen_math", "langgraph_sql", "veadk_gaia"]  # [] = all
# RUN_TOPK_LIST = [(0, 0), (5, 5), (10, 10), (15, 15), (20, 20), (25, 25)]  # (topkg, topka) pairs
# RUN_BANK_MODELS = ["claude-sonnet-4.5", "deepseek-v3-2-251201", "doubao-seed-1-8-251228", "kimi-k2-250905", "gpt-5.2-chat"]


# === Main ===
def main():
    parser = argparse.ArgumentParser(description="Unified agent runner")
    parser.add_argument(
        "--mode",
        type=str,
        default="batch",
        choices=["debug", "test_batch", "single"],
        help="debug, test_batch, or single (from env)",
    )
    parser.add_argument(
        "--agent",
        type=str,
        nargs="+",
        default=["all"],
        choices=list(AGENT_CONFIGS.keys()) + ["all"],
        help="Agent(s) to run, or 'all' for all agents",
    )
    parser.add_argument("--task_id", type=str, default="")
    parser.add_argument("--topkg", type=int, default=None, help="Override topkg (debug mode)")
    parser.add_argument("--topka", type=int, default=None, help="Override topka (debug mode)")
    parser.add_argument("--test_dir", type=str, default="")
    parser.add_argument("--workers", type=int, default=5, help="Max parallel workers")
    parser.add_argument("--bank_path", type=str, default="")
    parser.add_argument("--output_base", type=str, default="")
    parser.add_argument("--principle_scope", choices=["global", "agent", "both"], default="both")

    load_dotenv()

    args = parser.parse_args()

    # Single mode: run from environment variables
    if args.mode == "single":
        sys.exit(run_single_task_from_env(args))

    # Expand 'all' to all agents
    all_agents = list(AGENT_CONFIGS.keys()) if "all" in args.agent else args.agent
    agents = [a for a in all_agents if not RUN_SYSTEMS or a in RUN_SYSTEMS]

    project_root = get_project_root()

    # Scan all model_names from bank_path
    bank_base = args.bank_path or str(project_root / "results/train_bank")
    # Determine bank models to use
    if RUN_BANK_MODELS:
        bank_models = RUN_BANK_MODELS
    else:
        bank_models = sorted([d.name for d in Path(bank_base).iterdir() if d.is_dir()])
    if not bank_models:
        print(f"No bank models found in {bank_base}")
        sys.exit(1)

    # Determine topk configs
    if args.topkg is not None and args.topka is not None:
        topk_configs = [(args.topkg, args.topka)]  # CLI override
    else:
        topk_configs = RUN_TOPK_LIST

    print(
        f"Agents: {agents}, Bank models: {bank_models}, TopK configs: {topk_configs}, Workers: {args.workers}",
        flush=True,
    )

    if args.mode == "debug":
        if len(agents) != 1:
            print("Debug mode requires exactly one agent")
            sys.exit(1)
        agent_name = agents[0]
        topkg, topka = topk_configs[0]
        bank_model = bank_models[0]
        config = AGENT_CONFIGS[agent_name]
        agent_dir = project_root / "agent_system" / config["dir"]
        tasks, extra_dir = load_agent_tasks(agent_name, config, agent_dir)
        # Set env for debug mode (runpy uses current process env)
        os.environ["OPENAI_BASE_URL"] = os.getenv("SEED_OPENAI_BASE_URL") or os.getenv("OPENAI_BASE_URL")
        os.environ["OPENAI_API_KEY"] = os.getenv("SEED_OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
        os.environ["OPENAI_MODEL"] = os.getenv("SEED_OPENAI_MODEL") or os.getenv("OPENAI_MODEL")
        if agent_name in ("agno_rca", "adk_swe") and "OPENAI_MODEL_NAME" not in os.environ:
            os.environ["OPENAI_MODEL_NAME"] = f"openai/{os.environ.get('OPENAI_MODEL', 'doubao-seed-1-8-251228')}"
        # Ensure prompt backup
        source_prompt = agent_dir / "prompt.py"
        backup_prompt = agent_dir / "prompt.py.bak"
        if not backup_prompt.exists():
            shutil.copy(source_prompt, backup_prompt)

        bank_path = str(Path(bank_base) / bank_model / agent_name)
        output_base = args.output_base or str(project_root / "results/principle_run")
        if topkg == 0 and topka == 0:
            workdir_root = os.path.join(output_base, "baseline", f"g{topkg}_a{topka}", agent_name)
        else:
            workdir_root = os.path.join(output_base, f"{bank_model}", f"g{topkg}_a{topka}", agent_name)
        os.makedirs(workdir_root, exist_ok=True)

        task = tasks.get(args.task_id) if args.task_id else next(iter(tasks.values()), None)
        if not task:
            print(f"No task found for {agent_name}:{args.task_id}")
            sys.exit(1)
        print(f"Debug: bank={bank_model}/{agent_name}:{task['task_id']}, workdir_root={workdir_root}", flush=True)

        if existing := find_existing_workdir(workdir_root, task["task_id"]):
            if load_completed_result(existing):
                print(f"Task already done: {existing}")
                sys.exit(0)
            shutil.rmtree(existing, ignore_errors=True)

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        workdir = os.path.join(workdir_root, f"{task['task_id']}_{timestamp}")
        query = prepare_task_workdir(task, workdir, agent_name, extra_dir)

        shutil.copy(backup_prompt, source_prompt)
        # Retrieve principles first
        cache_dir = Path(output_base) / bank_model / "retrieval_cache" / agent_name
        cache_file = cache_dir / "retrieval.jsonl"
        cache = RetrievalCache(str(cache_file))
        cache_data = retrieve_principles_once(
            query,
            task["task_id"],
            bank_path,
            config["prompt_map"],
            retrieval_cache=cache,
            use_rerank=True,
            top_e=20,
            top_l=20,
            rerank_model_name=bank_model,
        )
        global_top, agent_dict = extract_top_principles(cache_data, topkg, topka)
        create_modified_prompt(str(source_prompt), config["prompt_map"], global_top, agent_dict)
        shutil.copy(source_prompt, os.path.join(workdir, "prompt.py"))

        main_script = str(agent_dir / "main.py")
        sys.argv = [main_script, "--workdir", workdir]
        orig_dir = os.getcwd()
        try:
            os.chdir(agent_dir)
            runpy.run_path(main_script, run_name="__main__")
        finally:
            os.chdir(orig_dir)
            shutil.copy(backup_prompt, source_prompt)

    elif args.mode == "test_batch":
        # Phase 1: Retrieve all cases (once per bank_model/agent)
        retrieval_caches = {}  # (bank_model, agent_name) -> RetrievalCache
        retrieval_data = {}  # (bank_model, agent_name, task_id) -> cache_data

        print(f"\n{'='*60}\n[PHASE 1] Retrieving principles...", flush=True)
        output_base = args.output_base or str(project_root / "results/principle_run")
        for bank_model in bank_models:
            for agent_name in agents:
                config = AGENT_CONFIGS[agent_name]
                agent_dir = project_root / "agent_system" / config["dir"]
                tasks, extra_dir = load_agent_tasks(agent_name, config, agent_dir)
                paths = get_default_paths(agent_name, project_root)
                test_dir = args.test_dir or str(paths["test_dir"])
                test_task_ids = collect_task_ids_from_test_dir(test_dir)

                bank_path = str(Path(bank_base) / bank_model / agent_name)
                pb_file = Path(bank_path) / "principlebank" / "principlebank.jsonl"
                if not pb_file.exists():
                    print(f"[PHASE1 SKIP] {bank_model}/{agent_name}: {pb_file} not found", flush=True)
                    continue
                print(f"[PHASE1] Processing {bank_model}/{agent_name}", flush=True)

                # Initialize cache
                cache_dir = Path(output_base) / bank_model / "retrieval_cache" / agent_name
                cache_file = cache_dir / "retrieval.jsonl"
                cache = RetrievalCache(str(cache_file))
                retrieval_caches[(bank_model, agent_name)] = cache

                # Retrieve for each task
                for tid in test_task_ids:
                    if tid not in tasks:
                        continue
                    task = tasks[tid]
                    # Extract query
                    info = task.get("info", task)
                    query = ""
                    for key in [
                        "question",
                        "Question",
                        "task",
                        "Task",
                        "query",
                        "Query",
                        "instruction",
                        "problem_statement",
                    ]:
                        if key in info and info[key]:
                            query = info[key]
                            break

                    cache_data = retrieve_principles_once(
                        query,
                        tid,
                        bank_path,
                        config["prompt_map"],
                        retrieval_cache=cache,
                        use_rerank=True,
                        top_e=20,
                        top_l=20,
                        rerank_model_name=bank_model,
                    )
                    retrieval_data[(bank_model, agent_name, tid)] = cache_data

        print(f"[PHASE 1] Retrieved {len(retrieval_data)} mappings, cache_files={len(retrieval_caches)}", flush=True)
        print(f"{'='*60}\n", flush=True)
        # Build all tasks across (bank_model, topk, agent) combinations
        all_task_items = (
            []
        )  # [(agent_name, config, task, workdir_root, agent_dir, extra_dir, bank_path, topkg, topka), ...]
        agent_locks = {name: threading.Lock() for name in agents}
        task_stats = defaultdict(lambda: {"total": 0, "pending": 0})  # (bank_model, topk, agent) -> counts

        # Ensure prompt backups for all agents
        for agent_name in agents:
            config = AGENT_CONFIGS[agent_name]
            agent_dir = project_root / "agent_system" / config["dir"]
            source_prompt = agent_dir / "prompt.py"
            backup_prompt = agent_dir / "prompt.py.bak"
            if not backup_prompt.exists():
                shutil.copy(source_prompt, backup_prompt)

        baseline_done = False
        # Iterate: bank_model (serial) -> topk -> agent -> tasks
        for bank_model in bank_models:
            for topkg, topka in topk_configs:
                # Skip duplicate g0_a0 runs
                if topkg == 0 and topka == 0:
                    if baseline_done:
                        continue
                    baseline_done = True

                topk_name = f"g{topkg}_a{topka}"

                for agent_name in agents:
                    config = AGENT_CONFIGS[agent_name]
                    agent_dir = project_root / "agent_system" / config["dir"]
                    tasks, extra_dir = load_agent_tasks(agent_name, config, agent_dir)
                    paths = get_default_paths(agent_name, project_root)
                    test_dir = args.test_dir or str(paths["test_dir"])
                    test_task_ids = collect_task_ids_from_test_dir(test_dir)
                    batch_tasks = [tasks[tid] for tid in test_task_ids if tid in tasks]

                    bank_path = str(Path(bank_base) / bank_model / agent_name)
                    # Skip if no principlebank (except topk=0)
                    if (topkg > 0 or topka > 0) and not (
                        Path(bank_path) / "principlebank" / "principlebank.jsonl"
                    ).exists():
                        continue

                    output_base = args.output_base or str(project_root / "results/principle_run")
                    if topkg == 0 and topka == 0:
                        workdir_root = os.path.join(output_base, "baseline", topk_name, agent_name)
                    else:
                        workdir_root = os.path.join(output_base, bank_model, topk_name, agent_name)
                    os.makedirs(workdir_root, exist_ok=True)

                    # Collect pending tasks
                    pending_count = 0
                    for task in batch_tasks:
                        existing = find_existing_workdir(workdir_root, task["task_id"])
                        if existing and load_completed_result(existing):
                            continue  # Skip completed
                        if existing:
                            shutil.rmtree(existing, ignore_errors=True)
                        all_task_items.append(
                            (
                                agent_name,
                                config,
                                task,
                                workdir_root,
                                str(agent_dir),
                                extra_dir,
                                bank_path,
                                topkg,
                                topka,
                                retrieval_data.get((bank_model, agent_name, task["task_id"]), {}),
                            )
                        )
                        pending_count += 1
                    key = f"{bank_model}/{topk_name}/{agent_name}"
                    task_stats[key] = {"total": len(batch_tasks), "pending": pending_count}
        # Print collection summary
        print(f"\n{'='*60}\n[COLLECT] Task collection summary:", flush=True)
        for key, stats in sorted(task_stats.items()):
            if stats["pending"] > 0:
                print(f"  {key}: {stats['pending']}/{stats['total']} pending", flush=True)
        print(f"[COLLECT] Total: {len(all_task_items)} pending tasks, workers={args.workers}", flush=True)

        if not all_task_items:
            print("No tasks to run.")
            return

        # Group serial agent tasks for batch submission
        serial_agents = set()  # veadk now supports parallel
        parallel_limit_agents = {"veadk_gaia": 16}

        serial_tasks = defaultdict(list)  # agent_name -> [task_items]
        limited_parallel_tasks = defaultdict(list)  # agent_name -> [task_items]
        parallel_tasks = []
        for item in all_task_items:
            agent_name = item[0]
            if agent_name in serial_agents:
                serial_tasks[agent_name].append(item)
            elif agent_name in parallel_limit_agents:
                limited_parallel_tasks[agent_name].append(item)
            else:
                parallel_tasks.append(item)
        print(
            f"[DISPATCH] Serial: {list(serial_tasks.keys()) or 'none'}, "
            f"Limited parallel: {list(limited_parallel_tasks.keys()) or 'none'}, "
            f"Parallel tasks: {len(parallel_tasks)}",
            flush=True,
        )
        print(f"{'='*60}\n", flush=True)

        completed_count = 0
        total_count = len(all_task_items)
        all_results = defaultdict(dict)  # agent -> {tid: (ok, corr)}

        def run_serial_batch(agent_name, items):
            """Run serial agent tasks sequentially"""
            res = {}
            for i, (an, cfg, task, wr, ad, ed, bp, tkg, tka, cache_data) in enumerate(items):
                print(f"[{agent_name}] Serial progress: {i+1}/{len(items)}", flush=True)
                tid, ok, corr = run_single_task(
                    task, cfg, an, wr, ad, agent_locks[an], ed, tkg, tka, bp, cache_data, output_base
                )
                res[tid] = (ok, corr)
            return res

        def run_limited_parallel_batch(agent_name, items, max_workers):
            """Run limited-parallel agent tasks with bounded concurrency"""
            res = {}
            print(f"[{agent_name}] Limited parallel: {len(items)} tasks, max_workers={max_workers}", flush=True)
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futs = {}
                for an, cfg, task, wr, ad, ed, bp, tkg, tka, cache_data in items:
                    f = ex.submit(
                        run_single_task,
                        task,
                        cfg,
                        an,
                        wr,
                        ad,
                        agent_locks[an],
                        ed,
                        tkg,
                        tka,
                        bp,
                        cache_data,
                        output_base,
                    )
                    futs[f] = task["task_id"]
                for i, f in enumerate(as_completed(futs), 1):
                    tid = futs[f]
                    _, ok, corr = f.result()
                    res[tid] = (ok, corr)
                    if i % 10 == 0:
                        print(f"[{agent_name}] Progress: {i}/{len(items)}", flush=True)
            return res

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futs = {}
            # Submit serial agents as single batch jobs
            for agent_name, items in serial_tasks.items():
                f = executor.submit(run_serial_batch, agent_name, items)
                futs[f] = (agent_name, True, len(items))
            # Submit limited-parallel agents as batch jobs
            for agent_name, items in limited_parallel_tasks.items():
                max_w = min(args.workers, parallel_limit_agents.get(agent_name, args.workers))
                f = executor.submit(run_limited_parallel_batch, agent_name, items, max_w)
                futs[f] = (agent_name, True, len(items))
            # Submit parallel tasks individually
            for item in parallel_tasks:
                agent_name, cfg, task, wr, ad, ed, bp, tkg, tka, cache_data = item
                f = executor.submit(
                    run_single_task,
                    task,
                    cfg,
                    agent_name,
                    wr,
                    ad,
                    agent_locks[agent_name],
                    ed,
                    tkg,
                    tka,
                    bp,
                    cache_data,
                    output_base,
                )
                futs[f] = (agent_name, False, 1)

            for f in as_completed(futs):
                agent_name, is_batch, count = futs[f]
                completed_count += count
                if is_batch:
                    all_results[agent_name].update(f.result())
                else:
                    tid, ok, corr = f.result()
                    all_results[agent_name][tid] = (ok, corr)
                # Progress log every 10 tasks or batch complete
                if is_batch or completed_count % 10 == 0:
                    print(f"[PROGRESS] {completed_count}/{total_count} tasks completed", flush=True)
        # Summary
        print(f"\n{'='*60}")
        print("[SUMMARY]")
        for agent_name in agents:
            results = all_results.get(agent_name, {})
            if results:
                s = sum(1 for ok, _ in results.values() if ok)
                c = sum(corr for _, corr in results.values())
                print(f"  [{agent_name}] Success: {s}/{len(results)}, Correct: {c}/{len(results)}")
        print("All configurations completed.")


if __name__ == "__main__":
    main()
