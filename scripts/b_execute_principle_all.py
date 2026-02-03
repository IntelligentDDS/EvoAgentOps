# run_all.py
import os
import sys
import subprocess
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from collections import defaultdict
from dotenv import load_dotenv
import shutil
import json

# Environment variable keys (must match main.py)
ENV_MODEL = "EXEC_MODEL"
ENV_DATASET = "EXEC_DATASET"
ENV_TASK_ID = "EXEC_TASK_ID"
ENV_TOPKG = "EXEC_TOPKG"
ENV_TOPKA = "EXEC_TOPKA"
ENV_PRINCIPLE_SCOPE = "EXEC_PRINCIPLE_SCOPE"


# Files/dirs to keep in workdir
KEEP_ITEMS = {"qa.json", "prompt.py", "output.json", "trace", "metrics", "logs", "retrieval.jsonl"}

# Serial agents: only ONE task globally at a time (across all models)
SERIAL_AGENTS = set()  # veadk now supports parallel

# Parallel limit for specific agents (e.g., browser pool limit)
PARALLEL_LIMIT_AGENTS = {"veadk_gaia": 16}

# Global lock for serial agents (not used for veadk anymore)
veadk_global_lock = threading.Lock()

# Semaphores for parallel-limited agents
_agent_semaphores = {}


def get_agent_semaphore(dataset: str) -> threading.Semaphore | None:
    if dataset not in PARALLEL_LIMIT_AGENTS:
        return None
    if dataset not in _agent_semaphores:
        _agent_semaphores[dataset] = threading.Semaphore(PARALLEL_LIMIT_AGENTS[dataset])
    return _agent_semaphores[dataset]


def get_project_root():
    return Path(__file__).resolve().parent.parent


def extract_task_id(folder_name: str) -> str:
    parts = folder_name.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit() and len(parts[1]) >= 14:
        return parts[0]
    return folder_name


def is_task_done(output_dir: str, model: str, topkg: int, topka: int, dataset: str, task_id: str) -> bool:
    """Check if task completed by finding workdir with output.json"""
    topk_name = f"g{topkg}_a{topka}"
    if topkg == 0 and topka == 0:
        workdir_root = Path(output_dir) / "baseline" / topk_name / dataset
    else:
        workdir_root = Path(output_dir) / model / topk_name / dataset

    if not workdir_root.exists():
        return False

    for d in workdir_root.iterdir():
        if d.is_dir() and extract_task_id(d.name) == task_id:
            output_file = d / "output.json"
            if output_file.exists() and output_file.stat().st_size > 10:
                return True
    return False


def cleanup_workdir(workdir: Path):
    """Remove files/dirs not in KEEP_ITEMS"""
    if not workdir.is_dir():
        return
    for item in workdir.iterdir():
        if item.name not in KEEP_ITEMS:
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)


def _score_workdir(d: Path) -> tuple:
    check_items = {"qa.json", "prompt.py", "output.json", "trace", "metrics", "logs"}
    keep_count = sum(1 for item in check_items if (d / item).exists())
    is_fail = False
    try:
        data = json.loads((d / "output.json").read_text())
        is_fail = data.get("status") == "fail"
    except:
        pass
    return (keep_count, is_fail, d.name)


def _dedup_workdirs(workdir_root: Path) -> int:
    """Remove duplicate workdirs for same task_id, keep best one."""
    from collections import defaultdict

    groups = defaultdict(list)
    for d in workdir_root.iterdir():
        if d.is_dir():
            groups[extract_task_id(d.name)].append(d)
    removed = 0
    for tid, dirs in groups.items():
        if len(dirs) > 1:
            dirs.sort(key=_score_workdir, reverse=True)
            print(f"[DEDUP] {tid}: keep {dirs[0].name}, remove {len(dirs)-1}", flush=True)
            for d in dirs[1:]:
                shutil.rmtree(d, ignore_errors=True)
                removed += 1
    return removed


def scan_and_cleanup(output_dir: str, models: list, datasets: list, topk_list: list, principle_scope: str):
    """Scan output_dir, remove incomplete workdirs, cleanup completed ones"""
    output_path = Path(output_dir)
    if not output_path.exists():
        return

    removed, cleaned, deduped = 0, 0, 0
    dirs_to_scan = set()
    for topkg, topka in topk_list:
        eff_topkg, eff_topka = apply_principle_scope(topkg, topka, principle_scope)
        topk_name = f"g{eff_topkg}_a{eff_topka}"
        for dataset in datasets:
            if eff_topkg == 0 and eff_topka == 0:
                dirs_to_scan.add(output_path / "baseline" / topk_name / dataset)
            else:
                for model in models:
                    dirs_to_scan.add(output_path / model / topk_name / dataset)

    for workdir_root in dirs_to_scan:
        if not workdir_root.exists():
            continue
        deduped += _dedup_workdirs(workdir_root)  # dedup first
        for workdir in workdir_root.iterdir():
            if not workdir.is_dir():
                continue
            output_file = workdir / "output.json"
            if not output_file.exists() or output_file.stat().st_size < 10:
                shutil.rmtree(workdir, ignore_errors=True)
                removed += 1
            else:
                cleanup_workdir(workdir)
                cleaned += 1

    print(f"[CLEANUP] Deduped {deduped}, removed {removed} incomplete, cleaned {cleaned} workdirs", flush=True)


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


def apply_principle_scope(topkg: int, topka: int, principle_scope: str) -> tuple:
    """Apply principle_scope to adjust topkg/topka values"""
    if principle_scope == "global":
        return topkg, 0  # No agent level
    elif principle_scope == "agent":
        return 0, topka  # No global level
    return topkg, topka  # both


def run_single_task(
    model: str,
    dataset: str,
    task_id: str,
    topkg: int,
    topka: int,
    base_args: list,
    output_dir: str,
    principle_scope: str = "both",
) -> dict:
    """Run main.py --mode single with env vars. Returns result dict."""
    # Apply principle_scope to get effective topk values
    eff_topkg, eff_topka = apply_principle_scope(topkg, topka, principle_scope)
    topk_name = f"g{eff_topkg}_a{eff_topka}"
    full_task_id = f"{model}/{topk_name}/{dataset}:{task_id}"

    # Skip if already done
    if is_task_done(output_dir, model, eff_topkg, eff_topka, dataset, task_id):
        print(f"[SKIP] {full_task_id} (already done)", flush=True)
        return {"task_id": full_task_id, "success": True, "skipped": True}

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env[ENV_MODEL] = model
    env[ENV_DATASET] = dataset
    env[ENV_TASK_ID] = task_id
    env[ENV_TOPKG] = str(topkg)
    env[ENV_TOPKA] = str(topka)
    env[ENV_PRINCIPLE_SCOPE] = principle_scope

    cmd = [sys.executable, "b_execute_principle.py", "--mode", "single"] + base_args
    print(f"[START] {full_task_id}", flush=True)

    # Use global lock for serial agents, or semaphore for parallel-limited agents
    lock = veadk_global_lock if dataset in SERIAL_AGENTS else None
    sem = get_agent_semaphore(dataset)

    try:
        if lock:
            lock.acquire()
        if sem:
            sem.acquire()
        result = subprocess.run(cmd, env=env, timeout=7200)
        success = result.returncode == 0
        print(f"[{'DONE' if success else 'FAIL'}] {full_task_id}", flush=True)
        return {"task_id": full_task_id, "success": success, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {full_task_id}", flush=True)
        return {"task_id": full_task_id, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"[ERROR] {full_task_id}: {e}", flush=True)
        return {"task_id": full_task_id, "success": False, "error": str(e)}
    finally:
        if lock and lock.locked():
            lock.release()
        if sem:
            sem.release()


def run_dataset_tasks(
    model: str,
    dataset: str,
    task_ids: list,
    topk_list: list,
    base_args: list,
    output_dir: str,
    principle_scope: str = "both",
    workers: int = 1,
) -> list:
    """Run all tasks for one (model, dataset). Parallel for limited agents, serial for others."""
    results = []
    total = len(task_ids) * len(topk_list)

    # Build all task combinations
    all_tasks = [(topkg, topka, task_id) for topkg, topka in topk_list for task_id in task_ids]

    # Parallel execution, apply limit for specific agents
    max_workers = min(total, workers, PARALLEL_LIMIT_AGENTS.get(dataset, workers))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_task, model, dataset, task_id, topkg, topka, base_args, output_dir, principle_scope
            ): (topkg, topka, task_id)
            for topkg, topka, task_id in all_tasks
        }
        for idx, future in enumerate(as_completed(futures), 1):
            topkg, topka, task_id = futures[future]
            print(f"[{model}/{dataset}] Done {idx}/{total}: g{topkg}_a{topka}/{task_id}", flush=True)
            results.append(future.result())
    return results


def run_all_tasks_for_dataset(
    dataset: str,
    models: list,
    task_ids: list,
    topk_list: list,
    base_args: list,
    output_dir: str,
    principle_scope: str,
    workers: int,
) -> list:
    """Run all tasks for one dataset across all models. Models interleaved for fair scheduling."""
    # Build tasks: baseline (0,0) only needs one model (shared dir), others need all models
    all_tasks = []
    for topkg, topka in topk_list:
        eff_topkg, eff_topka = apply_principle_scope(topkg, topka, principle_scope)
        for task_id in task_ids:
            if eff_topkg == 0 and eff_topka == 0:
                all_tasks.append((models[0], topkg, topka, task_id))  # baseline: one model only
            else:
                for model in models:
                    all_tasks.append((model, topkg, topka, task_id))
    total = len(all_tasks)
    max_workers = min(total, workers, PARALLEL_LIMIT_AGENTS.get(dataset, workers))
    print(f"[{dataset}] START: {total} tasks, max_workers={max_workers}", flush=True)

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                run_single_task, model, dataset, task_id, topkg, topka, base_args, output_dir, principle_scope
            ): (model, topkg, topka, task_id)
            for model, topkg, topka, task_id in all_tasks
        }
        for idx, future in enumerate(as_completed(futures), 1):
            model, topkg, topka, task_id = futures[future]
            if idx % 20 == 0 or idx == total:
                print(f"[{dataset}] Progress: {idx}/{total}", flush=True)
            results.append(future.result())

    ok = sum(1 for r in results if r.get("success"))
    print(f"[{dataset}] DONE: {ok}/{total} success", flush=True)
    return results


def run_model_datasets(
    model: str,
    datasets_tasks: dict,
    topk_list: list,
    base_args: list,
    output_dir: str,
    dataset_workers: int,
    principle_scope: str = "both",
) -> list:
    """Run all datasets for one model. Non-serial datasets parallel, serial ones use global lock."""
    print(f"[MODEL] {model} START: {len(datasets_tasks)} datasets", flush=True)
    all_results = []

    # Separate serial and parallel datasets
    serial_items = [(d, t) for d, t in datasets_tasks.items() if d in SERIAL_AGENTS]
    parallel_items = [(d, t) for d, t in datasets_tasks.items() if d not in SERIAL_AGENTS]

    with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
        futures = {}

        # Submit parallel datasets
        for dataset, task_ids in parallel_items:
            f = executor.submit(
                run_dataset_tasks,
                model,
                dataset,
                task_ids,
                topk_list,
                base_args,
                output_dir,
                principle_scope,
                dataset_workers,
            )
            futures[f] = dataset

        # Submit serial datasets (will use global lock internally)
        for dataset, task_ids in serial_items:
            f = executor.submit(
                run_dataset_tasks,
                model,
                dataset,
                task_ids,
                topk_list,
                base_args,
                output_dir,
                principle_scope,
                dataset_workers,
            )
            futures[f] = dataset

        done_count = 0
        for future in as_completed(futures):
            dataset = futures[future]
            done_count += 1
            try:
                results = future.result()
                all_results.extend(results)
                ok = sum(1 for r in results if r.get("success"))
                print(f"[MODEL] {model} [{done_count}/{len(futures)}] {dataset}: {ok}/{len(results)} ok", flush=True)
            except Exception as e:
                print(f"[MODEL] {model} [{done_count}/{len(futures)}] {dataset} EXCEPTION: {e}", flush=True)
                all_results.append({"task_id": f"{model}/*/{dataset}", "success": False, "error": str(e)})

    ok = sum(1 for r in all_results if r.get("success"))
    print(f"[MODEL] {model} DONE: {ok}/{len(all_results)} success", flush=True)
    return all_results


# Default configs
DEFAULT_MODELS = [
    "gpt-5.2-chat",
    "claude-sonnet-4.5",
    "kimi-k2-250905",
    "doubao-seed-1-8-251228",
    "deepseek-v3-2-251201",
]
# ["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"]
DEFAULT_DATASETS = ["autogen_math", "langgraph_sql", "veadk_gaia"]
DEFAULT_TOPK = [(0, 0), (5, 5), (10, 10)]
DEFAULT_PRINCIPLE_SCOPE = "both"  # "global", "agent", or "both"


def main():
    parser = argparse.ArgumentParser(description="Batch scheduler for execute tasks")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument(
        "--topk", nargs="+", type=str, default=None, help="TopK configs as 'g,a' pairs, e.g. '0,0' '5,5' '10,10'"
    )
    parser.add_argument("--dataset_workers", type=int, default=1, help="Parallel datasets per model")
    parser.add_argument("--bank_path", default="../results/train_bank")
    parser.add_argument("--output_dir", default="../results/principle_run")
    parser.add_argument("--test_dir_base", default="../datasets/raw_split")
    parser.add_argument("--principle_scope", choices=["global", "agent", "both"], default="both")

    args = parser.parse_args()

    # Parse topk configs
    if args.topk:
        topk_list = [tuple(map(int, t.split(","))) for t in args.topk]
    else:
        topk_list = DEFAULT_TOPK

    # Build base args for main.py
    base_args = []
    if args.bank_path:
        base_args.extend(["--bank_path", args.bank_path])
    if args.output_dir:
        base_args.extend(["--output_base", args.output_dir])
    if args.principle_scope != "both":
        base_args.extend(["--principle_scope", args.principle_scope])

    load_dotenv()
    project_root = get_project_root()

    # Collect tasks for each dataset
    all_datasets_tasks = {}  # dataset -> [task_ids]
    for dataset in args.datasets:
        test_dir = Path(args.test_dir_base) / dataset / "test"
        if test_dir.exists():
            task_ids = collect_task_ids_from_test_dir(str(test_dir))
            if task_ids:
                all_datasets_tasks[dataset] = task_ids

    total_tasks = sum(len(t) for t in all_datasets_tasks.values()) * len(args.models) * len(topk_list)
    # Cleanup first: remove incomplete workdirs, clean extra files in completed ones
    scan_and_cleanup(args.output_dir, args.models, args.datasets, topk_list, args.principle_scope)

    # Count pending after cleanup
    pending = 0
    baseline_counted = set()
    for m in args.models:
        for d, task_ids in all_datasets_tasks.items():
            for topkg, topka in topk_list:
                eff_topkg, eff_topka = apply_principle_scope(topkg, topka, args.principle_scope)
                for tid in task_ids:
                    if eff_topkg == 0 and eff_topka == 0:
                        if (d, tid) in baseline_counted:
                            continue
                        baseline_counted.add((d, tid))
                    if not is_task_done(args.output_dir, m, eff_topkg, eff_topka, d, tid):
                        pending += 1

    print(f"{'='*60}", flush=True)
    print(f"[BATCH] Total={total_tasks}, Pending={pending}, Done={total_tasks-pending}", flush=True)
    print(f"[BATCH] Models: {args.models}", flush=True)
    print(f"[BATCH] Datasets: {list(all_datasets_tasks.keys())}", flush=True)
    print(f"[BATCH] TopK: {topk_list}, dataset_workers={args.dataset_workers}", flush=True)
    print(f"[BATCH] Serial agents (global lock): {SERIAL_AGENTS}", flush=True)
    print(f"{'='*60}", flush=True)

    if pending == 0:
        print("[BATCH] All tasks already done!", flush=True)
        return

    # Parallel execution: one thread per dataset, each with workers tasks
    results = []
    with ThreadPoolExecutor(max_workers=len(all_datasets_tasks)) as executor:
        futures = {
            executor.submit(
                run_all_tasks_for_dataset,
                dataset,
                args.models,
                task_ids,
                topk_list,
                base_args,
                args.output_dir,
                args.principle_scope,
                args.dataset_workers,
            ): dataset
            for dataset, task_ids in all_datasets_tasks.items()
        }
        done_count = 0
        for future in as_completed(futures):
            dataset = futures[future]
            done_count += 1
            try:
                dataset_results = future.result()
                results.extend(dataset_results)
                ok = sum(1 for r in dataset_results if r.get("success"))
                print(
                    f"[BATCH] Dataset {done_count}/{len(futures)} {dataset}: {ok}/{len(dataset_results)} success",
                    flush=True,
                )
            except Exception as e:
                print(f"[BATCH] Dataset {done_count}/{len(futures)} {dataset} EXCEPTION: {e}", flush=True)
                results.append({"task_id": f"*/*/{dataset}", "success": False, "error": str(e)})

    # Summary
    success = sum(1 for r in results if r.get("success"))
    skipped = sum(1 for r in results if r.get("skipped"))
    failed = len(results) - success
    print(f"{'='*60}", flush=True)
    print(f"[SUMMARY] Total={len(results)}, Success={success} (skipped={skipped}), Failed={failed}", flush=True)
    if failed:
        print("[FAILED TASKS]", flush=True)
        for r in results:
            if not r.get("success"):
                print(f"  {r['task_id']}: {r.get('error', r.get('returncode'))}", flush=True)


if __name__ == "__main__":
    main()
