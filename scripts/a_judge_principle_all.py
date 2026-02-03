# run_all.py
import os
import sys
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
import json

# Environment variable keys (must match main.py)
ENV_MODEL = "JUDGE_MODEL"
ENV_DATASET = "JUDGE_DATASET"
ENV_N = "JUDGE_N"
ENV_TOP_K = "JUDGE_TOP_K"
ENV_USE_PRINCIPLES = "JUDGE_USE_PRINCIPLES"
ENV_PRINCIPLE_SCOPE = "JUDGE_PRINCIPLE_SCOPE"


def is_task_done(output_dir: str, model: str, n: int, top_k: int, dataset: str, fault_dir: str) -> bool:
    """Check if task is completed by reading judge_results.jsonl"""
    config_name = f"n{n}_topk{top_k}" if top_k > 0 else f"n{n}"
    result_file = Path(output_dir) / model / config_name / dataset / "judge_results.jsonl"
    fault_path = Path(fault_dir) / dataset / "test" / "fail"

    # Get all case_ids from fault_dir
    if not fault_path.exists():
        return True  # No cases to process
    all_cases = {f.name for f in fault_path.iterdir() if f.is_dir()}
    if not all_cases:
        return True

    # Get done case_ids from judge_results.jsonl
    done_cases = set()
    if result_file.exists():
        try:
            with open(result_file, encoding="utf-8") as f:
                for line in f:
                    try:
                        done_cases.add(json.loads(line).get("case_id", ""))
                    except:
                        pass
        except:
            pass

    return len(done_cases) >= len(all_cases)


def run_single_task(
    model: str, dataset: str, n: int, top_k: int, base_args: list, output_dir: str, principle_scope: str = "both"
) -> dict:
    """Run main.py --mode single with env vars. Returns result dict."""
    use_principles = top_k > 0
    config_name = f"n{n}_topk{top_k}" if use_principles else f"n{n}"
    task_id = f"{model}/{config_name}/{dataset}"

    # Skip if already done
    if is_task_done(
        output_dir,
        model,
        n,
        top_k,
        dataset,
        base_args[base_args.index("--fault_dir") + 1] if "--fault_dir" in base_args else "../datasets/fault_injected",
    ):
        print(f"[SKIP] {task_id} (already done)", flush=True)
        return {"task_id": task_id, "success": True, "skipped": True}

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env[ENV_MODEL] = model
    env[ENV_DATASET] = dataset
    env[ENV_N] = str(n)
    env[ENV_TOP_K] = str(top_k)
    env[ENV_USE_PRINCIPLES] = "true" if use_principles else "false"
    env[ENV_PRINCIPLE_SCOPE] = principle_scope

    cmd = [sys.executable, "a_judge_principle.py", "--mode", "single"] + base_args
    print(f"[START] {task_id}", flush=True)

    try:
        result = subprocess.run(cmd, env=env, timeout=7200)
        success = result.returncode == 0
        print(f"[{'DONE' if success else 'FAIL'}] {task_id}", flush=True)
        return {"task_id": task_id, "success": success, "returncode": result.returncode}
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {task_id}", flush=True)
        return {"task_id": task_id, "success": False, "error": "timeout"}
    except Exception as e:
        print(f"[ERROR] {task_id}: {e}", flush=True)
        return {"task_id": task_id, "success": False, "error": str(e)}


def run_dataset_configs(
    model: str,
    dataset: str,
    n_list: list,
    top_k_list: list,
    base_args: list,
    output_dir: str,
    principle_scope: str = "both",
) -> list:
    """Run all (n, top_k) configs for one (model, dataset) serially. Returns list of results."""
    results = []
    total = len(n_list) * len(top_k_list)
    for i, n in enumerate(n_list):
        for j, top_k in enumerate(top_k_list):
            idx = i * len(top_k_list) + j + 1
            print(f"[{model}/{dataset}] Config {idx}/{total}: n={n}, top_k={top_k}", flush=True)
            r = run_single_task(model, dataset, n, top_k, base_args, output_dir, principle_scope)
            results.append(r)
    return results


def run_model_datasets(
    model: str,
    datasets: list,
    n_list: list,
    top_k_list: list,
    base_args: list,
    output_dir: str,
    dataset_workers: int,
    principle_scope: str = "both",
) -> list:
    """Run all datasets for one model. Datasets parallel, configs serial within each."""
    print(f"[MODEL] {model} START: {len(datasets)} datasets, {dataset_workers} workers", flush=True)
    all_results = []
    with ThreadPoolExecutor(max_workers=dataset_workers) as executor:
        futures = {
            executor.submit(
                run_dataset_configs, model, d, n_list, top_k_list, base_args, output_dir, principle_scope
            ): d
            for d in datasets
        }
        done_count = 0
        for future in as_completed(futures):
            dataset = futures[future]
            done_count += 1
            try:
                results = future.result()
                all_results.extend(results)
                ok = sum(1 for r in results if r.get("success"))
                print(f"[MODEL] {model} [{done_count}/{len(datasets)}] {dataset}: {ok}/{len(results)} ok", flush=True)
            except Exception as e:
                print(f"[MODEL] {model} [{done_count}/{len(datasets)}] {dataset} EXCEPTION: {e}", flush=True)
                all_results.append({"task_id": f"{model}/*/{dataset}", "success": False, "error": str(e)})
    ok = sum(1 for r in all_results if r.get("success"))
    print(f"[MODEL] {model} DONE: {ok}/{len(all_results)} success", flush=True)
    return all_results


# principle_scope
# top_k

# Default configs
DEFAULT_MODELS = [
    # "gpt-5.2-chat",
    "claude-sonnet-4.5",
    # "kimi-k2-250905",
    # "doubao-seed-1-8-251228",
    # "deepseek-v3-2-251201",
]
DEFAULT_DATASETS = ["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"]
DEFAULT_N = [1]
DEFAULT_TOP_K = [3, 8, 15, 20]  # 0 means no principles


def main():
    parser = argparse.ArgumentParser(description="Batch scheduler for judge tasks")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--n", type=int, nargs="+", default=DEFAULT_N)
    parser.add_argument("--top_k", type=int, nargs="+", default=DEFAULT_TOP_K)
    parser.add_argument("--dataset_workers", type=int, default=10, help="Parallel datasets per model")
    parser.add_argument("--principle_scope", choices=["global", "agent", "both"], default="both")
    parser.add_argument("--parsed_dir", default="../datasets/parsed")
    parser.add_argument("--fault_dir", default="../datasets/fault_injected")
    parser.add_argument("--principlebank_dir", default="../results/train_bank")
    parser.add_argument("--output_dir", default="../results/principle_judge")
    parser.add_argument("--llm_max_concurrency", type=int, default=None)

    args = parser.parse_args()

    # Build base args for main.py
    base_args = []
    for key in ["parsed_dir", "fault_dir", "principlebank_dir", "output_dir"]:
        if val := getattr(args, key, None):
            base_args.extend([f"--{key}", val])
    if args.llm_max_concurrency:
        base_args.extend(["--llm_max_concurrency", str(args.llm_max_concurrency)])
    if args.principle_scope != "both":
        base_args.extend(["--principle_scope", args.principle_scope])

    load_dotenv()

    # Count total tasks and pending
    total_tasks = len(args.models) * len(args.datasets) * len(args.n) * len(args.top_k)
    pending = 0
    for m in args.models:
        for d in args.datasets:
            for n in args.n:
                for k in args.top_k:
                    if not is_task_done(args.output_dir, m, n, k, d, args.fault_dir):
                        pending += 1
    print(f"{'='*60}", flush=True)
    print(f"[BATCH] Total={total_tasks}, Pending={pending}, Done={total_tasks-pending}", flush=True)
    print(f"[BATCH] Models: {args.models}", flush=True)
    print(f"[BATCH] Datasets: {args.datasets}", flush=True)
    print(f"[BATCH] n={args.n}, top_k={args.top_k}, dataset_workers={args.dataset_workers}", flush=True)
    print(f"{'='*60}", flush=True)
    if pending == 0:
        print("[BATCH] All tasks already done!", flush=True)
        return

    # Parallel execution
    results = []
    with ThreadPoolExecutor(max_workers=len(args.models)) as executor:
        futures = {
            executor.submit(
                run_model_datasets,
                model,
                args.datasets,
                args.n,
                args.top_k,
                base_args,
                args.output_dir,
                args.dataset_workers,
                args.principle_scope,
            ): model
            for model in args.models
        }
        model_done = 0
        for future in as_completed(futures):
            model = futures[future]
            model_done += 1
            try:
                model_results = future.result()
                results.extend(model_results)
                ok = sum(1 for r in model_results if r.get("success"))
                print(
                    f"[BATCH] Model {model_done}/{len(args.models)} {model}: {ok}/{len(model_results)} success",
                    flush=True,
                )
            except Exception as e:
                print(f"[BATCH] Model {model_done}/{len(args.models)} {model} EXCEPTION: {e}", flush=True)
                results.append({"task_id": f"{model}/*", "success": False, "error": str(e)})

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
