# compare_who_all.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from evoagentops.util import init_logger, logger
from compare_who import run_single_case, load_case_data, build_who_format_data

# Model config mapping for batch execution
MODEL_CONFIG_MAP = {
    "deepseek": "DEEPSEEK",
    "gpt": "GPT",
    "seed": "SEED",
    "doubao": "SEED",
    "kimi": "KIMI",
    "claude": "CLAUDE",
}


def get_model_config(model_name: str) -> tuple:
    """Get (base_url, api_key, model) from env vars based on model_name keyword"""
    model_lower = model_name.lower() if model_name else ""
    for keyword, prefix in MODEL_CONFIG_MAP.items():
        if keyword in model_lower:
            return (
                os.getenv(f"{prefix}_OPENAI_BASE_URL"),
                os.getenv(f"{prefix}_OPENAI_API_KEY"),
                os.getenv(f"{prefix}_OPENAI_MODEL"),
            )
    return os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")


def get_all_cases(fault_dir: Path, dataset: str) -> list:
    """Get all case directories for a dataset"""
    dataset_dir = fault_dir / dataset / "test" / "fail"
    if not dataset_dir.exists():
        return []
    cases = []
    for folder in sorted(dataset_dir.iterdir()):
        if folder.is_dir() and (folder / "agent_steps.json").exists() and (folder / "label.json").exists():
            cases.append(folder)
    return cases


def get_pending_cases(cases: list, output_base: Path, method: str, dataset: str, model: str) -> list:
    pending = []
    for case_dir in cases:
        output_dir = output_base / method / model / dataset / case_dir.name
        output_file = output_dir / "output.json"
        if output_file.exists():
            try:
                with open(output_file, encoding="utf-8") as f:
                    data = json.load(f)
                    # Validate required fields exist
                    if data.get("case_id") and "predicted_step" in data:
                        continue  # Valid result, skip
            except Exception as e:
                logger.warning(f"Invalid output {output_file}: {e}")
        pending.append(case_dir)
    return pending


def ensure_data_json(cases: list, output_base: Path, method: str, dataset: str, model: str):
    """Generate data"""
    count = 0
    for case_dir in cases:
        output_dir = output_base / method / model / dataset / case_dir.name
        output_file = output_dir / "output.json"
        data_file = output_dir / "data.json"
        old_steps_file = output_dir / "steps.json"
        # Remove old steps.json if exists
        if old_steps_file.exists():
            old_steps_file.unlink()
        # Generate/overwrite data.json if output exists
        if output_file.exists():
            try:
                data = load_case_data(case_dir)
                with open(data_file, "w", encoding="utf-8") as f:
                    json.dump(build_who_format_data(data, case_dir.name), f, ensure_ascii=False, indent=2)
                # Fix output.json: remove deprecated label field
                with open(output_file, encoding="utf-8") as f:
                    output_data = json.load(f)
                if "label" in output_data:
                    del output_data["label"]
                    with open(output_file, "w", encoding="utf-8") as f:
                        json.dump(output_data, f, ensure_ascii=False, indent=2)
                count += 1
            except Exception as e:
                logger.warning(f"Failed to update files for {case_dir.name}: {e}")
    if count > 0:
        logger.info(f"[{method}/{model}/{dataset}] Updated {count} data.json files")


def load_completed_results(output_base: Path, method: str, dataset: str, model: str) -> list:
    """Load all completed results for a method/dataset/model combination"""
    results = []
    output_dir = output_base / method / model / dataset
    if not output_dir.exists():
        return results
    for case_folder in output_dir.iterdir():
        if case_folder.is_dir():
            output_file = case_folder / "output.json"
            if output_file.exists():
                try:
                    with open(output_file, encoding="utf-8") as f:
                        results.append(json.load(f))
                except:
                    pass
    return results


async def _run_case(
    method: str,
    case_dir: Path,
    output_base: Path,
    dataset: str,
    model: str,
    model_config: tuple,
) -> dict:
    """Process single case, return result dict"""
    output_dir = output_base / method / model / dataset / case_dir.name
    try:
        base_url, api_key, model_name = model_config
        result = await run_single_case(method, case_dir, output_dir, base_url, api_key, model_name)
        return {"success": True, "case_id": case_dir.name, "result": result}
    except Exception as e:
        logger.error(f"[{method}] Failed: {case_dir.name}: {e}")
        return {"success": False, "case_id": case_dir.name, "error": str(e)}


async def process_dataset_method(
    method: str,
    dataset: str,
    model: str,
    fault_dir: Path,
    output_base: Path,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process all cases for one (method, dataset, model) combination"""
    cases = get_all_cases(fault_dir, dataset)
    if not cases:
        logger.warning(f"[{method}/{model}/{dataset}] No cases found")
        return {"total": 0, "pending": 0, "success": 0, "failed": 0}

    ensure_data_json(cases, output_base, method, dataset, model)

    pending = get_pending_cases(cases, output_base, method, dataset, model)
    done_count = len(cases) - len(pending)

    logger.info(f"[{method}/{model}/{dataset}] Total={len(cases)}, Done={done_count}, Pending={len(pending)}")
    model_config = get_model_config(model)
    if not pending:
        # Load existing results for stats
        results = load_completed_results(output_base, method, dataset, model)
        agent_correct = sum(1 for r in results if r.get("is_agent_correct"))
        step_correct = sum(1 for r in results if r.get("is_step_correct"))
        return {
            "total": len(cases),
            "pending": 0,
            "success": len(results),
            "failed": 0,
            "agent_acc": agent_correct / len(results) * 100 if results else 0,
            "step_acc": step_correct / len(results) * 100 if results else 0,
        }

    async def run_with_semaphore(case_dir):
        async with semaphore:
            return await _run_case(method, case_dir, output_base, dataset, model, model_config)

    # Progress bar
    tasks = [run_with_semaphore(c) for c in pending]
    results = []
    with tqdm(total=len(pending), desc=f"{method[:8]}/{model[:12]}/{dataset}", ncols=100) as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            pbar.update(1)
            if result["success"]:
                r = result["result"]
                pbar.set_postfix_str(f"a={r['is_agent_correct']},s={r['is_step_correct']}")

    # Calculate stats
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count

    # Load all results (including previously completed) for accuracy
    all_results = load_completed_results(output_base, method, dataset, model)
    agent_correct = sum(1 for r in all_results if r.get("is_agent_correct"))
    step_correct = sum(1 for r in all_results if r.get("is_step_correct"))

    stats = {
        "total": len(cases),
        "pending": len(pending),
        "success": success_count,
        "failed": failed_count,
        "agent_acc": agent_correct / len(all_results) * 100 if all_results else 0,
        "step_acc": step_correct / len(all_results) * 100 if all_results else 0,
    }

    logger.info(
        f"[{method}/{model}/{dataset}] Completed: success={success_count}, failed={failed_count}, "
        f"agent_acc={stats['agent_acc']:.2f}%, step_acc={stats['step_acc']:.2f}%"
    )

    return stats


async def run_all(
    methods: list,
    models: list,
    datasets: list,
    fault_dir: Path,
    output_base: Path,
    max_concurrency: int = 5,
):
    """Run all combinations of (method, model, dataset)"""
    total_tasks = len(methods) * len(models) * len(datasets)
    logger.info(
        f"{'='*60}\n[BATCH] {len(methods)} methods x {len(models)} models x {len(datasets)} datasets = {total_tasks}\n{'='*60}"
    )

    # Per-model semaphore to prioritize different model concurrency
    model_semaphores = {model: asyncio.Semaphore(max_concurrency) for model in models}

    # Build all tasks, interleave by model for better distribution
    all_tasks = []
    for dataset in datasets:
        for method in methods:
            for model in models:
                all_tasks.append((method, model, dataset))

    async def run_one(method, model, dataset):
        stats = await process_dataset_method(method, dataset, model, fault_dir, output_base, model_semaphores[model])
        stats.update({"method": method, "model": model, "dataset": dataset})
        return stats

    # Run all tasks concurrently
    coros = [run_one(m, mo, d) for m, mo, d in all_tasks]
    all_stats = await asyncio.gather(*coros)

    # Summary
    logger.info(f"\n{'='*60}\n[SUMMARY]")
    for method in methods:
        method_stats = [s for s in all_stats if s["method"] == method]
        total_cases = sum(s["total"] for s in method_stats)
        avg_agent_acc = sum(s["agent_acc"] * s["total"] for s in method_stats) / total_cases if total_cases else 0
        avg_step_acc = sum(s["step_acc"] * s["total"] for s in method_stats) / total_cases if total_cases else 0

        logger.info(f"[{method}] Cases={total_cases}, Agent_Acc={avg_agent_acc:.2f}%, Step_Acc={avg_step_acc:.2f}%")

    # Save summary
    summary_file = output_base / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)
    logger.info(f"\nSummary saved to: {summary_file}")


# Default configs
DEFAULT_METHODS = ["all_at_once", "step_by_step", "binary_search"]
DEFAULT_MODELS = [
    "doubao-seed-1-8-251228",
    "deepseek-v3-2-251201",
    "gpt-5.2-chat",
    "claude-sonnet-4.5",
    "kimi-k2-250905",
]
DEFAULT_DATASETS = ["veadk_gaia", "langgraph_sql", "autogen_math", "agno_rca", "adk_swe"]


def main():
    parser = argparse.ArgumentParser(description="Batch execution for compare methods")
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS, choices=["all_at_once", "step_by_step", "binary_search"]
    )
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    parser.add_argument("--fault_dir", type=str, default="../datasets/fault_injected")
    parser.add_argument("--output_dir", type=str, default="../results/compare_results")
    parser.add_argument("--max_concurrency", type=int, default=10, help="Max concurrent cases per dataset")
    args = parser.parse_args()

    load_dotenv()

    output_base = Path(args.output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    init_logger(str(output_base / "run.log"), level="INFO")

    asyncio.run(
        run_all(
            methods=args.methods,
            models=args.models,
            datasets=args.datasets,
            fault_dir=Path(args.fault_dir),
            output_base=output_base,
            max_concurrency=args.max_concurrency,
        )
    )


if __name__ == "__main__":
    main()
