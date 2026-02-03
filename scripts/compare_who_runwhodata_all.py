# compare_who_runwhodata_all.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from evoagentops.util import init_logger, logger
from compare_who_runwhodata import run_single_who_case

# Model config mapping
MODEL_CONFIG_MAP = {
    "deepseek": "DEEPSEEK",
    "gpt": "GPT",
    "seed": "SEED",
    "doubao": "SEED",
    "kimi": "KIMI",
    "claude": "CLAUDE",
}


def get_model_config(model_name: str) -> tuple:
    """Get (base_url, api_key, model) from env vars"""
    model_lower = model_name.lower() if model_name else ""
    for keyword, prefix in MODEL_CONFIG_MAP.items():
        if keyword in model_lower:
            return (
                os.getenv(f"{prefix}_OPENAI_BASE_URL"),
                os.getenv(f"{prefix}_OPENAI_API_KEY"),
                os.getenv(f"{prefix}_OPENAI_MODEL"),
            )
    return os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")


def get_json_files(data_dir: Path) -> list:
    """Get sorted JSON files from directory"""
    files = list(data_dir.glob("*.json"))
    return sorted(files, key=lambda x: int("".join(filter(str.isdigit, x.stem)) or 0))


def get_pending_files(json_files: list, output_base: Path) -> list:
    """Get pending files by checking output.json existence"""
    pending = []
    for f in json_files:
        output_file = output_base / f.stem / "output.json"
        if output_file.exists():
            try:
                with open(output_file, encoding="utf-8") as fp:
                    data = json.load(fp)
                if data.get("case_id") and "predicted_step" in data:
                    continue  # Valid result, skip
            except Exception as e:
                logger.warning(f"Invalid {output_file}: {e}")
        pending.append(f)
    return pending


def load_completed_results(output_base: Path) -> list:
    """Load all completed results from output directory"""
    results = []
    if not output_base.exists():
        return results
    for case_folder in output_base.iterdir():
        if case_folder.is_dir():
            output_file = case_folder / "output.json"
            if output_file.exists():
                try:
                    with open(output_file, encoding="utf-8") as f:
                        results.append(json.load(f))
                except Exception as e:
                    logger.warning(f"Failed to load {output_file}: {e}")
    return results


async def _run_case(method: str, file_path: Path, output_base: Path, is_handcrafted: bool, model_config: tuple) -> dict:
    """Process single case, return result dict"""
    output_dir = output_base / file_path.stem
    try:
        result = await run_single_who_case(method, file_path, output_dir, is_handcrafted, *model_config)
        return {"success": True, "case_id": file_path.stem, "result": result}
    except Exception as e:
        logger.error(f"[{method}] Failed: {file_path.stem}: {e}")
        return {"success": False, "case_id": file_path.stem, "error": str(e)}


async def process_dataset_method(
    method: str,
    dataset: str,
    model: str,
    data_dir: Path,
    output_base: Path,
    is_handcrafted: bool,
    semaphore: asyncio.Semaphore,
) -> dict:
    """Process all cases for one (method, dataset, model) combination"""
    json_files = get_json_files(data_dir)
    if not json_files:
        logger.warning(f"[{method}/{model}/{dataset}] No JSON files in {data_dir}")
        return {"total": 0, "pending": 0, "success": 0, "failed": 0}

    total = len(json_files)
    out_dir = output_base / method / model / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    pending = get_pending_files(json_files, out_dir)
    done_count = total - len(pending)
    logger.info(f"[{method}/{model}/{dataset}] Total={total}, Done={done_count}, Pending={len(pending)}")

    model_config = get_model_config(model)

    if not pending:
        results = load_completed_results(out_dir)
        agent_correct = sum(1 for r in results if r.get("is_agent_correct"))
        step_correct = sum(1 for r in results if r.get("is_step_correct"))
        return {
            "total": total,
            "pending": 0,
            "success": len(results),
            "failed": 0,
            "agent_acc": agent_correct / len(results) * 100 if results else 0,
            "step_acc": step_correct / len(results) * 100 if results else 0,
        }

    async def run_with_semaphore(f):
        async with semaphore:
            return await _run_case(method, f, out_dir, is_handcrafted, model_config)

    tasks = [run_with_semaphore(f) for f in pending]
    results = []
    with tqdm(total=len(pending), desc=f"{method[:8]}/{model[:12]}/{dataset[:12]}", ncols=100) as pbar:
        for coro in asyncio.as_completed(tasks):
            r = await coro
            results.append(r)
            pbar.update(1)
            if r["success"]:
                res = r["result"]
                pbar.set_postfix_str(f"a={res['is_agent_correct']},s={res['is_step_correct']}")
    success_count = sum(1 for r in results if r["success"])
    failed_count = len(results) - success_count

    all_results = load_completed_results(out_dir)
    agent_correct = sum(1 for r in all_results if r.get("is_agent_correct"))
    step_correct = sum(1 for r in all_results if r.get("is_step_correct"))

    stats = {
        "total": total,
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
    data_base: Path,
    output_base: Path,
    max_concurrency: int = 10,
):
    """Run all combinations of (method, model, dataset) concurrently"""
    total_tasks = len(methods) * len(models) * len(datasets)
    logger.info(
        f"{'='*60}\n[BATCH] {len(methods)} methods x {len(models)} models x {len(datasets)} datasets = {total_tasks}\n{'='*60}"
    )

    # Per-model semaphore to prioritize different model concurrency
    model_semaphores = {model: asyncio.Semaphore(max_concurrency) for model in models}

    # Build all tasks
    all_tasks = []
    for dataset in datasets:
        data_dir = data_base / dataset
        if not data_dir.exists():
            logger.warning(f"Dataset not found: {data_dir}")
            continue
        is_handcrafted = "Hand-Crafted" in dataset
        for method in methods:
            for model in models:
                all_tasks.append((method, model, dataset, data_dir, is_handcrafted))

    async def run_one(method, model, dataset, data_dir, is_handcrafted):
        stats = await process_dataset_method(
            method, dataset, model, data_dir, output_base, is_handcrafted, model_semaphores[model]
        )
        stats.update({"method": method, "model": model, "dataset": dataset})
        return stats

    # Run all tasks concurrently
    coros = [run_one(m, mo, d, dd, hc) for m, mo, d, dd, hc in all_tasks]
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


DEFAULT_METHODS = ["all_at_once", "step_by_step", "binary_search"]
DEFAULT_MODELS = [
    "doubao-seed-1-8-251228",
    "deepseek-v3-2-251201",
    "gpt-5.2-chat",
    "claude-sonnet-4.5",
    "kimi-k2-250905",
]


def main():
    parser = argparse.ArgumentParser(description="Batch run WHO methods on Who&When dataset")
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS, choices=DEFAULT_METHODS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--data_dir", default="../datasets/Who&When", help="Base data directory")
    parser.add_argument("--output_dir", default="../results/compare_runwhodata_results", help="Output directory")
    parser.add_argument("--datasets", nargs="+", default=["Algorithm-Generated", "Hand-Crafted"])
    parser.add_argument("--max_concurrency", type=int, default=20, help="Max concurrent cases per model")
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
            data_base=Path(args.data_dir),
            output_base=output_base,
            max_concurrency=args.max_concurrency,
        )
    )


if __name__ == "__main__":
    main()
