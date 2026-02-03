# compare_run_whdata_aggre_train_bank.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import asyncio
import json
import hashlib
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from evoagentops.config import Config
from evoagentops.util import init_logger, logger, acall_llm
from evoagentops.casebank import PrincipleBank
from dotenv import load_dotenv

# Model config mapping
MODEL_PREFIX_MAP = {
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
    for keyword, prefix in MODEL_PREFIX_MAP.items():
        if keyword in model_lower:
            return (
                os.getenv(f"{prefix}_OPENAI_BASE_URL"),
                os.getenv(f"{prefix}_OPENAI_API_KEY"),
                os.getenv(f"{prefix}_OPENAI_MODEL"),
            )
    return os.getenv("OPENAI_BASE_URL"), os.getenv("OPENAI_API_KEY"), os.getenv("OPENAI_MODEL")


def load_principle_bank(bank_file: Path) -> list:
    """Load principle bank from jsonl file, return list of principles"""
    if not bank_file.exists():
        return []
    principles = []
    with open(bank_file, encoding="utf-8") as f:
        for line in f:
            try:
                principles.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skip malformed line in {bank_file}: {e}")
    return principles


def aggregate_principles(all_principles: list[list]) -> list:
    """Aggregate principles from multiple datasets, dedup by content"""
    seen = set()
    result = []
    for principles in all_principles:
        for p in principles:
            content = p.get("content", "") or p.get("title", "")
            if content and content not in seen:
                seen.add(content)
                # Normalize agent_name to None for agent-level
                p_copy = p.copy()
                if p_copy.get("type") == "agent":
                    p_copy["agent_name"] = None
                result.append(p_copy)
    return result


async def process_model(model_name: str, input_dir: Path, output_dir: Path) -> dict:
    """Process one model: merge datasets then fuse with LLM"""
    model_dir = input_dir / model_name
    model_output_dir = output_dir / model_name / "principlebank"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    # Step 1: Merge all datasets
    datasets = [d.name for d in sorted(model_dir.iterdir()) if d.is_dir()]
    if not datasets:
        logger.warning(f"[{model_name}] No datasets found, skip")
        return {"model": model_name, "aggre": 0, "fused": 0}

    all_principles = []
    for dataset_name in datasets:
        bank_file = model_dir / dataset_name / "principlebank" / "principlebank.jsonl"
        principles = load_principle_bank(bank_file)
        if principles:
            all_principles.append(principles)
            logger.info(f"[{model_name}] Loaded {dataset_name}: {len(principles)} principles")
    aggregated = aggregate_principles(all_principles)
    logger.info(f"[{model_name}] Aggregated from {len(datasets)} datasets: {len(aggregated)} principles")
    # Write aggregated results
    aggre_file = model_output_dir / "principlebank_aggre.jsonl"
    with open(aggre_file, "w", encoding="utf-8") as f:
        for p in aggregated:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    # Step 2: Fuse with LLM
    base_url, api_key, llm_model = get_model_config(model_name)
    config = Config(output_dir=str(model_output_dir))
    if base_url:
        config.openai_base_url = base_url
    if api_key:
        config.openai_api_key = api_key
    if llm_model:
        config.openai_model = llm_model
    # Use PrincipleBank._merge_principles_batch directly
    pb = PrincipleBank(str(model_output_dir), "/dev/null", config)

    fused_file = model_output_dir / "principlebank.jsonl"

    def on_scope_done(merged_list):
        with open(fused_file, "w", encoding="utf-8") as f:
            for p in merged_list:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    fused = await pb._merge_principles_batch(aggregated, [], on_scope_done=on_scope_done)

    # Final write
    with open(fused_file, "w", encoding="utf-8") as f:
        for p in fused:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    logger.info(f"[{model_name}] Fused (LLM={llm_model or 'default'}): {len(aggregated)}->{len(fused)}")
    return {"model": model_name, "aggre": len(aggregated), "fused": len(fused)}


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Merge principle banks across datasets by model")
    parser.add_argument("--input_dir", default="../results/train_bank", help="Input train_bank directory")
    parser.add_argument(
        "--output_dir", default="../results/run_whodata_train_bank", help="Output merged bank directory"
    )
    parser.add_argument("--max_concurrency", type=int, default=5, help="Max concurrent models")
    args = parser.parse_args()

    load_dotenv()
    input_dir, output_dir = Path(args.input_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_logger(f"{output_dir}/merge.log")

    if not input_dir.exists():
        logger.error(f"Input dir not found: {input_dir}")
        return

    # Load existing results
    done_models = set()

    models = [d.name for d in sorted(input_dir.iterdir()) if d.is_dir()]
    for m in models:
        fused_file = output_dir / m / "principlebank" / "principlebank.jsonl"
        if fused_file.exists():
            try:
                with open(fused_file, encoding="utf-8") as f:
                    valid = sum(1 for line in f if line.strip() and json.loads(line).get("principle_id"))
                if valid > 0:
                    done_models.add(m)
                    logger.info(f"Resume skip: {m} ({valid} principles)")
            except Exception as e:
                logger.warning(f"Resume check failed: {m}, err={e}")
    pending = [m for m in models if m not in done_models]
    logger.info(f"Merge start: input={input_dir}, output={output_dir}, models={len(models)}, pending={len(pending)}")
    results = []
    if pending:
        sem = asyncio.Semaphore(args.max_concurrency)

        async def run_with_sem(m):
            async with sem:
                try:
                    result = await process_model(m, input_dir, output_dir)
                except Exception as e:
                    logger.error(f"[{m}] Failed: {e}")
                    result = {"model": m, "aggre": {}, "fused": {}, "error": str(e)}

                return result

        tasks = [run_with_sem(m) for m in pending]
        with tqdm(total=len(pending), desc="Models", unit="model") as pbar:
            for coro in asyncio.as_completed(tasks):
                result = await coro
                results.append(result)
                pbar.update(1)
                pbar.set_postfix_str(f"{result['model'][:12]}{'(err)' if 'error' in result else ''}")

    done_count = len(done_models) + len([r for r in results if "error" not in r])
    logger.info(f"Done: {done_count}/{len(models)} models, output -> {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
