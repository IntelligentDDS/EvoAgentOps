# compare_run_whodata.py
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))
import json
import asyncio
import argparse
from pathlib import Path
from dotenv import load_dotenv
from evoagentops.judge import Judge
from evoagentops.config import Config
from evoagentops.util import init_logger, logger


def convert_who_to_our_format(data: dict) -> tuple:
    """Convert Who&When format to our format. Returns (agent_steps, agent_dependency, agent_settings, label)"""
    history = data.get("history", [])
    is_handcrafted = data.get("is_handcrafted", False)
    index_agent = "role" if is_handcrafted else "name"

    # Convert history to agent_steps (1-indexed)
    agent_steps = []
    agent_names = set()
    for i, entry in enumerate(history):
        agent_name = entry.get(index_agent, "Unknown_Agent")
        agent_names.add(agent_name)
        agent_steps.append(
            {
                "step": i + 1,  # 1-indexed
                "agent_name": agent_name,
                "agent": {
                    "input": [{"role": "user", "content": data.get("question", "")}] if i == 0 else [],
                    "output": {"role": "assistant", "content": entry.get("content", "")},
                    "tools_called": [],
                },
            }
        )

    # Build agent_dependency from agent_names
    agent_dependency = {name: {"agent": [], "tool": []} for name in agent_names}

    # Build agent_settings from system_prompt
    system_prompts = data.get("system_prompt", {})
    agent_settings = {
        "prompt": system_prompts if isinstance(system_prompts, dict) else {},
        "tool": [],
    }

    # Build label (convert 0-indexed to 1-indexed)
    label = {
        "mistake_agent": data.get("mistake_agent", ""),
        "mistake_step_0idx": str(data.get("mistake_step", "0")),  # Original 0-indexed
        "root_cause_step": int(data.get("mistake_step", 0)) + 1,  # Convert to 1-indexed
        "root_cause_agent": data.get("mistake_agent", ""),
        "ground_truth": data.get("ground_truth", ""),
        "question": data.get("question", ""),
    }

    return agent_steps, agent_dependency, agent_settings, label


def extract_prediction_from_judge(result: dict, agent_steps: list) -> tuple:
    """Extract predicted agent and step from judge result. Returns (agent, step_0idx_str)"""
    # Try fault_sorted first
    fault_sorted = result.get("global_result", {}).get("fault_sorted", [])
    if fault_sorted:
        fault = fault_sorted[0]
        step_1idx = fault.get("step", 1)
        agent = fault.get("agent", "")
        return agent, str(step_1idx - 1)  # Convert to 0-indexed string

    # Try judge_result reasons
    judge_result = result.get("global_result", {}).get("judge_result", [])
    for metric in judge_result:
        reasons = metric.get("reasons", [])
        if reasons:
            reason = reasons[0]
            step_1idx = reason.get("step", 1)
            agent = reason.get("agent", "")
            return agent, str(step_1idx - 1)  # Convert to 0-indexed string

    # Fallback to last step
    if agent_steps:
        last = agent_steps[-1]
        return last.get("agent_name", ""), str(last.get("step", 1) - 1)
    return "", "0"


async def run_single_case(
    json_path: Path,
    output_dir: Path,
    config: Config,
    principles_str: str = "",
    global_level: bool = True,
    agent_level: bool = True,
) -> dict:
    """Run Judge on single Who&When case. Returns result dict with evaluation."""
    # Load and convert data
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # Detect dataset type from path
    is_handcrafted = "Hand-Crafted" in str(json_path)
    data["is_handcrafted"] = is_handcrafted

    agent_steps, agent_dependency, agent_settings, label = convert_who_to_our_format(data)
    case_id = json_path.stem

    # Save converted data
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump({"original": data, "converted_steps": agent_steps, "label": label}, f, ensure_ascii=False, indent=2)

    logger.info(f"Processing: {case_id}, history_len={len(agent_steps)}")

    # Run Judge
    judge = Judge(agent_steps, agent_dependency, agent_settings, config)
    result = await judge.judge_once(
        task_compare_withlabel="fail",  # Who&When cases are all incorrect
        global_level=global_level,
        agent_level=agent_level,
        principles_str=principles_str,
    )

    # Extract prediction
    pred_agent, pred_step_0idx = extract_prediction_from_judge(result, agent_steps)

    # Evaluate using string contains (same as reference)
    label_agent = label["mistake_agent"]
    label_step_0idx = label["mistake_step_0idx"]
    is_agent_correct = (label_agent in pred_agent) if pred_agent else False
    is_step_correct = (label_step_0idx == pred_step_0idx)

    output = {
        "case_id": case_id,
        "predicted_agent": pred_agent,
        "predicted_step": pred_step_0idx,  # 0-indexed string for compatibility
        "label_agent": label_agent,
        "label_step": label_step_0idx,
        "is_agent_correct": is_agent_correct,
        "is_step_correct": is_step_correct,
        "judge_result": result,
    }

    with open(output_dir / "output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"Done: {case_id}, pred={pred_agent}/{pred_step_0idx}, label={label_agent}/{label_step_0idx}, "
        f"agent_ok={is_agent_correct}, step_ok={is_step_correct}"
    )

    return output


async def main():
    parser = argparse.ArgumentParser(description="Run Judge on single Who&When case")
    parser.add_argument("--json_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--openai_base_url", type=str, default=None)
    parser.add_argument("--openai_api_key", type=str, default=None)
    parser.add_argument("--openai_model", type=str, default=None)
    parser.add_argument("--global_level", type=bool, default=True)
    parser.add_argument("--agent_level", type=bool, default=True)
    args = parser.parse_args()

    load_dotenv()

    json_path = Path(args.json_path)
    output_dir = Path(args.output_dir) if args.output_dir else json_path.parent / "output" / json_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Config(output_dir=str(output_dir))
    if args.openai_base_url:
        config.openai_base_url = args.openai_base_url
    if args.openai_api_key:
        config.openai_api_key = args.openai_api_key
    if args.openai_model:
        config.openai_model = args.openai_model

    init_logger(str(output_dir / "run.log"), level="INFO")

    result = await run_single_case(json_path, output_dir, config, "", args.global_level, args.agent_level)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
