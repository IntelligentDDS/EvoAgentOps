# compare_who.py
import os
import json
import asyncio
import argparse
import re
import random
from pathlib import Path
from dotenv import load_dotenv
from evoagentops.config import Config
from evoagentops.util import init_logger, logger, acall_llm


def load_case_data(case_dir: Path) -> dict:
    """Load agent_steps.json and label.json, build unified data structure"""
    with open(case_dir / "agent_steps.json", encoding="utf-8") as f:
        steps = json.load(f)
    with open(case_dir / "label.json", encoding="utf-8") as f:
        label = json.load(f)
    settings = {}
    settings_file = case_dir / "agent_settings.json"
    if settings_file.exists():
        with open(settings_file, encoding="utf-8") as f:
            settings = json.load(f)

    # Extract question from first step
    question = steps[0]["agent"]["input"][0]["content"] if steps else ""

    # Get root_cause_agent from steps using root_cause_step
    root_cause_step = label.get("root_cause_step", 0)
    root_cause_agent = (
        steps[root_cause_step - 1]["agent_name"] if root_cause_step > 0 and root_cause_step <= len(steps) else ""
    )

    return {
        "steps": steps,
        "question": question,
        "original_correct_answer": label.get("original_correct_answer", ""),
        "root_cause_step": root_cause_step,
        "root_cause_agent": root_cause_agent,
        "label": label,
        "settings": settings,
    }


def build_who_format_data(data: dict, case_id: str = "") -> dict:
    """Build data in WHO baseline format"""
    history = []
    for s in data["steps"]:
        agent_data = s.get("agent", {})
        history.append(
            {
                "content": str(agent_data) if agent_data else "",
                "role": "assistant",
                "name": s.get("agent_name", ""),
            }
        )
    label = data.get("label", {})
    settings = data.get("settings", {})
    # Convert 1-indexed to 0-indexed for WHO format
    step_0idx = data["root_cause_step"] - 1 if data["root_cause_step"] > 0 else 0
    return {
        "is_correct": False,
        "question": data["question"],
        "question_ID": case_id,
        "level": "3",
        "ground_truth": data["original_correct_answer"],
        "history": history,
        "mistake_agent": data["root_cause_agent"],
        "mistake_step": str(step_0idx),
        "fault_type": label.get("fault_type", ""),
        "mistake_reason": label.get("causality_chain", "") or str(label.get("modify_description", "")),
        "system_prompt": settings.get("prompt", {}),
        "tools": settings.get("tool", []),
    }


def format_chat_content(steps: list) -> str:
    """Format steps into chat content string (matches reference format)"""
    lines = []
    for entry in steps:
        agent_name = entry.get("agent_name", "Unknown Agent")
        agent_data = entry.get("agent", {})
        # Format agent data as string
        content = str(agent_data) if agent_data else ""
        lines.append(f"{agent_name}: {content}")
    return "\n".join(lines)


async def all_at_once(data: dict, config: Config) -> dict:
    """
    All-at-once method: Feed entire conversation to LLM at once.
    Returns: {"predicted_agent": str, "predicted_step": int, "reason": str}
    """
    problem = data["question"]
    ground_truth = data["original_correct_answer"]
    chat_content = format_chat_content(data["steps"])

    # Original prompt from reference implementation
    prompt = (
        "You are an AI assistant tasked with analyzing a multi-agent conversation history when solving a real world problem. "
        f"The problem is:  {problem}\n"
        f"The Answer for the problem is: {ground_truth}\n"
        "Identify which agent made an error, at which step, and explain the reason for the error. "
        "Here's the conversation:\n\n"
        + chat_content
        + "\n\nBased on this conversation, please predict the following:\n"
        "1. The name of the agent who made a mistake that should be directly responsible for the wrong solution to the real world problem. If there are no agents that make obvious mistakes, decide one single agent in your mind. Directly output the name of the Expert.\n"
        "2. In which step the mistake agent first made mistake. For example, in a conversation structured as follows: "
        """
{
    "agent a": "xx",
    "agent b": "xxxx",
    "agent c": "xxxxx",
    "agent a": "xxxxxxx"
},
"""
        "each entry represents a 'step' where an agent provides input. The 'x' symbolizes the speech of each agent. If the mistake is in agent c's speech, the step number is 2. If the second speech by 'agent a' contains the mistake, the step number is 3, and so on. Please determine the step number where the first mistake occurred.\n"
        "3. The reason for your prediction."
        "Please answer in the format: Agent Name: (Your prediction)\n Step Number: (Your prediction)\n Reason for Mistake: \n"
    )

    messages = [
        {"role": "system", "content": "You are a helpful assistant skilled in analyzing conversations."},
        {"role": "user", "content": prompt},
    ]

    result = await acall_llm(messages, config)

    # Parse response (reference implementation logic)
    agent_match = re.search(r"Agent Name:\s*([\w_]+)", result, re.IGNORECASE)
    step_match = re.search(r"Step Number:\s*(\d+)", result, re.IGNORECASE)
    reason_match = re.search(r"Reason for Mistake:\s*(.+)", result, re.DOTALL | re.IGNORECASE)

    predicted_agent = agent_match.group(1) if agent_match else ""
    # Convert 0-indexed to 1-indexed (reference uses 0-indexed, our data uses 1-indexed)
    predicted_step = int(step_match.group(1)) + 1 if step_match else 0
    reason = reason_match.group(1).strip() if reason_match else result

    return {
        "predicted_agent": predicted_agent,
        "predicted_step": predicted_step,
        "reason": reason,
        "raw_response": result,
    }


async def step_by_step(data: dict, config: Config) -> dict:
    """
    Step-by-step method: Check each step sequentially until error found.
    Returns: {"predicted_agent": str, "predicted_step": int, "reason": str}
    """
    problem = data["question"]
    ground_truth = data["original_correct_answer"]
    steps = data["steps"]

    current_conversation_history = ""
    intermediate_results = []

    for idx, entry in enumerate(steps):
        agent_name = entry.get("agent_name", "Unknown Agent")
        content = str(entry.get("agent", ""))
        current_conversation_history += f"Step {idx} - {agent_name}: {content}\n"

        # Original prompt from reference implementation
        prompt = (
            f"You are an AI assistant tasked with evaluating the correctness of each step in an ongoing multi-agent conversation aimed at solving a real-world problem. The problem being addressed is: {problem}. "
            f"The Answer for the problem is: {ground_truth}\n"
            f"Here is the conversation history up to the current step:\n{current_conversation_history}\n"
            f"The most recent step ({idx}) was by '{agent_name}'.\n"
            f"Your task is to determine whether this most recent agent's action (Step {idx}) contains an error that could hinder the problem-solving process or lead to an incorrect solution. "
            "Please respond with 'Yes' or 'No' and provide a clear explanation for your judgment. "
            "Note: Please avoid being overly critical in your evaluation. Focus on errors that clearly derail the process."
            "Respond ONLY in the format: 1. Yes/No.\n2. Reason: [Your explanation here]"
        )

        messages = [
            {"role": "system", "content": "You are a precise step-by-step conversation evaluator."},
            {"role": "user", "content": prompt},
        ]

        logger.info(f"Evaluating Step {idx} by {agent_name}...")
        answer = await acall_llm(messages, config)

        # Save intermediate result
        is_error = answer.lower().strip().startswith("1. yes")
        intermediate_results.append({"step": idx, "agent": agent_name, "response": answer, "is_error": is_error})
        with open(Path(config.output_dir) / "intermediate.json", "w", encoding="utf-8") as f:
            json.dump(intermediate_results, f, ensure_ascii=False, indent=2)

        # Check for "Yes" response (reference logic)
        if is_error:
            reason = answer.split("Reason:", 1)[-1].strip() if "Reason:" in answer else answer
            # Convert to 1-indexed
            return {
                "predicted_agent": agent_name,
                "predicted_step": idx + 1,
                "reason": reason,
                "raw_response": answer,
            }

    # No error found - return last step (fallback)
    last_entry = steps[-1] if steps else {}
    return {
        "predicted_agent": last_entry.get("agent_name", ""),
        "predicted_step": len(steps),
        "reason": "No decisive errors found by step-by-step analysis",
        "raw_response": "",
    }


async def binary_search(data: dict, config: Config) -> dict:
    """
    Binary search method: Recursively narrow down error location.
    Returns: {"predicted_agent": str, "predicted_step": int, "reason": str}
    """
    problem = data["question"]
    answer = data["original_correct_answer"]
    steps = data["steps"]
    search_trace = []

    async def find_error_recursive(start: int, end: int) -> tuple:
        """Returns (step_idx, agent_name, reason)"""
        if start > end:
            return end if end >= 0 else 0, steps[end]["agent_name"] if end >= 0 else "", "Invalid range"
        if start == end:
            return start, steps[start]["agent_name"], "Binary search converged"

        # Build segment content
        segment = steps[start : end + 1]
        chat_content = "\n".join(
            [f"{entry.get('agent_name', 'Unknown')}: {entry.get('agent', '')}" for entry in segment]
        )

        mid = start + (end - start) // 2
        range_desc = f"from step {start} to step {end}"
        upper_desc = f"from step {start} to step {mid}"
        lower_desc = f"from step {mid + 1} to step {end}"

        # Original prompt from reference implementation
        prompt = (
            "You are an AI assistant tasked with analyzing a segment of a multi-agent conversation. Multiple agents are collaborating to address a user query, with the goal of resolving the query through their collective dialogue.\n"
            "Your primary task is to identify the location of the most critical mistake within the provided segment. Determine which half of the segment contains the single step where this crucial error occurs, ultimately leading to the failure in resolving the user's query.\n"
            f"The problem to address is as follows: {problem}\n"
            f"The Answer for the problem is: {answer}\n"
            f"Review the following conversation segment {range_desc}:\n\n{chat_content}\n\n"
            f"Based on your analysis, predict whether the most critical error is more likely to be located in the upper half ({upper_desc}) or the lower half ({lower_desc}) of this segment.\n"
            "Please provide your prediction by responding with ONLY 'upper half' or 'lower half'. Remember, your answer should be based on identifying the mistake that directly contributes to the failure in resolving the user's query. If no single clear error is evident, consider the step you believe is most responsible for the failure, allowing for subjective judgment, and base your answer on that."
        )

        messages = [
            {
                "role": "system",
                "content": "You are an AI assistant specializing in localizing errors in conversation segments.",
            },
            {"role": "user", "content": prompt},
        ]

        logger.info(f"Analyzing segment [{start}-{end}]...")
        result = await acall_llm(messages, config)
        result_lower = result.lower()
        # Determine choice and save trace
        if "upper half" in result_lower:
            choice = "upper"
        elif "lower half" in result_lower:
            choice = "lower"
        else:
            choice = "random"
        search_trace.append({"range": [start, end], "mid": mid, "response": result, "choice": choice})
        with open(Path(config.output_dir) / "intermediate.json", "w", encoding="utf-8") as f:
            json.dump(search_trace, f, ensure_ascii=False, indent=2)

        # Parse response (reference logic with random fallback)
        if choice == "upper":
            return await find_error_recursive(start, mid)
        elif choice == "lower":
            new_start = min(mid + 1, end)
            return await find_error_recursive(new_start, end)
        else:
            logger.warning(f"Ambiguous response: {result}, randomly choosing")
            if random.randint(0, 1) == 0:
                return await find_error_recursive(start, mid)
            else:
                return await find_error_recursive(min(mid + 1, end), end)

    if not steps:
        return {"predicted_agent": "", "predicted_step": 0, "reason": "No steps", "raw_response": ""}

    step_idx, agent_name, reason = await find_error_recursive(0, len(steps) - 1)
    # Convert to 1-indexed
    return {
        "predicted_agent": agent_name,
        "predicted_step": step_idx + 1,
        "reason": reason,
        "raw_response": "",
    }


async def run_single_case(
    method: str,
    case_dir: Path,
    output_dir: Path,
    openai_base_url: str = None,
    openai_api_key: str = None,
    openai_model: str = None,
) -> dict:
    """Run single case with specified method. Returns result dict."""
    openai_base_url = openai_base_url or os.getenv("OPENAI_BASE_URL")
    openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
    openai_model = openai_model or os.getenv("OPENAI_MODEL")
    # Setup config
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Config(output_dir=str(output_dir))
    if openai_base_url:
        config.openai_base_url = openai_base_url
    if openai_api_key:
        config.openai_api_key = openai_api_key
    if openai_model:
        config.openai_model = openai_model

    # Load data
    data = load_case_data(case_dir)
    case_id = case_dir.name

    # Save WHO baseline format
    with open(output_dir / "data.json", "w", encoding="utf-8") as f:
        json.dump(build_who_format_data(data, case_id), f, ensure_ascii=False, indent=2)

    logger.info(f"[{method}] Processing: {case_id}, steps={len(data['steps'])}")

    # Run method
    if method == "all_at_once":
        result = await all_at_once(data, config)
    elif method == "step_by_step":
        result = await step_by_step(data, config)
    elif method == "binary_search":
        result = await binary_search(data, config)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build output
    output = {
        "case_id": case_id,
        "method": method,
        "model": openai_model or "unknown",
        "predicted_agent": result["predicted_agent"],
        "predicted_step": result["predicted_step"],
        "reason": result["reason"],
        "is_agent_correct": data["root_cause_agent"] in result["predicted_agent"],
        "is_step_correct": result["predicted_step"] == data["root_cause_step"],
    }

    # Save result
    with open(output_dir / "output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[{method}] Done: {case_id}, pred_step={result['predicted_step']}, "
        f"agent_ok={output['is_agent_correct']}, step_ok={output['is_step_correct']}"
    )

    return output


async def main():
    parser = argparse.ArgumentParser(description="Run single case with compare methods")
    parser.add_argument("--method", type=str, required=True, choices=["all_at_once", "step_by_step", "binary_search"])
    parser.add_argument(
        "--case_dir", type=str, required=True, help="Path to case directory (contains agent_steps.json, label.json)"
    )
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (default: case_dir/output)")
    parser.add_argument("--openai_base_url", type=str, default=None, help="OpenAI API base URL")
    parser.add_argument("--openai_api_key", type=str, default=None, help="OpenAI API key")
    parser.add_argument("--openai_model", type=str, default=None, help="OpenAI model name")
    args = parser.parse_args()

    load_dotenv()

    case_dir = Path(args.case_dir)
    output_dir = Path(args.output_dir) if args.output_dir else case_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    init_logger(str(output_dir / "run.log"), level="INFO")

    result = await run_single_case(
        args.method, case_dir, output_dir, args.openai_base_url, args.openai_api_key, args.openai_model
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
