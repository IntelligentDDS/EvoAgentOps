from .util import logger, acall_llm, set_call_context
from .config import Config
import json
from typing import Dict, Optional, List
from pydantic import BaseModel, Field


class ReasonCheckOutput(BaseModel):
    """Reason match check output"""

    global_is_match: bool = Field(description="Whether global judge matches the injected fault")
    global_description: str = Field(description="Explanation for global judge match result")
    agent_is_match: bool = Field(description="Whether agent judge matches the injected fault")
    agent_description: str = Field(description="Explanation for agent judge match result")


async def check_is_success(injected_data: Dict, judge_result: Dict, config: Config = None) -> Dict:
    """Check if judge successfully identifies the fault - compatible with all/global/agent cases"""
    config = config or Config()
    set_call_context(stage="fault_check", case_id=None, idx=None)
    logger.info("Start checking judge identification results")

    # get injected fault info
    root_injection_step = injected_data["label"]["root_cause_step"]
    modify_description = "\n".join(injected_data["label"]["modify_description"])
    step_to_agent = {s["step"]: s["agent_name"] for s in injected_data["agent_steps"]}
    expected_agent = step_to_agent.get(root_injection_step)

    # safe get global and agent faults
    global_faults = judge_result.get("global_case", {}).get("fault_root_cause", [])
    agent_faults = []
    for agent_case in judge_result.get("agent_cases", []):
        agent_faults.extend(agent_case.get("fault_root_cause", []))

    # check global
    global_matched = find_matched_fault(global_faults, root_injection_step)
    global_step = global_matched is not None
    global_agent = check_agent_match(global_matched, expected_agent) if global_matched else False

    # check agent
    agent_matched = find_matched_fault(agent_faults, root_injection_step)
    agent_step = agent_matched is not None
    agent_agent = check_agent_match(agent_matched, expected_agent) if agent_matched else False

    # check reason level
    if global_step or agent_step:
        reason_result = await check_reason_match(global_matched, agent_matched, modify_description, config)
        global_reason = reason_result["global_is_match"] if global_step else False
        global_desc = reason_result["global_description"] if global_step else "No fault detected at step level"
        agent_reason = reason_result["agent_is_match"] if agent_step else False
        agent_desc = reason_result["agent_description"] if agent_step else "No fault detected at step level"
    else:
        global_reason = False
        global_desc = "No fault detected at step level"
        agent_reason = False
        agent_desc = "No fault detected at step level"

    # build result
    result = {
        "fault_injection": {
            "root_cause_step": root_injection_step,
            "modify_description": modify_description,
            "expected_agent": expected_agent,
        },
        "all_judge": {
            "step_level": global_step or agent_step,
            "agent_level": global_agent or agent_agent,
            "reason_level": global_reason or agent_reason,
        },
        "global_judge": {
            "step_level": global_step,
            "agent_level": global_agent,
            "reason_level": global_reason,
            "check_description": global_desc,
            "matched_fault_root_causes": global_matched,
            "all_fault_root_causes": global_faults,
        },
        "agent_judge": {
            "step_level": agent_step,
            "agent_level": agent_agent,
            "reason_level": agent_reason,
            "check_description": agent_desc,
            "matched_fault_root_causes": agent_matched,
            "all_fault_root_causes": agent_faults,
        },
    }

    logger.info(
        f"Check completed: all({result['all_judge']['step_level']}/{result['all_judge']['agent_level']}/{result['all_judge']['reason_level']}), "
        f"global({global_step}/{global_agent}/{global_reason}), agent({agent_step}/{agent_agent}/{agent_reason})"
    )

    return result


def find_matched_fault(faults: List[Dict], target_step: int) -> Optional[Dict]:
    """Find fault that matches the target step"""
    for fault in faults:
        if str(fault.get("root_cause", {}).get("step", "")) == str(target_step):
            return fault
    return None


def check_agent_match(fault: Optional[Dict], expected_agent: str) -> bool:
    """Check if the fault's agent matches the expected agent"""
    if not fault:
        return False
    return fault.get("root_cause", {}).get("agent") == expected_agent


async def check_reason_match(
    global_fault: Optional[Dict], agent_fault: Optional[Dict], modify_description: str, config: Config
) -> Dict:
    """Check if the fault's reason matches the injected description using LLM"""

    # build global info
    if global_fault:
        global_fault_text = global_fault.get("fault", "")
        global_reasons = "\n".join(global_fault.get("root_cause", {}).get("reasons_content", []))
        global_info = f"""Fault: {global_fault_text}
Reasons:
{global_reasons}"""
    else:
        global_info = "No fault detected at the injected step"

    # build agent info
    if agent_fault:
        agent_fault_text = agent_fault.get("fault", "")
        agent_reasons = "\n".join(agent_fault.get("root_cause", {}).get("reasons_content", []))
        agent_info = f"""Fault: {agent_fault_text}
Reasons:
{agent_reasons}"""
    else:
        agent_info = "No fault detected at the injected step"

    prompt = f"""## Injected Fault Description
{modify_description}

## Judge Results
### Global Judge Result
{global_info}

### Agent Judge Result
{agent_info}

## Task
Ignore the step numbers in the descriptions. Focus on the fault content.
Evaluate whether each judge (global and agent) correctly identified the injected fault.

For each judge that detected a fault:
- Compare the judge's identified fault and reasons with the injected fault description
- Determine if they match
- Provide detailed explanation

For each judge that did NOT detect a fault at the injected step:
- Set is_match to false
- Explain that no fault was detected

Return JSON:
{{
  "global_is_match": true/false,
  "global_description": "explanation for global judge",
  "agent_is_match": true/false,
  "agent_description": "explanation for agent judge"
}}
"""

    messages = [{"role": "user", "content": prompt}]

    try:
        response = await acall_llm(messages, config, output_schema=ReasonCheckOutput)
        return json.loads(response)
    except Exception as e:
        logger.error(f"Reason check failed: {e}")
        return {
            "global_is_match": False,
            "global_description": f"Error: {e}",
            "agent_is_match": False,
            "agent_description": f"Error: {e}",
        }
