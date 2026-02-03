# judge.py
from .prompt import (
    GLOBAL_LEVEL_JUDGE_USER,
    GLOBAL_LEVEL_INSIGHT_USER,
    AGENT_LEVEL_JUDGE_USER,
    AGENT_LEVEL_INSIGHT_USER,
    build_judge_system_prompt,
    build_global_insight_system_prompt,
    build_agent_insight_system_prompt,
    FAULT_MAP,
    METRIC_MAP,
    METRIC_IDS,
    METRICS,
    fault_code_to_name,
    metric_id_to_name,
)
from .config import Config
from .util import logger, acall_llm, RetryableError
import json
from typing import List, Dict, Optional
import asyncio
import re
from pydantic import BaseModel, Field
from pathlib import Path
import tiktoken


class StepReason(BaseModel):
    """Single step evaluation."""

    step: int = Field(description="Step number")
    fault_type: str = Field(description="Fault code")
    detail: str = Field(description="Brief explanation")


class MetricResult(BaseModel):
    """Evaluation result for one metric."""

    metric: str = Field(description="Metric ID")
    reasons: List[StepReason] = Field(description="Step-level evidence")
    passed: bool = Field(description="Whether criteria satisfied")


class FaultSorted(BaseModel):
    """Fault for investigation, ordered by priority."""

    step: int = Field(description="Step number")
    fault_type: str = Field(description="Fault code")
    detail: str = Field(description="Why investigate this")


class JudgeOutput(BaseModel):
    statement_action: List[str] = Field(description="Key actions")
    judge_result: List[MetricResult] = Field(description="All metrics results")
    fault_sorted: List[FaultSorted] = Field(default=[], description="Faults by investigation priority")
    is_success: bool = Field(description="Whether task completed")


class Principle(BaseModel):
    title: str = Field(description="Concise phrase: [Action] + [Scenario]. No specific values.")
    content: str = Field(description="When [trigger], [action] by [method], avoiding [pitfall].")
    source_metric: list[str] = Field(description="Metric that triggered this principle")


class InsightOutput(BaseModel):
    execute_principle: List[Principle] = Field(description="Principles to improve agent execution behavior.")
    judge_principle: List[Principle] = Field(description="Principles to improve evaluation accuracy.")


# Token batching utility
_enc = tiktoken.get_encoding("cl100k_base")


def _batch_by_token_limit(
    items: list[str], max_tokens: int = 131072, reserve: int = 65536, step_counts: list[int] = None, max_steps: int = 15
) -> list[list[int]]:
    """Split item indices into batches by token limit and total steps"""
    limit = max_tokens - reserve
    token_counts = [len(t) for t in _enc.encode_batch(items)]
    step_counts = step_counts or [1] * len(items)
    batches, batch, total, steps = [], [], 0, 0
    for i, cnt in enumerate(token_counts):
        sc = step_counts[i]
        if cnt > limit or sc > max_steps:  # Oversized item alone
            if batch:
                batches.append(batch)
            batches.append([i])
            batch, total, steps = [], 0, 0
        elif total + cnt > limit or steps + sc > max_steps:
            batches.append(batch)
            batch, total, steps = [i], cnt, sc
        else:
            batch.append(i)
            total += cnt
            steps += sc
    if batch:
        batches.append(batch)
    return batches


class SegmentJudgeItem(BaseModel):
    """Single segment result in batch output"""

    seg_idx: int = Field(description="Segment index from input")
    statement_action: List[str] = Field(description="key actions")
    judge_result: List[MetricResult] = Field(description="All metrics")
    fault_sorted: List[FaultSorted] = Field(default=[], description="Faults by priority")
    is_success: bool = Field(description="Success status")


class BatchAgentJudgeOutput(BaseModel):
    """Batch output for multiple agent segments"""

    results: List[SegmentJudgeItem] = Field(description="Results for each segment, ordered by seg_idx")


class SegmentPrincipleItem(BaseModel):
    """Single segment principle result in batch output"""

    seg_idx: int = Field(description="Segment index from input")
    execute_principle: List[Principle] = Field(description="Principles to improve agent execution")
    judge_principle: List[Principle] = Field(description="Principles to improve evaluation")


class BatchAgentInsightOutput(BaseModel):
    """Batch output for multiple agent segments principles"""

    results: List[SegmentPrincipleItem] = Field(description="Results for each segment, ordered by seg_idx")


class Judge:
    """Judge agent execution at global or agent level."""

    def __init__(
        self,
        agent_steps: List[Dict],
        agent_dependency: Dict,
        agent_settings: Dict,
        config: Config = None,
    ):
        self.agent_steps = agent_steps
        self.agent_dependency = agent_dependency
        self.agent_settings = agent_settings
        self.config = config or Config()

    def _format_step(self, step: Dict) -> Dict:
        """Format step for prompt."""
        return {
            "step": step["step"],
            "agent_name": step["agent_name"],
            "input": step["agent"]["input"],
            "output": step["agent"]["output"],
            "tools_called": step["agent"].get("tools_called", []),
        }

    def _add_agent_to_result(self, result: Dict) -> Dict:
        """Add agent name to each step in result."""
        step_to_agent = {s["step"]: s["agent_name"] for s in self.agent_steps}

        for metric in result.get("judge_result", []):
            for reason in metric.get("reasons", []):
                if reason.get("step") in step_to_agent:
                    reason["agent"] = step_to_agent[reason["step"]]

        for fault in result.get("fault_sorted", []):
            if fault.get("step") in step_to_agent:
                fault["agent"] = step_to_agent[fault["step"]]

        return result

    def _expand_fault_types(self, result: Dict) -> Dict:
        """Convert codes to names: F01->Hallucination, M1->Task Completion"""
        if not result:
            return result

        # Expand in judge_result
        for metric in result.get("judge_result", []):
            if "metric" in metric:
                metric["metric"] = metric_id_to_name(metric["metric"])
            for reason in metric.get("reasons", []):
                if "fault_type" in reason:
                    reason["fault_type"] = fault_code_to_name(reason["fault_type"])

        # Expand in fault_sorted
        for fault in result.get("fault_sorted", []):
            if "fault_type" in fault:
                fault["fault_type"] = fault_code_to_name(fault["fault_type"])

        return result

    def _identify_segments(self) -> List[Dict]:
        """Identify agent segments from steps."""
        steps = self.agent_steps
        if not steps:
            return []
        # Ensure steps are sorted by step number
        steps = sorted(steps, key=lambda x: x.get("step", 0))
        segments = []
        current_agent = steps[0]["agent_name"]
        start_idx = 0

        for i in range(1, len(steps)):
            if steps[i]["agent_name"] != current_agent:
                segments.append(
                    {
                        "agent_name": current_agent,
                        "start_step": steps[start_idx]["step"],
                        "end_step": steps[i - 1]["step"],
                        "start_idx": start_idx,
                        "end_idx": i - 1,
                    }
                )
                current_agent = steps[i]["agent_name"]
                start_idx = i

        segments.append(
            {
                "agent_name": current_agent,
                "start_step": steps[start_idx]["step"],
                "end_step": steps[-1]["step"],
                "start_idx": start_idx,
                "end_idx": len(steps) - 1,
            }
        )
        return segments

    def _filter_passed_metrics(self, result: Dict) -> Dict:
        """Remove passed=true metrics from judge_result, keep only failed ones."""
        if not result or "judge_result" not in result:
            return result
        result["judge_result"] = [m for m in result["judge_result"] if not m.get("passed", True)]
        return result

    async def _judge_global_once(
        self,
        task_compare_withlabel: str,
        principles: List[str],
        principles_str: str = "",
    ) -> Optional[Dict]:
        """Single global level judge."""
        if not principles_str and principles:
            principles_str = (
                "<principles>\nPrinciples are advisory, not mandatory.\n" + "\n".join(principles) + "\n</principles>"
            )

        system_prompt = build_judge_system_prompt("global", principles_str)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": GLOBAL_LEVEL_JUDGE_USER.format(
                    dependency=json.dumps(self.agent_dependency, ensure_ascii=False),
                    agent_prompts=json.dumps(self.agent_settings.get("prompt", {}), ensure_ascii=False),
                    tool_definitions=json.dumps(self.agent_settings.get("tool", {}), ensure_ascii=False),
                    system_task=json.dumps(self.agent_steps[0]["agent"]["input"], ensure_ascii=False),
                    system_output=self.agent_steps[-1]["agent"]["output"],
                    system_task_compare_withlabel=task_compare_withlabel,
                    agent_steps=json.dumps([self._format_step(s) for s in self.agent_steps], ensure_ascii=False),
                ),
            },
        ]

        try:
            response = await acall_llm(messages, self.config, output_schema=JudgeOutput)
            result = json.loads(response)
        except RetryableError:
            raise

        except Exception as e:
            logger.error(
                f"Global judge failed: steps={len(self.agent_steps)}, task_status={task_compare_withlabel}, err={e}"
            )
            return {"statement_action": [], "judge_result": [], "fault_sorted": [], "is_success": None}

        result = self._add_agent_to_result(result)
        result = self._expand_fault_types(result)
        result = self._filter_passed_metrics(result)

        # Log with pass count and fault count
        failed_cnt = len(result.get("judge_result", []))
        fault_cnt = len(result.get("fault_sorted", []))
        logger.info(
            f"Global judge done: ok={result.get('is_success')}, failed={failed_cnt}/{len(METRICS)}, faults={fault_cnt}"
        )
        return result

    async def _judge_agent_once(
        self,
        segment: Dict,
        task_compare_withlabel: str,
        principles: List[str],
        principles_str: str = "",
    ) -> Optional[Dict]:
        """Single agent level judge."""
        if not principles_str and principles:
            principles_str = (
                "<principles>\nPrinciples are advisory, not mandatory.\n" + "\n".join(principles) + "\n</principles>"
            )

        system_prompt = build_judge_system_prompt("agent", principles_str)

        agent_steps = self.agent_steps[segment["start_idx"] : segment["end_idx"] + 1]

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": AGENT_LEVEL_JUDGE_USER.format(
                    dependency=json.dumps(self.agent_dependency, ensure_ascii=False),
                    agent_prompts=json.dumps(self.agent_settings.get("prompt", {}), ensure_ascii=False),
                    tool_definitions=json.dumps(self.agent_settings.get("tool", {}), ensure_ascii=False),
                    system_task=json.dumps(self.agent_steps[0]["agent"]["input"], ensure_ascii=False),
                    system_output=self.agent_steps[-1]["agent"]["output"],
                    system_task_compare_withlabel=task_compare_withlabel,
                    agent_name=segment["agent_name"],
                    start_step=segment["start_step"],
                    end_step=segment["end_step"],
                    agent_steps=json.dumps([self._format_step(s) for s in agent_steps], ensure_ascii=False),
                ),
            },
        ]

        try:
            response = await acall_llm(messages, self.config, output_schema=JudgeOutput)
            result = json.loads(response)
        except RetryableError:
            raise  # Propagate retryable error
        except Exception as e:
            logger.error(
                f"Agent judge failed: agent={segment['agent_name']}, steps=[{segment['start_step']}-{segment['end_step']}], err={e}"
            )
            return {
                "statement_action": [],
                "judge_result": [],
                "fault_sorted": [],
                "is_success": None,
                "agent_name": segment["agent_name"],
                "start_step": segment["start_step"],
                "end_step": segment["end_step"],
            }

        # Add agent info to result
        result = self._add_agent_to_result(result)
        result = self._expand_fault_types(result)
        result = self._filter_passed_metrics(result)
        # Add segment info
        result["agent_name"] = segment["agent_name"]
        result["start_step"] = segment["start_step"]
        result["end_step"] = segment["end_step"]

        failed_cnt = len(result.get("judge_result", []))
        fault_cnt = len(result.get("fault_sorted", []))
        logger.info(
            f"Agent judge: {segment['agent_name']}[{segment['start_step']}-{segment['end_step']}], ok={result.get('is_success')}, failed={failed_cnt}/{len(METRICS)}, faults={fault_cnt}"
        )
        return result

    async def _judge_agents_batch(
        self,
        segments: List[Dict],
        task_compare_withlabel: str,
        agent_principles: Dict[str, List[str]],
        principles_str: str = "",
        max_tokens: int = 131072,
        reserve: int = 65536,
    ) -> List[Dict]:
        """Batch judge multiple agent segments in fewer LLM calls"""
        if not segments:
            return []

        # Pre-build segment data for token counting
        seg_data = []
        for seg in segments:
            steps = self.agent_steps[seg["start_idx"] : seg["end_idx"] + 1]
            seg_data.append(
                {
                    "agent_name": seg["agent_name"],
                    "start_step": seg["start_step"],
                    "end_step": seg["end_step"],
                    "steps": [self._format_step(s) for s in steps],
                }
            )

        # Estimate token count for each segment (use json string length as proxy)
        seg_tokens = [json.dumps(d, ensure_ascii=False) for d in seg_data]
        step_counts = [seg["end_idx"] - seg["start_idx"] + 1 for seg in segments]
        batches = _batch_by_token_limit(seg_tokens, max_tokens, reserve, step_counts)
        logger.info(
            f"Agent batch judge: {len(segments)} segs -> {len(batches)} batches, indices={[len(b) for b in batches]}"
        )

        all_results = [None] * len(segments)
        has_retryable = False

        async def process_batch(batch_idx: int, indices: List[int]) -> None:
            nonlocal has_retryable
            """Process one batch and fill results"""

            # Build batch input with relative indices (0, 1, 2... within batch)
            batch_inputs = []
            for rel_idx, orig_idx in enumerate(indices):
                data = seg_data[orig_idx].copy()
                data["seg_idx"] = rel_idx  # Use relative index within batch
                batch_inputs.append(json.dumps(data, ensure_ascii=False))

            # Collect principles for this batch
            batch_principles = set()
            for i in indices:
                batch_principles.update(agent_principles.get(segments[i]["agent_name"], []))
            p_str = principles_str or (
                f"<principles>\nPrinciples are advisory.\n{chr(10).join(batch_principles)}\n</principles>"
                if batch_principles
                else ""
            )

            system_prompt = (
                build_judge_system_prompt("agent", p_str)
                + """
<batch_mode>
You will judge MULTIPLE agent segments in one call.
CRITICAL: 
- seg_idx in output = seg_idx from input (0, 1, 2...)
- seg_idx is NOT start_step or end_step
- Example: input has seg_idx=0,1,2 → output must have seg_idx=0,1,2
<batch_example>
Input segments: [{"seg_idx": 0, "agent_name": "planner", ...}, {"seg_idx": 1, "agent_name": "searcher", ...}]
Output: {"results": [{"seg_idx": 0, "statement_action": [...], "judge_result": [...], "is_success": true}, {"seg_idx": 1, "statement_action": [...], "judge_result": [...], "is_success": false}]}
</batch_example>
</batch_mode>"""
            )

            user_content = f"""<context>
<dependency>{json.dumps(self.agent_dependency, ensure_ascii=False)}</dependency>
<agent_prompts>{json.dumps(self.agent_settings.get("prompt", {}), ensure_ascii=False)}</agent_prompts>
<tools>{json.dumps(self.agent_settings.get("tool", {}), ensure_ascii=False)}</tools>
<task>{json.dumps(self.agent_steps[0]["agent"]["input"], ensure_ascii=False)}</task>
<output>{self.agent_steps[-1]["agent"]["output"]}</output>
<status>{task_compare_withlabel}</status>
</context>
<segments>
{chr(10).join(batch_inputs)}
</segments>
Judge each segment. Return results with seg_idx matching input."""

            try:
                resp = await acall_llm(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                    self.config,
                    output_schema=BatchAgentJudgeOutput,
                )
                batch_result = json.loads(resp)
                # Handle both {"results": [...]} and direct [...] formats
                results_list = (
                    batch_result.get("results", batch_result) if isinstance(batch_result, dict) else batch_result
                )
                # Map results back: seg_idx is relative index within batch
                matched_count = 0
                unmatched_indices = set(range(len(indices)))
                for r in results_list:
                    if not isinstance(r, dict):
                        logger.warning(f"Batch {batch_idx+1}: invalid result type={type(r)}")
                        continue
                    rel_idx = r.get("seg_idx")
                    if rel_idx is None:
                        logger.warning(f"Batch {batch_idx+1}: missing seg_idx in result")
                        continue
                    # Fallback: if seg_idx looks like step number, try to find matching segment
                    if not (0 <= rel_idx < len(indices)):
                        # Try matching by start_step (LLM might have used step number)
                        for try_rel, try_orig in enumerate(indices):
                            if segments[try_orig]["start_step"] == rel_idx:
                                logger.warning(
                                    f"Batch {batch_idx+1}: seg_idx={rel_idx} looks like start_step, remapped to {try_rel}"
                                )
                                rel_idx = try_rel
                                break
                    if not (0 <= rel_idx < len(indices)):
                        logger.warning(f"Batch {batch_idx+1}: seg_idx={rel_idx} out of range [0,{len(indices)})")
                        continue
                    if not r.get("judge_result") and r.get("is_success") is None:
                        logger.warning(f"Batch {batch_idx+1}: seg_idx={rel_idx} missing judge_result and is_success")
                        continue
                    orig_idx = indices[rel_idx]
                    seg = segments[orig_idx]
                    unmatched_indices.discard(rel_idx)
                    result = {
                        "statement_action": r.get("statement_action", []),
                        "judge_result": r.get("judge_result", []),
                        "fault_sorted": r.get("fault_sorted", []),
                        "is_success": r.get("is_success"),
                        "agent_name": seg["agent_name"],
                        "start_step": seg["start_step"],
                        "end_step": seg["end_step"],
                    }
                    all_results[orig_idx] = self._filter_passed_metrics(
                        self._expand_fault_types(self._add_agent_to_result(result))
                    )
                    matched_count += 1
                if unmatched_indices:
                    logger.warning(
                        f"Batch {batch_idx+1}: unmatched seg_idx={list(unmatched_indices)}, will retry individually"
                    )
                    # Mark unmatched for individual retry
                    for rel_idx in unmatched_indices:
                        orig_idx = indices[rel_idx]
                        seg = segments[orig_idx]
                        # Leave as None - will be retried or marked failed
                        logger.info(
                            f"Segment {seg['agent_name']}[{seg['start_step']}-{seg['end_step']}] unmatched, needs retry"
                        )
                logger.info(
                    f"Batch {batch_idx+1}/{len(batches)}: segs={len(indices)}, matched={matched_count}, unmatched={list(unmatched_indices)}"
                )
            except RetryableError:
                has_retryable = True
                logger.warning(f"Batch {batch_idx+1}/{len(batches)} retryable, {len(indices)} segs skipped")
                return  # Don't mark as failed, don't fill all_results
            except Exception as e:
                logger.error(f"Batch {batch_idx+1}/{len(batches)} failed: {e}, marking {len(indices)} segs as failed")

                # Mark as failed directly (no retry - likely to fail again)
                for i in indices:
                    seg = segments[i]
                    all_results[i] = {
                        "statement_action": [],
                        "judge_result": [],
                        "fault_sorted": [],
                        "is_success": None,
                        "agent_name": seg["agent_name"],
                        "start_step": seg["start_step"],
                        "end_step": seg["end_step"],
                    }

        # Concurrent batch processing - exceptions handled inside process_batch
        await asyncio.gather(*[process_batch(i, indices) for i, indices in enumerate(batches)])
        # Check for missing results and log
        valid_results = [r for r in all_results if r is not None]
        if len(valid_results) < len(segments):
            logger.warning(
                f"Agent batch incomplete: {len(valid_results)}/{len(segments)} segs, retryable={has_retryable}"
            )
        return valid_results

    async def judge_once(
        self,
        task_compare_withlabel: str = "unknown",
        global_level: bool = True,
        agent_level: bool = False,
        global_principles: List[str] = None,
        agent_principles: Dict[str, List[str]] = None,
        principles_str: str = "",
        batch_agent: bool = True,
        max_tokens: int = 131072,
        reserve: int = 65536,
    ) -> Dict:
        """Single judge call for global and/or agent level"""
        global_principles = global_principles or []
        agent_principles = agent_principles or {}
        output = {}

        # Concurrent global and agent judge
        segments = self._identify_segments() if agent_level else []

        async def do_global():
            if not global_level:
                return None
            return await self._judge_global_once(task_compare_withlabel, global_principles, principles_str)

        async def do_agent():
            if not segments:
                return []
            if batch_agent and len(segments) > 1:
                return await self._judge_agents_batch(
                    segments, task_compare_withlabel, agent_principles, principles_str, max_tokens, reserve
                )
            tasks = [
                self._judge_agent_once(
                    seg, task_compare_withlabel, agent_principles.get(seg["agent_name"], []), principles_str
                )
                for seg in segments
            ]
            return [r for r in await asyncio.gather(*tasks) if r]

        # Use return_exceptions to avoid losing successful results when one fails
        results = await asyncio.gather(do_global(), do_agent(), return_exceptions=True)
        global_result, agent_results = results[0], results[1]

        # Check for RetryableError - only raise if ALL failed with retryable
        global_retryable = isinstance(global_result, RetryableError)
        agent_retryable = isinstance(agent_results, RetryableError)

        if global_retryable and agent_retryable:
            raise global_result  # Both failed with retryable, propagate

        # Handle partial success
        if not isinstance(global_result, Exception) and global_result:
            output["global_result"] = global_result
        elif global_retryable:
            logger.warning(f"Global judge retryable, agent may have succeeded")

        if not isinstance(agent_results, Exception) and agent_results:
            output["agent_results"] = agent_results
        elif agent_retryable:
            logger.warning(f"Agent judge retryable, global may have succeeded")

        # Log summary
        g_ok = "ok" if "global_result" in output else "skip"
        a_cnt = len(output.get("agent_results", []))
        a_ok = f"ok({a_cnt})" if a_cnt else "skip"
        logger.info(f"judge_once done: global={g_ok}, agent={a_ok}")

        return output

    async def _extract_global_principle(
        self,
        task_compare_withlabel: str,
        judge_results: List[Dict],
    ) -> Optional[Dict]:
        """Extract principles from global judge results."""
        if not judge_results:
            return {"execute_principle": [], "judge_principle": []}

        context = {"task_compare_withlabel": task_compare_withlabel}

        messages = [
            {
                "role": "system",
                "content": build_global_insight_system_prompt(
                    execute_principle_range=self.config.execute_principle_range,
                    judge_principle_range=self.config.judge_principle_range,
                ),
            },
            {
                "role": "user",
                "content": GLOBAL_LEVEL_INSIGHT_USER.format(
                    dependency=json.dumps(self.agent_dependency, ensure_ascii=False),
                    agent_prompts=json.dumps(self.agent_settings.get("prompt", {}), ensure_ascii=False),
                    tool_definitions=json.dumps(self.agent_settings.get("tool", {}), ensure_ascii=False),
                    system_task=json.dumps(self.agent_steps[0]["agent"]["input"], ensure_ascii=False),
                    system_output=self.agent_steps[-1]["agent"]["output"],
                    system_task_compare_withlabel=task_compare_withlabel,
                    agent_steps=json.dumps([self._format_step(s) for s in self.agent_steps], ensure_ascii=False),
                    judge_results=json.dumps({"context": context, "results": judge_results}, ensure_ascii=False),
                ),
            },
        ]

        try:
            response = await acall_llm(messages, self.config, output_schema=InsightOutput)
            result = json.loads(response)
        except RetryableError:
            raise  # Propagate for caller to handle
        except Exception as e:
            logger.error(
                f"Global extract failed: results_count={len(judge_results)}, task_status={task_compare_withlabel}, err={e}"
            )
            return {"execute_principle": [], "judge_principle": []}

        logger.info(
            f"Global extract done: exec={len(result.get('execute_principle', []))}, judge={len(result.get('judge_principle', []))}"
        )
        return result

    async def _extract_agent_principle(
        self,
        segment: Dict,
        task_compare_withlabel: str,
        judge_results: List[Dict],
    ) -> Optional[Dict]:
        """Extract principles from agent judge results."""
        if not judge_results:
            return {
                "execute_principle": [],
                "judge_principle": [],
                "agent_name": segment["agent_name"],
                "start_step": segment["start_step"],
            }

        context = {"agent_name": segment["agent_name"]}
        agent_steps = self.agent_steps[segment["start_idx"] : segment["end_idx"] + 1]
        messages = [
            {
                "role": "system",
                "content": build_agent_insight_system_prompt(
                    execute_principle_range=self.config.execute_principle_range,
                    judge_principle_range=self.config.judge_principle_range,
                ),
            },
            {
                "role": "user",
                "content": AGENT_LEVEL_INSIGHT_USER.format(
                    dependency=json.dumps(self.agent_dependency, ensure_ascii=False),
                    agent_prompts=json.dumps(self.agent_settings.get("prompt", {}), ensure_ascii=False),
                    tool_definitions=json.dumps(self.agent_settings.get("tool", {}), ensure_ascii=False),
                    system_task=json.dumps(self.agent_steps[0]["agent"]["input"], ensure_ascii=False),
                    system_output=self.agent_steps[-1]["agent"]["output"],
                    system_task_compare_withlabel=task_compare_withlabel,
                    agent_name=segment["agent_name"],
                    start_step=segment["start_step"],
                    end_step=segment["end_step"],
                    agent_steps=json.dumps([self._format_step(s) for s in agent_steps], ensure_ascii=False),
                    judge_results=json.dumps({"context": context, "results": judge_results}, ensure_ascii=False),
                ),
            },
        ]

        try:
            response = await acall_llm(messages, self.config, output_schema=InsightOutput)
            result = json.loads(response)
        except RetryableError:
            raise  # Propagate for caller to handle
        except Exception as e:
            logger.error(f"Agent extract failed: {segment['agent_name']}, {e}")
            return {
                "execute_principle": [],
                "judge_principle": [],
                "agent_name": segment["agent_name"],
                "start_step": segment["start_step"],
            }

        result["agent_name"] = segment["agent_name"]
        result["start_step"] = segment["start_step"]
        logger.info(f"Agent extract done: {segment['agent_name']}")
        return result

    async def _extract_agents_principle_batch(
        self,
        segments: List[Dict],
        task_compare_withlabel: str,
        agent_judge_results: Dict[str, List[Dict]],
    ) -> Dict:
        """Extract principles grouped by agent_name"""
        # Group segments and results by agent_name
        agent_data: Dict[str, Dict] = {}  # {agent_name: {"segments": [...], "results": [...]}}
        for seg in segments:
            key = f"{seg['agent_name']}_{seg['start_step']}"
            results = agent_judge_results.get(key, [])
            if not results:
                continue
            agent_name = seg["agent_name"]
            if agent_name not in agent_data:
                agent_data[agent_name] = {"segments": [], "results": [], "steps": []}
            agent_data[agent_name]["segments"].append(seg)
            agent_data[agent_name]["results"].extend(results)
            agent_steps = self.agent_steps[seg["start_idx"] : seg["end_idx"] + 1]
            agent_data[agent_name]["steps"].extend([self._format_step(s) for s in agent_steps])

        if not agent_data:
            return {"execute_principle": [], "judge_principle": []}

        logger.info(f"Agent extract by name: {list(agent_data.keys())}")
        system_prompt = build_agent_insight_system_prompt(
            execute_principle_range=self.config.execute_principle_range,
            judge_principle_range=self.config.judge_principle_range,
        )

        async def extract_one(agent_name: str, data: Dict) -> Dict:
            """Extract principles for one agent"""
            context = {"agent_name": agent_name}

            user_content = f"""<context>
<dependency>{json.dumps(self.agent_dependency, ensure_ascii=False)}</dependency>
<agent_prompts>{json.dumps(self.agent_settings.get("prompt", {}), ensure_ascii=False)}</agent_prompts>
<tools>{json.dumps(self.agent_settings.get("tool", {}), ensure_ascii=False)}</tools>
<task>{json.dumps(self.agent_steps[0]["agent"]["input"], ensure_ascii=False)}</task>
<output>{self.agent_steps[-1]["agent"]["output"]}</output>
<status>{task_compare_withlabel}</status>
</context>
<agent>
<agent_name>{agent_name}</agent_name>
</agent>
<execution_steps>{json.dumps(data["steps"], ensure_ascii=False)}</execution_steps>
<judge_results>{json.dumps({"context": context, "results": data["results"]}, ensure_ascii=False)}</judge_results>
Analyze all context. Return ONLY the JSON output."""

            try:
                resp = await acall_llm(
                    [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}],
                    self.config,
                    output_schema=InsightOutput,
                )
                result = json.loads(resp)
                for p in result.get("execute_principle", []):
                    p["agent_name"] = agent_name
                for p in result.get("judge_principle", []):
                    p["agent_name"] = agent_name
                logger.info(
                    f"Agent extract done: {agent_name}, exec={len(result.get('execute_principle', []))}, judge={len(result.get('judge_principle', []))}"
                )
                return result
            except RetryableError:
                raise  # Propagate for gather to handle
            except Exception as e:
                logger.error(f"Agent extract failed: {agent_name}, {e}")
                return {"execute_principle": [], "judge_principle": []}

        all_exec_p, all_judge_p = [], []
        # Concurrent extraction - semaphore controls actual concurrency
        results = await asyncio.gather(
            *[extract_one(name, data) for name, data in agent_data.items()], return_exceptions=True
        )
        has_retryable = False
        for r in results:
            if isinstance(r, RetryableError):
                has_retryable = True
                continue
            if isinstance(r, Exception):
                continue  # Already logged in extract_one
            all_exec_p.extend(r.get("execute_principle", []))
            all_judge_p.extend(r.get("judge_principle", []))
        # If all failed with retryable, propagate
        if has_retryable and not all_exec_p and not all_judge_p:
            raise RetryableError("All agent extracts failed with retryable error")
        return {"execute_principle": all_exec_p, "judge_principle": all_judge_p}

    async def extract_principle(
        self,
        task_compare_withlabel: str = "unknown",
        global_level: bool = True,
        agent_level: bool = False,
        global_judge_results: List[Dict] = None,
        agent_judge_results: Dict[str, List[Dict]] = None,
        batch_agent: bool = True,
    ) -> Dict:
        """Extract principles from judge results"""
        global_judge_results = global_judge_results or []
        agent_judge_results = agent_judge_results or {}
        output = {}

        async def do_global():
            if not (global_level and global_judge_results):
                return None
            return await self._extract_global_principle(task_compare_withlabel, global_judge_results)

        async def do_agent():
            if not agent_level:
                return None
            segments = self._identify_segments()
            if not segments:
                return {"execute_principle": [], "judge_principle": []}
            return await self._extract_agents_principle_batch(segments, task_compare_withlabel, agent_judge_results)

        results = await asyncio.gather(do_global(), do_agent(), return_exceptions=True)
        global_result, agent_result = results[0], results[1]

        # Check for RetryableError - only raise if ALL failed with retryable
        global_retryable = isinstance(global_result, RetryableError)
        agent_retryable = isinstance(agent_result, RetryableError)
        if global_retryable and agent_retryable:
            raise global_result  # Both failed with retryable, propagate

        # Handle partial success with detailed logging
        if not isinstance(global_result, Exception) and global_result:
            output["global_principle"] = global_result
            exec_cnt = len(global_result.get("execute_principle", []))
            judge_cnt = len(global_result.get("judge_principle", []))
            logger.info(f"extract_principle: global success (exec={exec_cnt}, judge={judge_cnt})")
        elif global_retryable:
            logger.warning("extract_principle: global retryable, will retry on restart")
        elif isinstance(global_result, Exception):
            logger.error(f"extract_principle: global failed permanently, err={global_result}")

        if not isinstance(agent_result, Exception) and agent_result is not None:
            output["agent_principle"] = agent_result
            exec_cnt = len(agent_result.get("execute_principle", []))
            judge_cnt = len(agent_result.get("judge_principle", []))
            logger.info(f"extract_principle: agent success (exec={exec_cnt}, judge={judge_cnt})")
        elif agent_retryable:
            logger.warning("extract_principle: agent retryable, will retry on restart")
        elif isinstance(agent_result, Exception):
            logger.error(f"extract_principle: agent failed permanently, err={agent_result}")

        # Summary log
        has_global = "global_principle" in output
        has_agent = "agent_principle" in output
        logger.info(f"extract_principle done: global={has_global}, agent={has_agent}")
        return output

    async def judge_and_extract(
        self,
        task_compare_withlabel: str = "unknown",
        global_level: bool = True,
        agent_level: bool = False,
        global_principles: List[str] = None,
        agent_principles: Dict[str, List[str]] = None,
        principles_str: str = "",
        do_extract: bool = True,
    ) -> Dict:
        """Run judge once, then optionally extract principles"""
        result = await self.judge_once(
            task_compare_withlabel=task_compare_withlabel,
            global_level=global_level,
            agent_level=agent_level,
            global_principles=global_principles,
            agent_principles=agent_principles,
            principles_str=principles_str,
        )
        # Build results dict
        global_judge_results = [result["global_result"]] if result.get("global_result") else []
        agent_judge_results = {}
        for ar in result.get("agent_results", []):
            key = f"{ar['agent_name']}_{ar['start_step']}"
            agent_judge_results[key] = [ar]
        output = {"global_judge_results": global_judge_results, "agent_judge_results": agent_judge_results}
        if do_extract:
            principles = await self.extract_principle(
                task_compare_withlabel=task_compare_withlabel,
                global_level=global_level,
                agent_level=agent_level,
                global_judge_results=global_judge_results,
                agent_judge_results=agent_judge_results,
            )
            output.update(principles)
        return output
