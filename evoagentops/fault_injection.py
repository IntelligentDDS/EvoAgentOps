# fault_injection.py
from .config import Config
from .util import logger, acall_llm, set_call_context
from .prompt import ALL_FAULTS, _fault_code
import uuid
import time
from pydantic import BaseModel, Field, ConfigDict, field_validator
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import re


class MessageItem(BaseModel):
    """Message in agent input"""

    role: str
    content: str


class ToolCallItem(BaseModel):
    """Tool call record"""

    tool_name: str
    tool_args: dict  # JSON string like '{"query": "..."}'
    tool_response: Optional[str] = None


class AgentData(BaseModel):
    """Agent input/output structure"""

    input: List[MessageItem]
    output: Any  # Can be str or dict {"role": "assistant", "content": "..."}
    tools_called: List[ToolCallItem] = Field(default_factory=list)


class StepObject(BaseModel):
    """Complete step structure for OpenAI strict schema"""

    step: int
    agent_name: str
    agent: AgentData


class FaultOutput(BaseModel):
    """LLM output schema - answers extracted by code from steps"""

    causality_chain: str = Field(
        description="Plan how fault leads to wrong answer: fault at step N -> error propagation -> wrong final answer."
    )
    root_cause_step: int = Field(description="The step number where fault is injected (root cause of failure).")
    is_modify_to_final_step: bool = Field(
        description="Whether the modification affects the final answer (not necessarily last step)."
    )
    modify_description: list[str] = Field(
        description="Description of modified steps. Must include 'Final answer changed from: <original> to: <wrong>'."
    )
    steps: List[StepObject] = Field(
        description="The modified steps list. Use exactly 'steps' as field name, NOT 'modified_steps'."
    )
    original_correct_answer: str = Field(
        description="Traverse ALL original steps to locate the actual answer. Often in middle step's tool_response, NOT last step's output. Extract raw data."
    )
    wrong_final_answer: str = Field(
        description="Traverse ALL modified steps using same extraction logic. Extract raw data from same position. MUST differ from original_correct_answer."
    )

    @field_validator("steps", mode="before")
    @classmethod
    def parse_steps_string(cls, v):
        """Convert string to list if Claude returns serialized JSON"""
        if isinstance(v, str):
            return json.loads(v)
        return v


# Build from ALL_FAULTS (Single Source of Truth), use original name as key
FAULT_TYPES = {f.name: f"{f.name}:{f.description}" for f in ALL_FAULTS}
FAULT_CODE_TO_KEY = {_fault_code(i): ALL_FAULTS[i].name for i in range(len(ALL_FAULTS))}


class FaultType:
    """Fault types from ALL_FAULTS (Single Source of Truth)"""

    @classmethod
    def get(cls, key: str) -> str:
        """Get fault type string by key"""
        return FAULT_TYPES.get(key, "")

    @classmethod
    def all_keys(cls) -> list:
        """Get all fault type keys"""
        return list(FAULT_TYPES.keys())

    @classmethod
    def code_to_key(cls, code: str) -> str:
        return FAULT_CODE_TO_KEY.get(code, "")


def _build_fault_definition(fault_name: str) -> str:
    """Build fault definition for injection prompt."""
    for f in ALL_FAULTS:
        if f.name == fault_name:
            return f"Name: {f.name}\nDefinition: {f.description}\nCriteria: {f.criteria}\nExample: {f.example}"
    return f"Name: {fault_name}"


class FaultInjector:
    """Fault injector"""

    def __init__(self, agent_steps: List[Dict], agent_dependency: Dict, agent_settings: Dict, config: Config = None):
        """Initialize"""
        self.agent_steps = agent_steps
        self.agent_dependency = agent_dependency
        self.agent_settings = agent_settings
        self.config = config or Config()

    def _generate_span_id(self):
        """Generate 16-byte hex span ID"""
        return "0x" + uuid.uuid4().hex[:16]

    def _generate_trace_id(self):
        """Generate 32-byte hex trace ID"""
        return "0x" + uuid.uuid4().hex

    def _restore_fields(self, step, original_step=None):
        """Restore all required fields"""
        # Copy from original step
        if original_step:
            step["span_id"] = original_step.get("span_id", self._generate_span_id())
            step["trace_id"] = original_step.get("trace_id", "")
            step["parent_span_id"] = original_step.get("parent_span_id", "")
            step["step_usage"] = original_step.get(
                "step_usage",
                {"input_tokens": 0, "output_tokens": 0, "llm_inference_time": 0, "model": "", "step_execution_time": 0},
            )
            step["accumulated_usage"] = original_step.get(
                "accumulated_usage",
                {
                    "accumulated_input_tokens": 0,
                    "accumulated_output_tokens": 0,
                    "accumulated_time": 0,
                    "accumulated_transferred_times": 0,
                },
            )
            step["start_time"] = original_step.get("start_time", str(time.time_ns()))
            step["end_time"] = original_step.get("end_time", str(time.time_ns()))
            step["environment"] = original_step.get("environment", None)
        else:
            # Generate new fields for inserted steps
            trace_id = (
                self.agent_steps[0].get("trace_id", self._generate_trace_id())
                if self.agent_steps
                else self._generate_trace_id()
            )
            parent_span_id = (
                self.agent_steps[0].get("span_id", self._generate_span_id())
                if self.agent_steps
                else self._generate_span_id()
            )
            step["span_id"] = self._generate_span_id()
            step["trace_id"] = trace_id
            step["parent_span_id"] = parent_span_id
            step["step_usage"] = {
                "input_tokens": 0,
                "output_tokens": 0,
                "llm_inference_time": 0,
                "model": "injected",
                "step_execution_time": 0,
            }
            step["accumulated_usage"] = {
                "accumulated_input_tokens": 0,
                "accumulated_output_tokens": 0,
                "accumulated_time": 0,
                "accumulated_transferred_times": 0,
            }
            current_time = str(time.time_ns())
            step["start_time"] = current_time
            step["end_time"] = current_time
            step["environment"] = None
        return step

    async def inject(self, fault_type: str, force_last_steps: bool = False, require_final_change: bool = True) -> Dict:
        """Inject fault"""
        set_call_context(stage="fault_inject", case_id=None, idx=None)
        logger.info(f"Start fault injection: fault_type={fault_type}")
        # Clean steps
        clean_steps = []
        for item in self.agent_steps:
            clean_item = {
                k: v
                for k, v in item.items()
                if k
                not in [
                    "environment",
                    "step_usage",
                    "accumulated_usage",
                    "start_time",
                    "end_time",
                    "span_id",
                    "trace_id",
                    "parent_span_id",
                ]
            }
            clean_steps.append(clean_item)
        # Build step range constraint
        total_steps = len(clean_steps)
        if force_last_steps:
            last_n = 10
            min_step = max(1, total_steps - last_n + 1)
            step_hint = f"MUST be >= {min_step}"
        else:
            last_n = 20
            min_step = max(1, total_steps - last_n + 1)
            step_hint = f"MUST be >= {min_step}"
            # step_hint = "YOU decide where this fault type most likely occurs. Prefer earlier steps for better error propagation, don't focus only on final 10 steps."

        # Build steps output constraint based on mode
        if require_final_change:
            steps_constraint = f"- CRITICAL: Return ALL steps from root_cause_step to step {total_steps} (inclusive)\n- Do NOT skip any intermediate steps - include ALL steps in the range even if unchanged\n- Example: If root_cause_step=5 and total_steps=8, return [step5, step6, step7, step8]"
        else:
            steps_constraint = (
                "- Return ONLY the modified steps (can be non-contiguous)\n- No need to return unchanged steps"
            )

        # Build fault definition for LLM understanding
        fault_def = _build_fault_definition(fault_type)
        # Extract user question safely
        user_question = ""
        if self.agent_steps and self.agent_steps[0].get("agent", {}).get("input"):
            user_question = self.agent_steps[0]["agent"]["input"][0].get("content", "")

        # Build prompt
        prompt = f"""<system>
You are a fault injection expert for agent system testing.
Rules:
- Inject fault as ROOT CAUSE of wrong final answer
- Modified steps must be logical and coherent
- Agent must NOT recover from fault
- Extract ACTUAL final answers
- If unsure, output {{"error": "reason"}}
</system>

<instructions>
Goal: Inject ONE "{fault_type}" fault into agent steps to cause wrong final answer.

Success criteria:
1. wrong_final_answer MUST differ from original_correct_answer
2. Fault propagates logically to affect the final answer
3. Final answer location may be in middle step's tool_response, NOT necessarily in last step's output

Constraints:
- root_cause_step {step_hint}
{steps_constraint}
- For faults like Step Repetition, you may INSERT new steps (use step N+1)

Task steps:
1. Choose root_cause_step where this fault type most likely occurs
2. Plan causality: fault -> propagate -> wrong answer
3. Modify steps[root_cause_step:] to cause wrong final answer
4. Extract original_correct_answer from ORIGINAL steps (the final answer agent gives to user)
5. Extract wrong_final_answer from MODIFIED steps (must differ from original)
6. Describe all changes in modify_description
</instructions>

<context>
<fault_definition>
{fault_def}
</fault_definition>
<agent_dependency>
{json.dumps(self.agent_dependency, ensure_ascii=False, indent=2)}
</agent_dependency>
<agent_settings>
{json.dumps(self.agent_settings, ensure_ascii=False, indent=2)}
</agent_settings>
<user_question>
{user_question}
</user_question>
<agent_steps total="{total_steps}">
{json.dumps(clean_steps, ensure_ascii=False, indent=2)}
</agent_steps>
</context>

<fault_injection_format_examples>
IMPORTANT: Maintain the EXACT same format as the original steps. If original output is {{"role": "assistant", "content": "..."}}, keep that structure.
<text_only_interaction>tools_called must be empty list
{{"step": N, "agent_name": "...", "agent": {{"input": [...], "output": "text content", "tools_called": []}}}}
</text_only_interaction>
<tool_call_interaction>output must be empty string
{{"step": N, "agent_name": "...", "agent": {{"input": [...], "output": "", "tools_called": [{{"tool_name": "...", "tool_args": {{"key": "value"}}, "tool_response": "..."}}]}}}}
</tool_call_interaction>
</fault_injection_format_examples>

<output_format>
Return ONLY valid JSON:
{{
  "causality_chain": "fault@stepN -> propagate -> wrong",
  "root_cause_step": int,
  "is_modify_to_final_step": true,
  "modify_description": ["change1", "change2", "Final answer changed from: <original> to: <wrong>"],
  "steps": [step objects - see constraints above],
  "original_correct_answer": "Extract ACTUAL final answer from ORIGINAL steps",
  "wrong_final_answer": "Extract ACTUAL final answer from MODIFIED steps"
}}

Answer extraction rules:
CRITICAL - Final answer location:
- Answer is often in tool_response of MIDDLE steps, NOT in last step's output
- Last step typically contains summary/explanation, NOT raw answer
- Traverse ALL steps to find where actual data/result appears

Extraction principle:
- Extract raw answer (data, result, output), NOT summaries or explanations
- Example: Step 3 tool_response="[(1,'A')]", Step 4 output="Found 1 result"
  → Answer is "[(1,'A')]" from step 3, NOT "Found 1 result" from step 4

Step format:
- Text: {{"step": N, "agent_name": "...", "agent": {{"input": [...], "output": "text", "tools_called": []}}}}
- Tool: {{"step": N, "agent_name": "...", "agent": {{"input": [...], "output": "", "tools_called": [...]}}}}
</output_format>

<example>
<context>
fault_type: Tool Param Value Error
task_type: Stock price query
total_steps: 3
step1: planner -> "Plan to query AAPL stock price"
step2: stock_agent calls get_price(symbol="AAPL", days=7) -> tool_response="150.5"
step3: reporter outputs "The current price of AAPL is $150.5"
</context>
<output_format>
{{
  "causality_chain": "step2: days=-7 (invalid) -> API error -> wrong price returned -> wrong final answer",
  "root_cause_step": 2,
  "is_modify_to_final_step": true,
  "modify_description": ["Changed days=7 to days=-7 in get_price call", "API returns error", "Final answer changed from: 150.5 to: Error message"],
  "steps": [
    {{"step": 2, "agent_name": "stock_agent", "agent": {{"input": [...], "output": "", "tools_called": [{{"tool_name": "get_price", "tool_args": {{"symbol": "AAPL", "days": -7}}, "tool_response": "Error: invalid days parameter"}}]}}}},
    {{"step": 3, "agent_name": "reporter", "agent": {{"input": [...], "output": "Unable to get price: Error: invalid days parameter", "tools_called": []}}}}
  ],
  "original_correct_answer": "150.5",
  "wrong_final_answer": "Error: invalid days parameter"
}}
</output_format>
</example>

<example>
<context>
fault_type: Information Loss
total_steps: 3
step1: search_financial_data -> raw data retrieved
step2: extract_metrics calls extract tool -> tool_response="{{\\"revenue\\": \\"$5M\\", \\"growth\\": \\"15%\\"}}"
step3: format_report -> "Financial Summary: Revenue is $5M with 15% growth rate"
</context>
<output_format>
{{
  "causality_chain": "step2: omit growth field in extraction -> step3: incomplete data -> wrong final answer",
  "root_cause_step": 2,
  "is_modify_to_final_step": true,
  "modify_description": ["Removed growth field from extract tool output", "Step3 only has revenue data", "Final answer changed from: revenue=$5M, growth=15% to: revenue=$5M only"],
  "steps": [
    {{"step": 2, "agent_name": "extract_metrics", "agent": {{"input": [...], "output": "", "tools_called": [{{"tool_name": "extract", "tool_args": {{}}, "tool_response": "{{\\\"revenue\\\": \\\"$5M\\\"}}"}}]}}}},
    {{"step": 3, "agent_name": "format_report", "agent": {{"input": [...], "output": "Financial Summary: Revenue is $5M", "tools_called": []}}}}
  ],
  "original_correct_answer": "{{\\"revenue\\": \\"$5M\\", \\"growth\\": \\"15%\\"}}",
  "wrong_final_answer": "{{\\"revenue\\": \\"$5M\\"}}"
}}
</output_format>
</example>"""

        messages = [{"role": "user", "content": prompt}]

        # Call LLM
        response = await acall_llm(messages, self.config, output_schema=FaultOutput)

        # Parse JSON
        try:
            result = json.loads(response)
            # Fix steps if it's a string (Claude sometimes serializes nested arrays)
            if isinstance(result.get("steps"), str):
                result["steps"] = json.loads(result["steps"])
        except Exception as e:
            logger.error(f"Fault injection failed: JSON parse error {e}")
            return None
        # Validate LLM output
        if not result.get("steps"):
            logger.error(f"Injection invalid: empty steps")
            return None
        # root_cause_step should be in original steps range
        # (inserted steps may have step > total_steps, but root_cause should not)
        root_step = result.get("root_cause_step", 0)
        if root_step < 1 or root_step > len(clean_steps):
            logger.error(f"Injection invalid: root_cause_step={root_step} out of range [1,{len(clean_steps)}]")
            return None
        logger.info(f"Injection complete: root_cause_step={result['root_cause_step']}")

        # Merge steps
        modified_steps = self._merge_steps(self.agent_steps, result)

        # Use LLM-extracted final answers (not last step output)
        original_final = result.get("original_correct_answer", "").strip()
        modified_final = result.get("wrong_final_answer", "").strip()

        # Validate: LLM must provide both answers
        if not original_final or not modified_final:
            logger.error(
                f"Injection invalid: missing answers (orig={bool(original_final)}, wrong={bool(modified_final)})"
            )
            return None
        # Validate
        is_final_changed = original_final != modified_final
        if require_final_change and not is_final_changed:
            logger.warning(
                f"require_final_change=True but answers identical: orig='{original_final[:50]}', wrong='{modified_final[:50]}'"
            )
            return None

        # Create label
        label = {
            "fault_type": fault_type,
            "is_modify_to_final_step": is_final_changed,
            "modify_description": result["modify_description"],
            "root_cause_step": result["root_cause_step"],
            "original_correct_answer": original_final,
            "wrong_final_answer": modified_final,
            "causality_chain": result.get("causality_chain", ""),
        }

        logger.info(
            f"Injection done: fault={fault_type}, root_step={result['root_cause_step']}, "
            f"modified_steps={len(result['steps'])}/{len(modified_steps)}, "
            f"orig='{label['original_correct_answer'][:80]}', wrong='{label['wrong_final_answer'][:80]}'"
        )
        return {"agent_steps": modified_steps, "label": label}

    def _update_existing_step(self, original_steps: list, modified_step: dict) -> bool:
        """Update existing step if found. Return True if updated."""
        step_num = modified_step["step"]
        for orig in original_steps:
            if orig["step"] == step_num:
                orig["agent_name"] = modified_step["agent_name"]
                # Preserve original output format (dict vs string)
                new_agent = modified_step["agent"]
                orig_output = orig.get("agent", {}).get("output")
                new_output = new_agent.get("output", "")
                # If original is dict format, wrap new output in same structure
                if isinstance(orig_output, dict) and isinstance(new_output, str):
                    new_agent["output"] = {"role": "assistant", "content": new_output}
                orig["agent"] = new_agent
                self._restore_fields(orig, orig)
                return True
        return False

    def _merge_steps(self, original_steps, result):
        """Merge original and modified steps"""
        import copy

        original_steps = copy.deepcopy(original_steps)

        # Classify steps by type
        int_steps = []
        float_steps = []
        for item_i in result["steps"]:
            step_num = item_i["step"]
            if isinstance(step_num, int) or (isinstance(step_num, str) and step_num.isdigit()):
                int_steps.append(item_i)
            else:
                float_steps.append(item_i)

        # Update existing integer steps
        for item_i in int_steps:
            if not self._update_existing_step(original_steps, item_i):
                # If step not found, append it
                new_step = self._restore_fields(item_i)
                original_steps.append(new_step)

        # Insert float steps if any (for faults like Step Repetition)
        if float_steps:
            # Infer insert position from first float step (e.g., 3.1 -> after step 3)
            first_float = float_steps[0]["step"]
            step_val = float(first_float) if isinstance(first_float, str) else first_float
            dividing_step = int(step_val)
            logger.info(f"Insert {len(float_steps)} steps after step {dividing_step}")
            for item in float_steps:
                self._restore_fields(item)
            dividing_pre = original_steps[:dividing_step]
            dividing_post = original_steps[dividing_step:]
            new_agent_steps = dividing_pre + float_steps + dividing_post
            for index, item in enumerate(new_agent_steps):
                item["step"] = index + 1
            return new_agent_steps

        # Sort and return
        original_steps.sort(key=lambda x: x["step"])
        return original_steps

    def save(self, data, output_file: str):
        """Save injected data to file"""
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved: {output_file}")
