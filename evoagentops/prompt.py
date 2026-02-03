# prompt.py
from dataclasses import dataclass
from typing import List, Dict, Set
from enum import Enum


# =============================================================================
# PRINCIPLE DEFINITIONS - Single Source of Truth
# =============================================================================
PRINCIPLE_DEFS = {
    "execute": {
        "name": "execute_principle",
        "purpose": "Rule to improve AGENT behavior",
        "focus": "agent execution improvement",
        "audience": "Agent that executes tasks",
        "injection": "Injected into agent prompt to prevent faults",
        "format": "When [trigger], [action] by [method], avoiding [pitfall]",
        "title_pattern": "[Action] + [Scenario]",
        "content_pattern": "When [trigger], [action] by [method], avoiding [pitfall]",
        "covers": "tool selection, parameter validation, action sequence, error recovery",
        "includes": ["action decisions", "tool usage", "error recovery", "workflow sequencing"],
        "excludes": ["fault detection", "evaluation criteria", "quality metrics"],
        "keywords": ["tool", "operation", "validation", "parameter", "sequence"],
        "example": {
            "title": "Validate API parameters before request",
            "content": "When calling external API, validate required parameters exist and match expected types, avoiding runtime errors.",
        },
    },
    "judge": {
        "name": "judge_principle",
        "purpose": "Rule to improve EVALUATOR accuracy for fault detection",
        "focus": "fault detection improvement",
        "audience": "Evaluator that judges agent execution",
        "injection": "Injected into judge prompt to improve fault detection",
        "format": "When evaluating [metric/fault], check [criteria] and mark [result]",
        "title_pattern": "[Check] + [Target]",
        "content_pattern": "When evaluating [metric/fault], check [criteria] and mark [result]",
        "covers": "validation criteria, fault detection patterns, check conditions, failure signals",
        "includes": ["fault detection", "validation rules", "quality checks", "evidence requirements"],
        "excludes": ["agent actions", "tool usage guidance", "workflow recommendations"],
        "keywords": ["fault", "check", "validate", "detect", "verify"],
        "example": {
            "title": "Detect tool parameter type mismatch",
            "content": "When evaluating Tool Correctness, check if parameter type matches schema. Mark fault when typeof(value) differs from schema.type.",
        },
    },
}


def format_principles_for_prompt(cache_data: dict, top_k: int, scope: str = "both") -> str:
    """Format principles from retrieval cache for injection into prompts"""
    if not cache_data:
        return ""
    lines = ["<principles>", "Principles are advisory, not mandatory."]
    # Global level
    if scope in ("global", "both"):
        global_top = cache_data.get("global_sorted", [])[:top_k]
        if global_top:
            lines.append("# global level")
            for p in global_top:
                lines.append(f"title: {p['title']}")
                lines.append(f"content: {p['content']}")
    # Agent level
    if scope in ("agent", "both"):
        agent_section_added = False
        for agent_name, sorted_list in cache_data.get("agent_sorted", {}).items():
            agent_top = sorted_list[:top_k]
            if agent_top:
                if not agent_section_added:
                    lines.append("# agent level")
                    agent_section_added = True
                lines.append(f"## {agent_name}")
                for p in agent_top:
                    lines.append(f"title: {p['title']}")
                    lines.append(f"content: {p['content']}")
    lines.append("</principles>")
    return "\n".join(lines) if len(lines) > 3 else ""


def format_principle_oneline(ptype: str) -> str:
    """One-line: 'execute_principle: Rule to improve AGENT behavior.'"""
    p = PRINCIPLE_DEFS[ptype]
    return f"{p['name']}: {p['purpose']}."


def format_principle_definition(ptype: str, include_boundary: bool = False) -> str:
    """Multi-line definition block for prompts."""
    p = PRINCIPLE_DEFS[ptype]
    lines = [
        f"- {p['name']}: {p['purpose']}.",
        f"  Injection: {p['injection']}.",
        f"  Audience: {p['audience']}.",
        f"  Format: \"{p['format']}\".",
    ]
    if include_boundary:
        lines.append(f"  MUST include: {', '.join(p['includes'])}.")
        lines.append(f"  MUST NOT include: {', '.join(p['excludes'])}.")
    return "\n".join(lines)


def format_principle_example(ptype: str) -> str:
    """Example block for prompts."""
    p = PRINCIPLE_DEFS[ptype]
    ex = p["example"]
    return f'title="{ex["title"]}", content="{ex["content"]}"'


def get_keyword_system_prompt(ptype: str) -> str:
    """Generate keyword extraction system prompt."""
    p = PRINCIPLE_DEFS[ptype]
    example_str = format_principle_example(ptype)
    keywords_list = ", ".join(p["keywords"])
    # Build type-specific example
    if ptype == "execute":
        ex_task = "Query database for Q3 orders and compute total revenue"
        ex_output = '{"keywords": ["SQL", "NULL", "aggregation", "date filter"]}'
    else:  # judge
        ex_task = "Search latest AI news and return JSON summary"
        ex_output = '{"keywords": ["hallucination", "recency", "JSON", "source"]}'
    return f"""<system>You are a keyword extractor for {p['name']} retrieval.</system>
<context>
{p['name']}: {p['purpose']}. {p['injection']}.
Principle example: {example_str}
</context>
<instructions>
Goal: Extract 3-5 keywords to match {p['name']} titles/content.
Success criteria: Keywords appear in relevant principle text.
Focus on: {keywords_list}
</instructions>
<output_format>Return ONLY valid JSON: {{"keywords": ["kw1", "kw2", ...]}}</output_format>
<example>
<task>{ex_task}</task>
<output>{ex_output}</output>
</example>"""


def get_rerank_system_prompt(ptype: str) -> str:
    """Generate rerank system prompt for execute or judge principles."""
    p = PRINCIPLE_DEFS[ptype]
    if ptype == "execute":
        criteria = """Criteria:
1) Direct-match: principle applies to this task's operations
2) Actionable: principle gives concrete steps
3) Error-prevent: principle avoids likely mistakes"""
        example_task = "Query database for Q3 orders and compute total revenue"
        example_candidates = """[0] Handle NULL in aggregation: Use COALESCE or WHERE NOT NULL before SUM.
[1] Use date range filter: Use date >= start AND date < end for period queries.
[2] Set web request timeout: Always set timeout=30s for HTTP requests.
[3] Limit result size: Add LIMIT for large dataset queries."""
        example_output = '{"sorted_indices": [1, 0, 3, 2], "reason": "1:Q3 needs date filter; 0:SUM needs NULL handling; 3:large data possible; 2:no web request"}'
    else:  # judge
        criteria = """Criteria:
1) Fault-match: principle detects faults likely in this task type
2) Specificity: principle targets specific errors, not generic checks
3) Actionable: principle has clear pass/fail criteria"""
        example_task = "Search latest AI news and return JSON summary"
        example_candidates = """[0] Detect hallucination: Check if output claims are absent from search results.
[1] Verify recency: Check if "latest" requirement is satisfied by result dates.
[2] Validate JSON: Check JSON syntax and required fields.
[3] Check tool params: Verify tool parameters match schema types."""
        example_output = '{"sorted_indices": [0, 1, 2, 3], "reason": "0:hallucination is critical; 1:recency matches task; 2:output format; 3:generic"}'

    return f"""<system>You rank {p['name']}s by relevance.</system>
<context>
{p['name']}: {p['purpose']}.
Task: SORT all candidates by relevance for {"executing" if ptype == "execute" else "detecting faults in"} given task.
</context>
<instructions>
Goal: Rank ALL indices (0 to n-1) by relevance.
{criteria}
MUST include ALL indices. No omission.
</instructions>
<output_format>{{"sorted_indices": [...], "reason": "brief rationale"}}</output_format>
<example>
<task>{example_task}</task>
<candidates>
{example_candidates}
</candidates>
<output>{example_output}</output>
</example>"""


# =============================================================================
# FAULT DEFINITIONS - Single Source of Truth
# =============================================================================


@dataclass
class FaultDef:
    name: str
    description: str
    criteria: str  # Binary check question
    example: str


# Define faults
_FAULT_DEFS = [
    (
        "Hallucination",
        "Output contains claims absent from all sources. Fabricated facts, invented data, or fake references.",
        "Does the output contain fabricated or unsupported claims?",
        'Typically occurs in step.tool_calls and step.output. Normal: calls flight_search() returns data then outputs result. Fault: tool_calls=[] but output="Query returned CA1234¥800/MU5678¥750/HU9012¥820, recommend MU5678", causing downstream booking step to use fake flight number MU5678, API returns NotFound error, transaction fails',
    ),
    (
        "Information Loss",
        "Output omits relevant data from context. Available information not used.",
        "Does the output omit critical information available in the context?",
        'Typically occurs in step.output. Normal: tool_response=[{"name":"Hilton","price":800},{"name":"Home Inn","price":200},{"name":"Marriott","price":1200}] iterate to find lowest price 200. Fault: only uses "Hilton 800", causing user to pay 800, actual lowest price 200, overpay 600',
    ),
    (
        "Task Misunderstanding",
        "Output violates explicit task constraints. Wrong format, scope, or requirements.",
        "Does output violate any constraint stated in task?",
        'Typically occurs in step.action_input. Normal: task="search 2024 Q1 data (2024-01-01 to 2024-03-31)" searches start_date="2024-01-01". Fault: searches start_date="2023-01-01", end_date="2023-12-31", causing report to use last year data, trend analysis completely deviates',
    ),
    (
        "Role Violation",
        "Agent acts outside defined scope. Unauthorized action executed.",
        "Does agent execute action not in role_definition?",
        'Typically occurs in step.action. Normal: search_agent.allowed_actions=[search,filter] only calls these two tools. Fault: calls send_email(to="boss@company.com",subject="competitor alert"), action not in allowed_actions list, causing unauthorized email sent, business intelligence leaked',
    ),
    (
        "Output Format Error",
        "Output fails to parse. Malformed JSON, XML, or structure.",
        "Does output fail to parse in expected format?",
        'Typically occurs in step.output. Normal: outputs {"thought":"...","action":"...","action_input":{...}}. Fault: outputs "I think should search product info first, then filter price" (plain text), causing JSON.parse() throws SyntaxError, step execution fails needs regeneration',
    ),
    (
        "Output Truncation",
        "Output cut off before completion. Unclosed brackets, incomplete sentences.",
        "Does output contain unclosed structures or incomplete content?",
        'Typically occurs in step.output. Normal: complete output: {"price":{"min":500,"max":1000}}. Fault: output: {"price":{"min":500 (truncated after this)',
    ),
    (
        "Logic-Action Mismatch",
        "Stated thought contradicts executed action. Intent and behavior misaligned.",
        "Does thought contradict action semantics?",
        'Typically occurs in (step.thought,step.action) pair. Normal: thought="should use read_file to read content"→action=read_file. Fault: thought="should use read_file to read content"→action=delete_file, action_input={"file":"report.pdf"}, causing file permanently deleted, user data lost',
    ),
    (
        "Tool Selection Error",
        "Wrong tool chosen. Suitable tool ignored or unsuitable tool used.",
        "Does agent use unsuitable tool or ignore suitable tool?",
        'Typically occurs in step.action. Normal: task="calculate 15% discount price, original price 1200" selects calculator tool. Fault: selects translate_text tool, passes action_input={"text":"1200*0.85","target_lang":"en"}, causing return text "one thousand two hundred times...", no correct result or error',
    ),
    (
        "Tool Name Error",
        "Tool name not in tool_definitions. Typo or hallucinated name.",
        "Is tool_name absent from tool_definitions?",
        'Typically occurs in step.action. Normal: available_tools=["web_search","file_read","calculator"] calls web_search. Fault: calls web_serach (spelling error) or google_search (tool does not exist), causing throws exception ToolNotFoundError, framework returns error must correct and retry',
    ),
    (
        "Tool Param Name Error",
        "Parameter name not in schema. Invalid or misspelled name.",
        "Is param_name absent from schema.properties?",
        'Typically occurs in step.action_input. Normal: search tool schema={query:str(required)} passes query="AI". Fault: passes qeury="AI" (parameter name spelling error), causing InvalidParameterError: qeury not defined, tool returns error',
    ),
    (
        "Tool Param Type Error",
        "Parameter type mismatches schema. Wrong data type provided.",
        "Does param_type mismatch schema type?",
        'Typically occurs in step.action_input. Normal: pagination tool schema={page_num:int} passes page_num=5. Fault: passes page_num=["5"] (list type), causing TypeError: expected int got list, tool returns type error',
    ),
    (
        "Tool Param Value Error",
        "Parameter value violates constraints. Out of range or invalid enum.",
        "Does param_value violate schema constraints?",
        "Typically occurs in step.action_input. Normal: date_filter tool schema={month:int(1-12)} passes month=3. Fault: passes month=2000 (passes year instead), causing ValueError: month=2000 violates [1,12], tool returns constraint error",
    ),
    (
        "Tool Param Missing",
        "Required parameter absent. schema.required param not provided.",
        "Is schema.required param not provided?",
        'Typically occurs in step.action_input. Normal: send_email tool required=[recipient,subject,body] all provided. Fault: only calls send_email(subject="notification",body="content") missing recipient parameter, causing MissingParameterError: recipient required, tool returns error',
    ),
    (
        "Tool Param Redundant",
        "Extra parameter not in schema. Undefined param provided.",
        "Is param provided but absent from schema.properties?",
        'Typically occurs in step.action_input. Normal: search tool schema={query,max_results} only passes these two parameters. Fault: passes search(query="AI",max_results=10,timeout=30,language="zh"), timeout and language not in schema, causing UnknownParameterError: timeout,language not in schema',
    ),
    (
        "Tool Definition Error",
        "Tool schema incomplete. Missing name, description, or parameters.",
        "Does tool_definition lack required fields?",
        'Typically occurs in tool_definition. Normal: tool description matches actual capability. Fault: Prompt describes db_search="search any database (user/product/order)" but actual code only supports product_catalog table, causing calls db_search(table="hr_salary") returns TableNotFoundError, Agent misjudges capability scope no alternative solution',
    ),
    (
        "Tool Execution Error",
        "Tool returns error status. HTTP 4xx/5xx or error flag.",
        "Does tool_response contain error code?",
        'Typically occurs in step.tool_response. Normal: returns {"status":200,"data":{...}}. Fault: returns {"status":429,"error":"Rate limit exceeded. Retry after 60s"}, status is 4xx/5xx or contains error field, causing if not handled directly outputs error, user receives error message not data',
    ),
    (
        "Tool Execution Timeout",
        "Tool exceeds time limit. Execution terminated.",
        "Does tool_response indicate timeout?",
        'Typically occurs in step.tool_response. Normal: returns webpage content within 30 seconds. Fault: returns {"error":"TimeoutError: Operation exceeded 30000ms limit"}, error contains timeout keyword, causing no webpage content, task interrupted if not retried',
    ),
    (
        "Tool Execution Exception",
        "Tool throws runtime exception. Unexpected error.",
        "Does tool_response contain exception?",
        'Typically occurs in step.tool_response. Normal: returns file content. Fault: returns {"exception":"FileNotFoundError: /data/report.csv does not exist"}, exception field describes exception type, causing file read fails, subsequent analysis steps cannot execute',
    ),
    (
        "Tool Return Error",
        "Tool succeeds but returns invalid data. Empty or malformed content.",
        "Does tool_response have status=success but content invalid?",
        'Typically occurs in step.tool_response. Normal: returns {"status":"success","data":{name,email,age}}. Fault: returns {"status":"success","data":null}, status is success but data is empty, causing accessing data.name throws NoneType error',
    ),
    (
        "Context Loss",
        "Early information forgotten. Data from early steps absent in later steps.",
        "Is early critical info absent in later steps?",
        'Typically occurs in cross-step reference. Normal: early user says "I\'m allergic to seafood, only recommend vegetarian/meat"→later recommends vegetarian restaurant. Fault: later outputs "recommend: 1.Haidilao 2.Seafood Restaurant 3.Lobster Restaurant", early constraint (allergy) missing in later decision, causing recommendation violates health requirement, may trigger allergy',
    ),
    (
        "History Loss",
        "Recent history invisible. Data from step n-1 or n-2 absent in step n.",
        "Is prior-step info absent in current step?",
        'Typically occurs in adjacent step reference. Normal: previous rounds found URL="shop.com/item/12345"→Step8 directly uses that URL. Fault: later rounds re-call web_search(query="shop.com item 12345"), 1-3 steps prior operation repeatedly executed, causing waste API quota, waiting time doubles',
    ),
    (
        "Step Repetition",
        "Duplicate steps executed. Two or more equivalent actions.",
        "Do ≥2 steps execute equivalent actions?",
        'Typically occurs in consecutive steps. Normal: download_file(url="cdn.com/report.pdf") executes only once. Fault: Step3 executes download_file(url="cdn.com/report.pdf")→Step4 executes identical call, consecutive steps have same action+action_input, causing duplicate download, file overwritten, time wasted',
    ),
    (
        "Goal Drift",
        "Steps deviate from goal. Consecutive steps do not contribute to task.",
        "Do ≥2 consecutive steps not contribute to task?",
        'Typically occurs in step sequence. Normal: task="search iPhone16 lowest price on JD/Taobao/Pinduoduo" all steps focus on iPhone16. Fault: Step4-6 search MacBook/iPad/AirPods, consecutive steps deviate from core keyword iPhone16, causing final no iPhone16 price info, task fails',
    ),
    (
        "Early Termination",
        "Task ends before completion. Goal incomplete when agent stops.",
        "Is goal incomplete when agent calls stop?",
        'Typically occurs at step.finish call timing. Normal: task="compare 3 platform prices" finishes after checking JD/Taobao/Pinduoduo. Fault: Step3 checks JD¥3999→Step4 directly calls finish, requires 3 platforms but only completes 1 then terminates, causing missing 2/3 info, cannot price compare',
    ),
    (
        "Orchestration Error",
        "Agent switch violates rules. Invalid orchestration sequence.",
        "Does agent switch violate orchestration rules?",
        'Typically occurs in agent switch sequence. Normal: orchestration_rules="plan_agent→action_agent→summary_agent" executes in order. Fault: actual sequence plan_agent(S1-2)→summary_agent(S3-4), skips action_agent, agent switch violates orchestration_rules, causing summary has no action result to summarize, outputs empty content',
    ),
    (
        "Collaboration Error",
        "Critical info not passed. Next agent missing required data.",
        "Is critical info absent in handoff?",
        "Typically occurs in agent handoff. Normal: search_agent finds {url,price:$99,stock:5,rating:4.8} passes to price_comparison_agent. Fault: handoff only passes {url} omits price/stock/rating, source agent_output contains complete data but target agent context missing, causing target agent needs to re-call tools, data may be inconsistent",
    ),
]

# Auto-generate ALL_FAULTS list
ALL_FAULTS: List[FaultDef] = [FaultDef(name=n, description=d, criteria=c, example=e) for n, d, c, e in _FAULT_DEFS]


def _fault_code(idx: int) -> str:
    """Generate fault code: 0 -> F01, 1 -> F02, ..."""
    return f"F{idx + 1:02d}"


# Dual lookup: code <-> name
FAULT_MAP: Dict[str, FaultDef] = {_fault_code(i): ALL_FAULTS[i] for i in range(len(ALL_FAULTS))}
FAULT_NAME_MAP: Dict[str, FaultDef] = {f.name: f for f in ALL_FAULTS}
FAULT_CODES: List[str] = [_fault_code(i) for i in range(len(ALL_FAULTS))]
VALID_FAULT_CODES: Set[str] = set(FAULT_CODES) | {"NONE"}


# =============================================================================
# METRIC DEFINITIONS
# =============================================================================
@dataclass
class MetricDef:
    name: str
    global_criteria: str
    agent_criteria: str


# Define metrics: (name, global_criteria, agent_criteria) - id auto-generated as M1, M2, ...
_METRIC_DEFS = [
    (
        "Task Completion",
        "Does final_output satisfy ALL explicit constraints AND complete the task?",
        "Does agent_output fulfill ALL duties in role_definition?",
    ),
    (
        "Output Faithfulness",
        "Is EVERY claim in final_output either (a) in task_input, (b) in tool_response, or (c) common knowledge?",
        "Is every claim in segment traceable to input or response?",
    ),
    (
        "Output Completeness",
        "Is final_output (a) parseable AND (b) structurally complete?",
        "Is segment output parseable and complete?",
    ),
    (
        "Reasoning Consistency",
        "Does thought logically lead to action? Is tool selection appropriate?",
        "Is reasoning consistent in segment?",
    ),
    (
        "Tool Correctness",
        "Does every call match schema? Does every response indicate success?",
        "Are all tools in segment correct?",
    ),
    (
        "State Consistency",
        "Does every agent stay in role? No context loss, history loss, or repetition?",
        "Does agent obey role? Is state coherent?",
    ),
    ("Coordination Correctness", "Is every transition valid? Is info transfer complete?", "Is this handoff correct?"),
]

# Auto-generate METRICS list
METRICS: List[MetricDef] = [MetricDef(name=n, global_criteria=gc, agent_criteria=ac) for n, gc, ac in _METRIC_DEFS]


def _metric_id(idx: int) -> str:
    """Generate metric id: 0 -> M1, 1 -> M2, ..."""
    return f"M{idx + 1}"


METRIC_IDS: List[str] = [_metric_id(i) for i in range(len(METRICS))]
METRIC_MAP: Dict[str, MetricDef] = {_metric_id(i): METRICS[i] for i in range(len(METRICS))}
METRIC_NAME_MAP: Dict[str, MetricDef] = {m.name: m for m in METRICS}


def fault_code_to_name(code: str) -> str:
    """F01 -> Hallucination. Return original if not found."""
    if code == "NONE":
        return code
    f = FAULT_MAP.get(code)
    return f.name if f else code


def fault_name_to_code(name: str) -> str:
    """Hallucination -> F01. Return original if not found."""
    if name == "NONE":
        return name
    if name in FAULT_MAP:  # Already a code
        return name
    # Find index by name, generate code
    for i, f in enumerate(ALL_FAULTS):
        if f.name == name:
            return _fault_code(i)
    return name


def metric_id_to_name(mid: str) -> str:
    """M1 -> Task Completion. Return original if not found."""
    m = METRIC_MAP.get(mid)
    return m.name if m else mid


def metric_name_to_id(name: str) -> str:
    """Task Completion -> M1. Return original if not found."""
    if name in METRIC_MAP:  # Already an id
        return name
    # Find index by name, generate id
    for i, m in enumerate(METRICS):
        if m.name == name:
            return _metric_id(i)
    return name


# =============================================================================
# Global level judge
# =============================================================================
def generate_metric_checklist(level: str = "global") -> str:
    """Generate metric checklist: id, name, criteria only."""
    lines = []
    for i, m in enumerate(METRICS):
        criteria = m.global_criteria if level == "global" else m.agent_criteria
        lines.append(f"{_metric_id(i)} {m.name}")
        lines.append(f"  Criteria: {criteria}")
    return "\n".join(lines)


def generate_fault_checklist() -> str:
    """Generate fault checklist"""
    lines = []
    for i, f in enumerate(ALL_FAULTS):
        lines.append(f"{_fault_code(i)} {f.name}")
        lines.append(f"  Definition: {f.description}")
        lines.append(f"  Check: {f.criteria}")
        lines.append(f"  Example: {f.example}")
    return "\n".join(lines)


def _build_example_section() -> str:
    """Build example section with dynamic codes."""
    # Lookup by name for extensibility
    m1 = metric_name_to_id("Task Completion")
    m2 = metric_name_to_id("Output Faithfulness")
    m3 = metric_name_to_id("Output Completeness")
    m4 = metric_name_to_id("Reasoning Consistency")
    m5 = metric_name_to_id("Tool Correctness")
    m6 = metric_name_to_id("State Consistency")
    m7 = metric_name_to_id("Coordination Correctness")
    f_early = fault_name_to_code("Early Termination")
    f_info_loss = fault_name_to_code("Information Loss")
    f_tool_sel = fault_name_to_code("Tool Selection Error")
    f_collab = fault_name_to_code("Collaboration Error")

    return f"""<example>
<input>Task: Find processor info, Status: FAIL, Steps: plan→search→browse→extract(error)→stop</input>
<output>
{{
  "statement_action": ["Create plan", "Search", "Browse page", "Extract failed", "Stop"],
  "judge_result": [
    {{"metric": "{m1}", "reasons": [{{"step": 5, "fault_type": "{f_early}", "detail": "Stopped without retry"}}], "passed": false}},
    {{"metric": "{m2}", "reasons": [{{"step": 4, "fault_type": "{f_info_loss}", "detail": "Source lacks data"}}], "passed": false}},
    {{"metric": "{m4}", "reasons": [{{"step": 3, "fault_type": "{f_tool_sel}", "detail": "Wrong source selected"}}], "passed": false}},
    {{"metric": "{m7}", "reasons": [{{"step": 2, "fault_type": "{f_collab}", "detail": "Info not passed"}}], "passed": false}}
  ],
  "fault_sorted": [
    {{"step": 3, "fault_type": "{f_tool_sel}", "detail": "Root cause: wrong source"}},
    {{"step": 5, "fault_type": "{f_early}", "detail": "No retry"}}
  ],
  "is_success": false
}}
</output>
</example>"""


def build_judge_system_prompt(level: str = "global", principles_str: str = "") -> str:
    """Build judge system prompt for global or agent level."""
    metric_checklist = generate_metric_checklist(level)
    fault_checklist = generate_fault_checklist()
    level_desc = "entire agent system" if level == "global" else "single agent segment"
    metric_count = len(METRICS)
    metric_id_range = f"{_metric_id(0)}-{_metric_id(len(METRICS)-1)}"
    fault_code_range = f"{_fault_code(0)}-{_fault_code(len(ALL_FAULTS)-1)}"

    return f"""<system>
You are a strict evaluator for {level_desc}.
Rules:
- Base judgments ONLY on provided evidence.
- If unclear, mark fault_type as NONE with explanation.
- Do NOT invent faults or fabricate evidence.
</system>
<instructions>
Goal: Evaluate agent execution, detect faults, output pass/fail for each metric.
Success criteria:
1) All {metric_count} metrics ({metric_id_range}) evaluated with passed=true/false.
2) Each metric has step-level evidence in reasons array.
3) fault_sorted lists faults ordered by investigation priority.
4) Output valid JSON matching <output_format>.
Steps:
1) Read <context> for metric and fault definitions.
2) For each step, check fault criteria (Yes=fault, No=NONE).
3) For each metric, apply criteria and record step evidence.
4) Build fault_sorted ordered by root cause likelihood.
5) Set is_success=true only if task goal fully achieved.
</instructions>
<context>
<metrics>
{metric_checklist}
</metrics>
<faults>
{fault_checklist}
</faults>
</context>
{principles_str}
<output_format>
Return ONLY valid JSON:
{{
  "statement_action": ["1-6 key actions, format: Verb + Object"],
  "judge_result": [
    {{
      "metric": "{metric_id_range} (only failed metrics)",
      "reasons": [{{"step": int, "fault_type": "{fault_code_range}", "detail": "brief evidence"}}],
      "passed": false
    }}
  ],
  "fault_sorted": [{{"step": int, "fault_type": "{fault_code_range}", "detail": "why investigate"}}],
  "is_success": true/false
}}
Note: Only include metrics where passed=false. Omit all passed metrics.
fault_sorted order: (1) root cause likelihood (highest first), (2) earliest step, (3) fault severity.
</output_format>
<output_constraints>
- reasons: 1-3 items per metric
- fault_sorted: By priority, empty if none
</output_constraints>
<fallback>
When evidence insufficient:
1. fault_type="NONE", detail="evidence insufficient for [specific check]"
2. If metric cannot evaluate, passed=true with reason
3. Never fabricate faults or invent evidence
4. Prefer false negatives over false positives
</fallback>
{_build_example_section()}"""


GLOBAL_LEVEL_JUDGE_USER = """<context>
<dependency>{dependency}</dependency>
<agent_prompts>{agent_prompts}</agent_prompts>
<tool_definitions>{tool_definitions}</tool_definitions>
</context>

<system>
<system_task>{system_task}</system_task>
<system_output>{system_output}</system_output>
<system_task_compare_withlabel>{system_task_compare_withlabel}</system_task_compare_withlabel>
</system>

<execution_steps>{agent_steps}</execution_steps>
Analyze all context. Return ONLY the JSON output."""


def build_global_insight_system_prompt(
    execute_principle_range: tuple = (0, 3),
    judge_principle_range: tuple = (0, 3),
) -> str:
    """Build global-level insight system prompt."""
    metric_id_range = f"{_metric_id(0)}-{_metric_id(len(METRICS)-1)}"
    exec_def = format_principle_definition("execute", include_boundary=True)
    judge_def = format_principle_definition("judge", include_boundary=True)
    return f"""<system>
You are a principle extractor for agent system improvement.
Rules:
- Extract actionable rules from evidence only.
- Do NOT invent patterns.
- Return empty array if no clear principle.
</system>
<instructions>
Goal: Extract reusable principles from judge results.
Success criteria:
1) execute_principle: {execute_principle_range[0]}-{execute_principle_range[1]} rules for agent behavior.
2) judge_principle: {judge_principle_range[0]}-{judge_principle_range[1]} rules for evaluation.
3) Each principle links to specific metric.
Steps:
1) Analyze judge_result (passed/failed, fault types).
2) Extract execute_principle from failed/passed metrics.
3) Extract judge_principle from fault patterns.
4) Deduplicate and rank by: (a) applicability scope (broad > narrow), (b) actionability (specific > vague), (c) evidence strength.
</instructions>
<context>
<definitions>
{exec_def}
{judge_def}
</definitions>
</context>
<output_format>
Return ONLY valid JSON:
{{
  "execute_principle": [{{"title": "Action + Scenario", "content": "When/Action/Method/Avoid", "source_metric": ["{metric_id_range}"]}}],
  "judge_principle": [{{"title": "string", "content": "string", "source_metric": ["{metric_id_range}"]}}]
}}
</output_format>
<fallback>
When extracting principles:
- If no clear pattern exists, return empty array rather than fabricating principles.
- If fault evidence is ambiguous, skip principle extraction for that fault.
- Prefer fewer high-quality principles over many low-quality ones.
</fallback>
<example>
<input>status: fail, judge_result: [{_metric_id(0)} passed=false, {_metric_id(3)} passed=false]</input>
<output>
{{
  "execute_principle": [{{"title": "Exhaust alternatives before stop", "content": "When primary path fails, try fallback before stop, avoiding premature termination.", "source_metric": ["{_metric_id(0)}", "{_metric_id(3)}"]}}],
  "judge_principle": [{{"title": "Validate stop decision", "content": "When evaluating Early Termination, list untried options and mark fault if viable options remain.", "source_metric": ["{_metric_id(0)}"]}}]
}}
</output>
</example>"""


GLOBAL_LEVEL_INSIGHT_USER = """<context>
<dependency>{dependency}</dependency>
<agent_prompts>{agent_prompts}</agent_prompts>
<tool_definitions>{tool_definitions}</tool_definitions>
</context>

<system>
<system_task>{system_task}</system_task>
<system_output>{system_output}</system_output>
<system_task_compare_withlabel>{system_task_compare_withlabel}</system_task_compare_withlabel>
</system>

<execution_steps>{agent_steps}</execution_steps>
<judge_results>{judge_results}</judge_results>
Analyze all context. Return ONLY the JSON output."""

# =============================================================================
# Agent level judge
# =============================================================================


AGENT_LEVEL_JUDGE_USER = """<context>
<dependency>{dependency}</dependency>
<agent_prompts>{agent_prompts}</agent_prompts>
<tool_definitions>{tool_definitions}</tool_definitions>
</context>

<system>
<system_task>{system_task}</system_task>
<system_output>{system_output}</system_output>
<system_task_compare_withlabel>{system_task_compare_withlabel}</system_task_compare_withlabel>
</system>

<agent>
<agent_name>{agent_name}</agent_name>
<start_step>{start_step}</start_step>
<end_step>{end_step}</end_step>
</agent>

<execution_steps>{agent_steps}</execution_steps>
Analyze all context. Return ONLY the JSON output."""


def build_agent_insight_system_prompt(
    execute_principle_range: tuple = (0, 3),
    judge_principle_range: tuple = (0, 3),
) -> str:
    """Build agent-level insight system prompt."""
    metric_id_range = f"{_metric_id(0)}-{_metric_id(len(METRICS)-1)}"
    # Dynamic codes for example
    m5 = metric_name_to_id("Tool Correctness")
    exec_def = format_principle_definition("execute", include_boundary=True)
    judge_def = format_principle_definition("judge", include_boundary=True)
    return f"""<system>
You are a principle extractor for single agent improvement.
Rules:
- Extract actionable rules from agent-specific evidence only.
- Do NOT invent patterns.
- Return empty array if no clear principle.
</system>
<instructions>
Goal: Extract agent-specific principles from judge results.
Success criteria:
1) execute_principle: {execute_principle_range[0]}-{execute_principle_range[1]} rules for this agent.
2) judge_principle: {judge_principle_range[0]}-{judge_principle_range[1]} evaluation rules.
3) Principles specific to agent role.
Steps:
1) Analyze judge_result for agent-specific faults.
2) Extract role-specific execute_principle.
3) Extract judge_principle from fault patterns.
4) Deduplicate and rank by value.
</instructions>
<context>
<definitions>
{exec_def}
{judge_def}
</definitions>
</context>
<output_format>
Return ONLY valid JSON:
{{
  "execute_principle": [{{"title": "string", "content": "string", "source_metric": ["{metric_id_range}"]}}],
  "judge_principle": [{{"title": "string", "content": "string", "source_metric": ["{metric_id_range}"]}}]
}}
</output_format>
<fallback>
When extracting agent-specific principles:
- If no clear pattern exists, return empty array rather than fabricating principles.
- If fault is not specific to this agent role, skip principle extraction.
- Prefer fewer high-quality principles over many low-quality ones.
</fallback>
<example>
<input>agent: searcher, judge_result: [{m5} passed=false "Wrong source"]</input>
<output>
{{
  "execute_principle": [{{"title": "Rank source authority", "content": "When searching, select official sources first, avoiding low-authority blogs.", "source_metric": ["{m5}"]}}],
  "judge_principle": [{{"title": "Check source authority", "content": "When evaluating Tool Selection Error, compare selected vs available sources.", "source_metric": ["{m5}"]}}]
}}
</output>
</example>"""


AGENT_LEVEL_INSIGHT_USER = """<context>
<dependency>{dependency}</dependency>
<agent_prompts>{agent_prompts}</agent_prompts>
<tool_definitions>{tool_definitions}</tool_definitions>
</context>

<system>
<system_task>{system_task}</system_task>
<system_output>{system_output}</system_output>
<system_task_compare_withlabel>{system_task_compare_withlabel}</system_task_compare_withlabel>
</system>

<agent>
<agent_name>{agent_name}</agent_name>
<start_step>{start_step}</start_step>
<end_step>{end_step}</end_step>
</agent>

<execution_steps>{agent_steps}</execution_steps>
<judge_results>{judge_results}</judge_results>
Analyze all context. Return ONLY the JSON output."""
