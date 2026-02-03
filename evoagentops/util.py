# util.py
from typing import Dict, List, Any, Optional, Tuple, Callable
import json
from abc import ABC, abstractmethod
from collections import defaultdict
import re
from loguru import logger
from pathlib import Path
import os
from datetime import datetime
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from openai import OpenAI, AsyncOpenAI
import asyncio
from asyncio import Semaphore
from .config import Config
import logging
from contextvars import ContextVar
import random

# Context for tracking async call source
call_context: ContextVar[dict] = ContextVar("call_context", default={})


def set_call_context(**kwargs):
    """Set context: dataset, case_id, stage (judge/extract/merge), idx"""
    ctx = call_context.get().copy()
    ctx.update({k: v for k, v in kwargs.items() if v is not None})
    call_context.set(ctx)


def clear_call_context():
    """Clear context"""
    call_context.set({})


def init_logger(log_file: str, level: str = "INFO"):
    """Configure logger with colorized output"""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger.remove()
    # Suppress veadk INFO/WARNING logs at Loguru sink level, keep ERROR+
    logging.getLogger("veadk").setLevel(logging.ERROR)  # only affects std logging
    veadk_filter = lambda record: ((not record["name"].startswith("veadk")) or record["level"].no >= logging.ERROR)
    # Output to console (with color)
    logger.add(
        sink=lambda msg: print(msg, end=""),
        level=level,
        colorize=True,
        filter=veadk_filter,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    )
    # Output to log file (without color)
    logger.add(
        Path(log_file),
        level=level,
        rotation="1 day",
        encoding="utf-8",
        filter=veadk_filter,
        format="{time:YYYY-MM-DD HH:mm:ss} [{level}] ({name}:{function}:{line}) {message}",
    )
    return logger


async def gather_with_concurrency(max_concurrency: int, tasks: List[Callable]) -> List[Any]:
    """Concurrency control for asyncio.gather - limit the number of tasks running simultaneously"""
    semaphore = Semaphore(max_concurrency)

    async def run_with_semaphore(task):
        async with semaphore:
            return await task

    return await asyncio.gather(*[run_with_semaphore(t) for t in tasks])


def log_api(log_file: str, data: dict):
    """Record API call logs"""
    os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else ".", exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as f:
        record = {"timestamp": datetime.now().isoformat(), **call_context.get(), **data}
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def call_embedding(
    text: str,
    config: Config,
) -> List[float]:
    """Unified embedding call interface"""
    log_file = os.path.join(config.output_dir, "call_embedding.jsonl")
    client = OpenAI(api_key=config.embedding_api_key, base_url=config.embedding_base_url)
    try:
        response = client.embeddings.create(model=config.embedding_model, input=[text], encoding_format="float")
        embedding = response.data[0].embedding
        # Record API log
        log_api(
            log_file,
            {
                "input": text,
                "embedding": embedding,
            },
        )
        return embedding
    except Exception as e:
        logger.error(f"Embedding call failed: {e}")
        raise e


async def acall_embedding(
    text: str,
    config: Config,
) -> List[float]:
    """Unified Embedding Calling Interface"""
    if not hasattr(config, "_embedding_semaphore"):
        config._embedding_semaphore = asyncio.Semaphore(getattr(config, "embedding_max_concurrency", 5))
    async with config._embedding_semaphore:
        log_file = os.path.join(config.output_dir, "call_embedding.jsonl")
        client = AsyncOpenAI(api_key=config.embedding_api_key, base_url=config.embedding_base_url)
        try:
            response = await client.embeddings.create(
                model=config.embedding_model, input=[text], encoding_format="float"
            )
            embedding = response.data[0].embedding
            # Record API log
            log_api(
                log_file,
                {
                    "input": text,
                    "embedding": embedding[:50],
                },
            )
            return embedding
        except Exception as e:
            logger.error(f"Embedding call failed: {e}")
            raise e


async def acall_embedding_batch(
    texts: List[str],
    config: Config,
    batch_size: int = 100,
) -> List[List[float]]:
    """Batch embedding call - single API call for multiple texts"""
    if not texts:
        return []
    if not hasattr(config, "_embedding_semaphore"):
        config._embedding_semaphore = asyncio.Semaphore(getattr(config, "embedding_max_concurrency", 5))

    all_embeddings = []
    log_file = os.path.join(config.output_dir, "call_embedding.jsonl")
    client = AsyncOpenAI(api_key=config.embedding_api_key, base_url=config.embedding_base_url)

    # Process in batches to avoid API limits
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        async with config._embedding_semaphore:
            try:
                response = await client.embeddings.create(
                    model=config.embedding_model, input=batch_texts, encoding_format="float"
                )
                embeddings = [d.embedding for d in response.data]
                all_embeddings.extend(embeddings)

                # Log
                dump = response.model_dump()
                for item in dump.get("data", []):
                    emb = item.get("embedding")
                    if isinstance(emb, list):
                        item["embedding_full_len"] = len(emb)
                        item["embedding"] = emb[:100]
                log_api(log_file, {"input": batch_texts, "output": dump})

            except Exception as e:
                err_str = str(e).lower()
                # Check for retryable errors
                retryable_keywords = [
                    "timeout",
                    "rate limit",
                    "rate_limit",
                    "ratelimit",
                    "429",
                    "503",
                    "overloaded",
                    "connection reset",
                    "connection refused",
                    "too many requests",
                ]
                if any(kw in err_str for kw in retryable_keywords):
                    logger.warning(f"Batch embedding retryable at batch {i//batch_size+1}: {e}")
                    raise RetryableError(str(e)) from e
                logger.error(f"Batch embedding failed at batch {i//batch_size+1}: {e}")
                raise e

    logger.info(f"Batch embedding done: {len(texts)} texts -> {len(all_embeddings)} embeddings")
    return all_embeddings


# Retryable error types (timeout, rate limit, etc.)
class RetryableError(Exception):
    pass


async def acall_llm(
    messages: List[Dict],
    config,
    output_schema: Optional[type[BaseModel]] = None,
) -> str:
    """Unified LLM Calling Interface"""
    if not hasattr(config, "_llm_semaphore"):
        config._llm_semaphore = asyncio.Semaphore(getattr(config, "llm_max_concurrency", 3))
    async with config._llm_semaphore:
        log_file = os.path.join(config.output_dir, "call_llm.jsonl")
        # Create LangChain ChatOpenAI client
        # Build extra params based on model type
        model_lower = config.openai_model.lower()
        extra_params = {}
        if "gpt" in model_lower:
            # none, low, medium, high. seed is ok too (minimal)
            extra_params["model_kwargs"] = {"reasoning_effort": "none"}
            pass
        elif "qwen" in model_lower or "seed" in model_lower or "glm" in model_lower:
            extra_params["temperature"] = 0.7
            # extra_params["model_kwargs"] = {"reasoning_effort": "medium"}
            extra_params["extra_body"] = {"thinking": {"type": "disabled"}}
        elif "claude" in model_lower:
            extra_params["temperature"] = 0.7

        llm = ChatOpenAI(
            model=config.openai_model,
            base_url=config.openai_base_url,
            api_key=config.openai_api_key,
            timeout=600,
            max_retries=0,
            **extra_params,
        )
        max_retries = 10
        for attempt in range(max_retries):
            try:
                if output_schema:
                    # Structured output with JSON schema
                    structured_llm = llm.with_structured_output(
                        output_schema,
                        method="json_schema",  # json_schema first, function_calling cannot thinking
                        include_raw=True,
                    )
                    raw_response = await structured_llm.ainvoke(messages)
                    response = raw_response["raw"]
                    parsed = raw_response.get("parsed")

                    # Log raw response for debugging
                    log_api(
                        log_file,
                        {
                            "input": messages,
                            "output": response.model_dump() if hasattr(response, "model_dump") else str(response),
                        },
                    )

                    # # Return parsed result as JSON string (function_calling: content is empty, data in parsed)
                    # Return clean JSON string from Pydantic model
                    return parsed.model_dump_json()
                else:
                    # Plain text output
                    response = await llm.ainvoke(messages)
                    log_api(
                        log_file,
                        {
                            "input": messages,
                            "output": response.model_dump() if hasattr(response, "model_dump") else str(response),
                        },
                    )
                    return response.content
            except Exception as e:
                err_str = str(e).lower()
                retryable_keywords = [
                    "timeout",
                    "rate limit",
                    "rate_limit",
                    "ratelimit",
                    "429",
                    "503",
                    "overloaded",
                    "connection reset",
                    "connection refused",
                ]

                if any(kw in err_str for kw in retryable_keywords):
                    if attempt < max_retries - 1:
                        wait_time = 5 * (2**attempt) + random.uniform(0, 1)
                        match = re.search(r"retry after (\d+)", err_str)
                        if match:
                            wait_time = int(match.group(1)) + 1

                        logger.warning(
                            f"LLM retryable error (attempt {attempt+1}/{max_retries}), wait {wait_time:.1f}s: {e}"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"LLM failed after {max_retries} retries: {e}")
                        raise RetryableError(str(e)) from e

                logger.error(f"LLM call failed (permanent): {e}")
                raise e


class BaseTraceParser(ABC):
    """Base Trace Parser"""

    def __init__(self, trace: List[Dict]):
        self.trace = self._deduplicate_trace(trace)
        self.span_map = {t["span_id"]: t for t in self.trace}

    def _deduplicate_trace(self, trace: List[Dict]) -> List[Dict]:
        seen_ids = set()
        deduplicated_trace = []
        for span in trace:
            span_id = span.get("span_id")
            if span_id and span_id not in seen_ids:
                seen_ids.add(span_id)
                deduplicated_trace.append(span)
        return deduplicated_trace

    def parse_json_value(self, span: Dict, field: str) -> Any:
        """Safely parse JSON field"""
        try:
            value = span.get("attributes", {}).get(field, "{}")
            return json.loads(value) if isinstance(value, str) else value
        except:
            return {}

    @abstractmethod
    def extract_agent_steps(self) -> List[Dict]:
        pass

    @abstractmethod
    def extract_agent_settings(self) -> Dict:
        pass

    @abstractmethod
    def extract_agent_dependency(self) -> Dict:
        pass

    def parse_all(self) -> Tuple:
        agent_steps = self.extract_agent_steps()
        agent_settings = self.extract_agent_settings()
        agent_dependency = self.extract_agent_dependency()
        return agent_steps, agent_settings, agent_dependency


class VeADKPhoenixParser(BaseTraceParser):
    def _get_agent_name(self, call_llm_span) -> str:
        """Extract agent name from call_llm span attributes"""
        if not call_llm_span:
            return "unknown"
        attrs = call_llm_span.get("attributes", {})
        # Try gcp.vertex.agent.llm_request -> config.labels.adk_agent_name
        for key in ["gcp.vertex.agent.llm_request", "llm.invocation_parameters"]:
            raw = attrs.get(key)
            if raw:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    # llm_request has config.labels, invocation_parameters has labels directly
                    labels = data.get("config", {}).get("labels") or data.get("labels")
                    if labels and labels.get("adk_agent_name"):
                        return labels["adk_agent_name"]
                except:
                    pass
        return attrs.get("gen_ai.agent.name", "unknown")

    def extract_agent_steps(self) -> List[Dict]:
        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        for cc_span in sorted(self.trace, key=lambda x: x["start_time_unix_nano"]):
            # Start traversing from ChatCompletion
            if cc_span.get("name") != "ChatCompletion":
                continue

            # find parent span call_llm
            call_llm_span = next((s for s in self.trace if s["span_id"] == cc_span.get("parent_span_id")), None)
            # Skip tool-internal LLM calls (e.g., ask_llm tool)
            if call_llm_span and "execute_tool" in call_llm_span.get("name", ""):
                continue
            # extract agent name from call_llm span
            agent_name = self._get_agent_name(call_llm_span)

            step_num += 1
            # directly parse data from current span(ChatCompletion)
            input_data = self.parse_json_value(cc_span, "input.value")
            output_data = self.parse_json_value(cc_span, "output.value")

            # extract agent information from input_data
            input_messages = input_data.get("messages", [])
            output_message = output_data.get("choices", [])[0].get("message", {})

            # veadk special context removal: remove system messages and keep last "For context" message
            non_system_messages = [msg for msg in input_messages if msg.get("role") != "system"]
            # input_final = (
            #     non_system_messages
            #     if len(non_system_messages) <= 1
            #     else [non_system_messages[0], non_system_messages[-1]]
            # )

            # veadk special context removal: remove system messages and keep last "For context" message
            def is_context_message(msg):
                # check if message is a For context
                content = msg.get("content", "")
                if isinstance(content, str):
                    return content.startswith("For context")
                if isinstance(content, list) and len(content) > 0:
                    first_text = content[0].get("text", "") if isinstance(content[0], dict) else ""
                    return first_text.startswith("For context")
                return False

            # if msg.get("role") not in ["system", "tool", "assistant"] and not is_context_message(msg)
            input_final = [msg for msg in input_messages if msg.get("role") == "user" and not is_context_message(msg)]

            # output_message may have tools, we need to distinguish them
            output_final = {"role": output_message.get("role", ""), "content": output_message.get("content", "")}

            # extract tool call
            tools_called = []
            tool_calls = output_message.get("tool_calls", [])
            for tool_call in tool_calls:
                tool_response = None
                for s in self.trace:
                    # find execute_tool within call_llm
                    if s.get("parent_span_id") == call_llm_span["span_id"] and "execute_tool" in s.get("name", ""):
                        out = self.parse_json_value(s, "gen_ai.tool.output")
                        if out.get("id") == tool_call.get("id"):
                            tool_response = out.get("response", {})
                            break
                tools_called.append(
                    {
                        "tool_name": tool_call.get("function", {}).get("name", ""),
                        "tool_args": tool_call.get("function", {}).get("arguments", {}),
                        "tool_response": tool_response,
                    }
                )

            # calculate usage
            cc_attrs = cc_span.get("attributes", {})
            input_tokens = int(cc_attrs.get("llm.token_count.prompt", 0))
            output_tokens = int(cc_attrs.get("llm.token_count.completion", 0))
            # calculate time from current span
            exec_time = (int(cc_span["end_time_unix_nano"]) - int(cc_span["start_time_unix_nano"])) / 1e9

            accumulated["input_tokens"] += input_tokens
            accumulated["output_tokens"] += output_tokens
            accumulated["time"] += exec_time
            if last_agent_name != agent_name:
                last_agent_name = agent_name
                accumulated["transfers"] += 1

            agent_steps.append(
                {
                    "step": step_num,
                    "agent_name": agent_name,
                    "agent": {"input": input_final, "output": output_final, "tools_called": tools_called},
                    "environment": None,
                    "step_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_inference_time": exec_time,
                        "model": cc_attrs.get("llm.model_name"),
                        "step_execution_time": exec_time,
                    },
                    "accumulated_usage": {
                        "accumulated_input_tokens": accumulated["input_tokens"],
                        "accumulated_output_tokens": accumulated["output_tokens"],
                        "accumulated_time": accumulated["time"],
                        "accumulated_transferred_times": accumulated["transfers"],
                    },
                    # use current span's time and ID info
                    "start_time": cc_span["start_time_unix_nano"],
                    "end_time": cc_span["end_time_unix_nano"],
                    "span_id": cc_span["span_id"],
                    "trace_id": cc_span["trace_id"],
                    "parent_span_id": cc_span.get("parent_span_id", ""),
                }
            )
        return agent_steps

    def extract_agent_settings(self) -> Dict:
        """extract agent settings and tool definitions"""
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()

        for span in self.trace:
            # only process ChatCompletion span
            if span.get("name") != "ChatCompletion":
                continue

            attrs = span.get("attributes", {})

            # get agent name from parent span
            parent_span = self.span_map.get(span.get("parent_span_id"))
            agent_name = self._get_agent_name(parent_span)
            if agent_name == "unknown":
                agent_name = None

            # parse system prompt from llm.input_messages.x.message.role
            idx = 0
            while True:
                role = attrs.get(f"llm.input_messages.{idx}.message.role")
                content = attrs.get(f"llm.input_messages.{idx}.message.content")
                if role is None:
                    break
                if role == "system" and agent_name and content:
                    # Extract only the main part of the prompt and remove the content after "You are an agent". Remove line breaks and spaces at the beginning and end.
                    agent_settings["prompt"][agent_name] = content.strip()
                    break
                idx += 1

            # extract tool definitions from llm.tools.x.tool.json_schema
            idx = 0
            while True:
                tool_schema_key = f"llm.tools.{idx}.tool.json_schema"
                tool_schema = attrs.get(tool_schema_key)
                if tool_schema is None:
                    break

                tool_data = json.loads(tool_schema) if isinstance(tool_schema, str) else tool_schema
                # Handle two formats: {"type": "function", "function": {...}} or {"name": "...", ...}
                if tool_data.get("type") == "function":
                    func = tool_data.get("function", {})
                else:
                    func = tool_data  # Direct format
                tool_name = func.get("name", "")
                if tool_name and tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    agent_settings["tool"].append(
                        {
                            "name": tool_name,
                            "description": func.get("description", "").strip(),
                            "parameters": func.get("parameters", {}),
                        }
                    )
                idx += 1
        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        """extract agent dependency based on parent-child span relationships"""
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})
        # Build parent->children map
        children_map = defaultdict(list)
        for span in self.trace:
            parent_id = span.get("parent_span_id")
            if parent_id:
                children_map[parent_id].append(span)

        # Traverse all call_llm spans
        for call_llm_span in self.trace:
            if call_llm_span.get("name") != "call_llm":
                continue
            agent_name = self._get_agent_name(call_llm_span)
            if agent_name == "unknown":
                continue
            _ = agent_dependency[agent_name]
            call_llm_span_id = call_llm_span.get("span_id")

            # Find child spans of call_llm
            for child_span in children_map.get(call_llm_span_id, []):
                child_name = child_span.get("name", "")

                # Tool dependency: execute_tool span
                if "execute_tool" in child_name:
                    attrs = child_span.get("attributes", {})
                    tool_name = attrs.get("gen_ai.tool.name") or attrs.get("tool.name")
                    if tool_name:
                        agent_dependency[agent_name]["tool"].append(tool_name)

                # Sub-agent dependency: nested call_llm span
                elif child_name == "call_llm":
                    sub_agent = self._get_agent_name(child_span)
                    if sub_agent != "unknown" and sub_agent != agent_name:
                        agent_dependency[agent_name]["agent"].append(sub_agent)

            # Also check grandchildren for sub-agents (call_llm may be nested deeper)
            for child_span in children_map.get(call_llm_span_id, []):
                for grandchild in children_map.get(child_span.get("span_id"), []):
                    if grandchild.get("name") == "call_llm":
                        sub_agent = self._get_agent_name(grandchild)
                        if sub_agent != "unknown" and sub_agent != agent_name:
                            agent_dependency[agent_name]["agent"].append(sub_agent)

        # remove duplicates and sort
        return {
            k: {"agent": sorted(list(set(v["agent"]))), "tool": sorted(list(set(v["tool"])))}
            for k, v in agent_dependency.items()
        }


class ADKPhoenixParser(BaseTraceParser):
    def extract_agent_steps(self) -> List[Dict]:
        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        for span in sorted(self.trace, key=lambda x: x["start_time_unix_nano"]):
            if not span.get("name", "").startswith("call_llm"):
                continue

            step_num += 1
            attrs = span.get("attributes", {})

            # Get agent name from parent agent_run span
            parent_id = span.get("parent_span_id")
            agent_name = "unknown"
            for s in self.trace:
                if s.get("span_id") == parent_id and "agent_run" in s.get("name", ""):
                    match = re.search(r"agent_run \[(.+?)\]", s.get("name", ""))
                    if match:
                        agent_name = match.group(1)
                    break

            # Parse input from llm.input_messages, only keep user role messages
            user_input = []
            idx = 0
            while True:
                role = attrs.get(f"llm.input_messages.{idx}.message.role")
                if role is None:
                    break
                if role == "user":
                    # Try direct content first, then nested contents
                    content = attrs.get(f"llm.input_messages.{idx}.message.content")
                    if not content:
                        content = attrs.get(f"llm.input_messages.{idx}.message.contents.0.message_content.text", "")
                    if content and not user_input:  # Only keep the first user message
                        user_input.append({"role": "user", "content": content})
                idx += 1

            # Parse output text from output.value
            output_data = self.parse_json_value(span, "output.value")
            output_text = ""
            if isinstance(output_data, dict):
                parts = output_data.get("content", {}).get("parts", [])
                for part in parts:
                    if "text" in part:
                        output_text = part["text"]
                        break

            # Extract tool calls from output_messages
            tools_called = []
            idx = 0
            while True:
                tool_name = attrs.get(f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.name")
                tool_args_str = attrs.get(
                    f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.function.arguments"
                )
                tool_id = attrs.get(f"llm.output_messages.0.message.tool_calls.{idx}.tool_call.id")
                if tool_name is None:
                    break
                # Find tool response from execute_tool span
                tool_response = None
                for tool_span in self.trace:
                    if "execute_tool" in tool_span.get("name", ""):
                        if tool_span.get("attributes", {}).get("gen_ai.tool.call.id") == tool_id:
                            tool_response = self.parse_json_value(tool_span, "output.value")
                            break
                tools_called.append(
                    {
                        "tool_name": tool_name,
                        "tool_args": json.loads(tool_args_str) if tool_args_str else {},
                        "tool_response": tool_response,
                    }
                )
                idx += 1

            # Parse token statistics
            input_tokens = int(attrs.get("llm.token_count.prompt") or attrs.get("gen_ai.usage.input_tokens") or 0)
            output_tokens = int(attrs.get("llm.token_count.completion") or attrs.get("gen_ai.usage.output_tokens") or 0)
            exec_time = (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"])) / 1e9

            accumulated["input_tokens"] += input_tokens
            accumulated["output_tokens"] += output_tokens
            accumulated["time"] += exec_time

            if last_agent_name != agent_name:
                last_agent_name = agent_name
                accumulated["transfers"] += 1

            agent_steps.append(
                {
                    "step": step_num,
                    "agent_name": agent_name,
                    "agent": {"input": user_input, "output": output_text, "tools_called": tools_called},
                    "environment": None,
                    "step_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_inference_time": exec_time,
                        "model": attrs.get("llm.model_name") or attrs.get("gen_ai.request.model"),
                        "step_execution_time": exec_time,
                    },
                    "accumulated_usage": {
                        "accumulated_input_tokens": accumulated["input_tokens"],
                        "accumulated_output_tokens": accumulated["output_tokens"],
                        "accumulated_time": accumulated["time"],
                        "accumulated_transferred_times": accumulated["transfers"],
                    },
                    "start_time": span["start_time_unix_nano"],
                    "end_time": span["end_time_unix_nano"],
                    "span_id": span["span_id"],
                    "trace_id": span["trace_id"],
                    "parent_span_id": span.get("parent_span_id", ""),
                }
            )

        return agent_steps

    def extract_agent_settings(self) -> Dict:
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()

        for span in self.trace:
            if not span.get("name", "").startswith("call_llm"):
                continue

            attrs = span.get("attributes", {})

            # Get agent name from parent agent_run span
            parent_id = span.get("parent_span_id")
            agent_name = None
            for s in self.trace:
                if s.get("span_id") == parent_id and "agent_run" in s.get("name", ""):
                    match = re.search(r"agent_run \[(.+?)\]", s.get("name", ""))
                    if match:
                        agent_name = match.group(1)
                    break

            # Extract system prompt from llm.input_messages.0 (system role)
            if attrs.get("llm.input_messages.0.message.role") == "system":
                system_prompt = attrs.get("llm.input_messages.0.message.content", "")
                if agent_name and system_prompt and agent_name not in agent_settings["prompt"]:
                    agent_settings["prompt"][agent_name] = system_prompt

            # Extract tool definitions from llm.tools.*.tool.json_schema
            idx = 0
            while True:
                tool_schema_str = attrs.get(f"llm.tools.{idx}.tool.json_schema")
                if tool_schema_str is None:
                    break
                try:
                    tool_schema = json.loads(tool_schema_str) if isinstance(tool_schema_str, str) else tool_schema_str
                    tool_name = tool_schema.get("name", "")
                    if tool_name and tool_name not in seen_tools:
                        seen_tools.add(tool_name)
                        agent_settings["tool"].append(
                            {
                                "name": tool_name,
                                "description": tool_schema.get("description", ""),
                                "parameters": tool_schema.get("parameters", {}),
                            }
                        )
                except:
                    pass
                idx += 1

        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})

        # Build set of agents that have direct LLM calls
        agents_with_llm = set()
        for span in self.trace:
            if "call_llm" in span.get("name", ""):
                parent_id = span.get("parent_span_id")
                for s in self.trace:
                    if s.get("span_id") == parent_id and "agent_run" in s.get("name", ""):
                        match = re.search(r"agent_run \[(.+?)\]", s.get("name", ""))
                        if match:
                            agents_with_llm.add(match.group(1))
                        break

        # Iterate through all spans of the AGENT type
        for agent_span in self.trace:
            if "agent_run" not in agent_span.get("name", ""):
                continue

            # Extract the agent name
            agent_match = re.search(r"agent_run \[(.+?)\]", agent_span.get("name", ""))
            if not agent_match:
                continue
            agent_name = agent_match.group(1)

            # Skip agents without direct LLM calls (wrapper agents)
            if agent_name not in agents_with_llm:
                continue

            _ = agent_dependency[agent_name]

            # Find child spans
            agent_span_id = agent_span.get("span_id")
            for child_span in self.trace:
                if child_span.get("parent_span_id") != agent_span_id:
                    continue
                child_kind = child_span.get("name", "")
                # child span is AGENT - agent dependency
                if "agent_run" in child_kind:
                    child_match = re.search(r"agent_run \[(.+?)\]", child_span.get("name", ""))
                    if child_match:
                        agent_dependency[agent_name]["agent"].append(child_match.group(1))
                # Sub-span is LLM - Find its tool calls
                elif "call_llm" in child_kind:
                    llm_span_id = child_span.get("span_id")
                    for grandchild_span in self.trace:
                        if grandchild_span.get("parent_span_id") == llm_span_id:
                            grandchild_kind = grandchild_span.get("name", "")

                            # Find AGENT under LLM - agent dependencies
                            if "agent_run" in grandchild_kind:
                                child_match = re.search(r"agent_run \[(.+?)\]", grandchild_span.get("name", ""))
                                if child_match:
                                    agent_dependency[agent_name]["agent"].append(child_match.group(1))

                            # Finding TOOL under LLM - tool dependencies
                            elif "execute_tool" in grandchild_kind:
                                attrs = grandchild_span.get("attributes", {})
                                tool_name = attrs.get("tool.name") or attrs.get("gen_ai.tool.name")
                                # Filter out placeholder tool names like "(merged tools)"
                                if tool_name and tool_name != "(merged tools)":
                                    agent_dependency[agent_name]["tool"].append(tool_name)

        # sort and remove duplicates
        return {
            k: {"agent": sorted(list(set(v["agent"]))), "tool": sorted(list(set(v["tool"])))}
            for k, v in agent_dependency.items()
        }


class LanggraphPhoenixParser(BaseTraceParser):
    """
    for resource_span in trace_data["data"]["resource_spans"]:
    for scope_span in resource_span["scope_spans"]:
        if scope_span["scope"]["name"] != "openinference.instrumentation.langchain": # only need scope name is langchain'span
            continue
        all_spans.extend(scope_span["spans"])
    """

    def extract_agent_steps(self) -> List[Dict]:
        """extract agent_steps"""

        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        # filter out LLM and TOOL type spans, and sort by start_time_unix_nano
        llm_and_tool_spans = [
            span
            for span in self.trace
            if span.get("attributes", {}).get("openinference.span.kind") in ["LLM", "TOOL"]
            and "langgraph_node" in self.parse_json_value(span, "metadata")
        ]
        sorted_spans = sorted(llm_and_tool_spans, key=lambda x: x["start_time_unix_nano"])

        # create span map for quick lookup
        # span_map = {span["span_id"]: span for span in sorted_spans}

        i = 0
        while i < len(sorted_spans):
            span = sorted_spans[i]
            span_kind = span.get("attributes", {}).get("openinference.span.kind")

            if span_kind == "LLM":
                step_num += 1
                attrs = span.get("attributes", {})

                # Parse basic information from LLM spans and get agent_name from metadata
                metadata = self.parse_json_value(span, "metadata")
                agent_name = metadata.get("langgraph_node", "unknown")

                # parse user_input from input.value
                input_data = self.parse_json_value(span, "input.value")
                user_input = []

                # Attempt to parse input messages from different fields
                if "messages" in input_data:
                    messages = input_data["messages"]
                    if isinstance(messages, list) and len(messages) > 0:
                        # Handle nested message structure
                        for msg_group in messages:
                            if isinstance(msg_group, list):
                                for msg in msg_group:
                                    if isinstance(msg, dict):
                                        role = msg.get("kwargs", {}).get("type")
                                        if role in ["system", "ai"]:  # skip system and ai messages
                                            continue
                                        content = msg.get("kwargs", {}).get("content", "")
                                        if content:
                                            user_input.append(
                                                {"role": "user" if role == "human" else role, "content": content}
                                            )
                            elif isinstance(msg_group, dict):
                                role = msg_group.get("kwargs", {}).get("type")
                                if role in ["system", "ai"]:  # skip system and ai messages
                                    continue
                                content = msg_group.get("kwargs", {}).get("content", "")
                                if content:
                                    user_input.append({"role": "user" if role == "human" else role, "content": content})

                # If it is not obtained from messages, try to obtain it from llm.input_messages
                if not user_input:
                    for attr_key, attr_value in attrs.items():
                        if attr_key.startswith("llm.input_messages.") and attr_key.endswith(".message.content"):
                            role_key = attr_key.replace(".content", ".role")
                            role = attrs.get(role_key)
                            if role == "system":
                                continue
                            if attr_value:
                                user_input.append({"role": role, "content": attr_value})

                # parse output_text from output.value
                output_data = self.parse_json_value(span, "output.value")
                output_text = ""

                # Attempt to parse output_text from different fields
                if "generations" in output_data:
                    generations = output_data.get("generations", [])
                    if generations and len(generations) > 0 and len(generations[0]) > 0:
                        output_text = generations[0][0].get("text", "")
                elif attrs.get("llm.output_messages.0.message.content"):
                    output_text = attrs.get("llm.output_messages.0.message.content", "")

                # parse tools_called from input.data
                tools_called = []

                # first check if tool_calls is in input.data
                tool_calls_found = False

                # Extract tool_calls from input.data
                if "tool_calls" in input_data:
                    tool_calls = input_data.get("tool_calls", [])
                    if tool_calls:
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("function", {}).get("name", "")
                                tool_args_str = tool_call.get("function", {}).get("arguments", "{}")

                                try:
                                    tool_args = json.loads(tool_args_str) if tool_args_str else {}
                                except:
                                    tool_args = {}

                                if tool_name:
                                    tool_calls_found = True
                                    # Find the TOOL span to get tool_response
                                    tool_response = None
                                    for j in range(i + 1, len(sorted_spans)):
                                        next_span = sorted_spans[j]
                                        if (
                                            next_span.get("attributes", {}).get("openinference.span.kind") == "TOOL"
                                            and next_span.get("attributes", {}).get("tool.name") == tool_name
                                        ):
                                            tool_response = self.parse_json_value(next_span, "output.value")
                                            break

                                    tools_called.append(
                                        {
                                            "tool_name": tool_name,
                                            "tool_args": tool_args,
                                            "tool_response": tool_response,
                                        }
                                    )

                # If LLM span has no tool_call info but subsequent TOOL span exists, parse from TOOL span
                if not tool_calls_found and i + 1 < len(sorted_spans):
                    next_span = sorted_spans[i + 1]
                    if next_span.get("attributes", {}).get("openinference.span.kind") == "TOOL":
                        tool_attrs = next_span.get("attributes", {})
                        tool_name = tool_attrs.get("tool.name", "")
                        try:
                            value = next_span.get("attributes", {}).get("input.value", "{}")
                            tool_input = json.loads(value) if isinstance(value, str) else value
                        except:
                            tool_input = value
                        try:
                            value = next_span.get("attributes", {}).get("output.value", "{}")
                            tool_response = json.loads(value) if isinstance(value, str) else value
                        except:
                            tool_response = value

                        # parse tool_args
                        tool_args = {}
                        if isinstance(tool_input, dict):
                            tool_args = tool_input
                        elif isinstance(tool_input, str):
                            # Try to parse parameters in string form
                            try:
                                tool_args = json.loads(tool_input)
                            except:
                                tool_args = {"input": tool_input}

                        if tool_name:
                            tools_called.append(
                                {
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "tool_response": tool_response,
                                }
                            )

                # Calculate usage
                input_tokens = int(attrs.get("llm.token_count.prompt", 0))
                output_tokens = int(attrs.get("llm.token_count.completion", 0))
                exec_time = (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"])) / 1e9

                accumulated["input_tokens"] += input_tokens
                accumulated["output_tokens"] += output_tokens
                accumulated["time"] += exec_time
                if last_agent_name != agent_name:
                    last_agent_name = agent_name
                    accumulated["transfers"] += 1

                agent_steps.append(
                    {
                        "step": step_num,
                        "agent_name": agent_name,
                        "agent": {"input": user_input, "output": output_text, "tools_called": tools_called},
                        "environment": None,
                        "step_usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": output_tokens,
                            "llm_inference_time": exec_time,
                            "model": attrs.get("llm.model_name"),
                            "step_execution_time": exec_time,
                        },
                        "accumulated_usage": {
                            "accumulated_input_tokens": accumulated["input_tokens"],
                            "accumulated_output_tokens": accumulated["output_tokens"],
                            "accumulated_time": accumulated["time"],
                            "accumulated_transferred_times": accumulated["transfers"],
                        },
                        "start_time": span["start_time_unix_nano"],
                        "end_time": span["end_time_unix_nano"],
                        "span_id": span["span_id"],
                        "trace_id": span["trace_id"],
                        "parent_span_id": span.get("parent_span_id", ""),
                    }
                )

            i += 1

        return agent_steps

    def extract_agent_settings(self) -> Dict:
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()

        for span in self.trace:
            if not span.get("attributes", {}).get("openinference.span.kind") == "LLM":
                continue

            # Skip spans without langgraph_node in metadata
            metadata = self.parse_json_value(span, "metadata")
            if "langgraph_node" not in metadata:
                continue

            # Extract system prompt from input.value
            input_data = self.parse_json_value(span, "input.value")
            system_content = ""
            attrs = span.get("attributes", {})

            # Try to find the system role message from messages
            if "messages" in input_data:
                messages = input_data["messages"]
                if isinstance(messages, list):
                    for msg_group in messages:
                        if isinstance(msg_group, list):
                            for msg in msg_group:
                                if isinstance(msg, dict) and msg.get("kwargs", {}).get("type") == "system":
                                    system_content = msg.get("kwargs", {}).get("content", "")
                                    break
                        elif isinstance(msg_group, dict) and msg_group.get("kwargs", {}).get("type") == "system":
                            system_content = msg_group.get("kwargs", {}).get("content", "")
                            break
                        if system_content:
                            break

            # If not found, try to extract from llm.input_messages as a fallback
            if not system_content:
                role = attrs.get("llm.input_messages.0.message.role", "")
                if role == "system":
                    system_content = attrs.get("llm.input_messages.0.message.content", "")

            if system_content:
                # Determine agent name based on metadata
                node_name = metadata.get("langgraph_node")
                if node_name not in agent_settings["prompt"]:
                    agent_settings["prompt"][node_name] = system_content

            # Extract tool definitions from llm.tools.N.tool.json_schema
            # This field is a JSON string, in the form of {"type":"function","function":{"name":...,"description":...,"parameters":{...}}}
            for k, v in attrs.items():
                if not isinstance(k, str):
                    continue
                if not k.startswith("llm.tools.") or not k.endswith(".tool.json_schema"):
                    continue
                try:
                    schema = json.loads(v) if isinstance(v, str) else (v or {})
                except Exception:
                    schema = {}
                fn = schema.get("function") or {}
                tool_name = fn.get("name")
                if tool_name and tool_name not in seen_tools:
                    seen_tools.add(tool_name)
                    description = (fn.get("description") or "").split("\n")[0]
                    parameters = fn.get("parameters", {})
                    agent_settings["tool"].append(
                        {
                            "name": tool_name,
                            "description": description,
                            "parameters": parameters,
                        }
                    )

        # If the tool definition cannot be extracted from llm.tools.N.tool.json_schema, try to extract it from the span where openinference.span.kind is TOOL
        for span in self.trace:
            if not span.get("attributes", {}).get("openinference.span.kind") == "TOOL":
                continue

            attrs = span.get("attributes", {})
            tool_name = attrs.get("tool.name")
            if tool_name and tool_name not in seen_tools:
                # Determine whether the tool has been extracted from the json_schema above through seen_tools
                seen_tools.add(tool_name)
                description = attrs.get("tool.description", "")

                # The parameters are inferred from the "input.value" field.
                # If "input.value" is something like "{'a': 12, 'b': 34}", then the parameters and their types are inferred from it;
                # If "input.value" is just a simple string, such as "SELECT Manufacturer, COUNT(*) AS count\nFROM club\nGROUP BY Manufacturer\nORDER BY count DESC\nLIMIT 1;", then the parameter defaults to input and the type is str.
                try:
                    value = span.get("attributes", {}).get("input.value", "{}")
                    input_data = json.loads(value) if isinstance(value, str) else value
                except:
                    input_data = value
                parameters = {"type": "OBJECT", "properties": {}, "required": []}
                if isinstance(input_data, dict) and input_data:
                    # If input is a dictionary, infer parameter types from values
                    for key, value in input_data.items():
                        if isinstance(value, int):
                            param_type = "INTEGER"
                        elif isinstance(value, float):
                            param_type = "NUMBER"  # or FLOAT？
                        elif isinstance(value, bool):
                            param_type = "BOOLEAN"
                        elif isinstance(value, list):
                            param_type = "ARRAY"
                        elif isinstance(value, dict):
                            param_type = "OBJECT"
                        else:
                            param_type = "STRING"

                        parameters["properties"][key] = {"type": param_type}
                        parameters["required"].append(key)
                elif isinstance(input_data, str):
                    # If input is a string, default to single input parameter
                    parameters["properties"]["input"] = {"type": "STRING"}
                    parameters["required"].append("input")

                agent_settings["tool"].append(
                    {
                        "name": tool_name,
                        "description": description,
                        "parameters": parameters,
                    }
                )

        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        """Trace the agent dependency relationships upwards based on ChatOpenAI"""
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})

        # 1. Find all LLM spans, and trace upwards from them to find agents
        valid_agents = set()
        for span in self.trace:
            if span.get("attributes", {}).get("openinference.span.kind") == "LLM":
                # Skip spans without langgraph_node in metadata
                metadata = self.parse_json_value(span, "metadata")
                if "langgraph_node" not in metadata:
                    continue
                # Trace all CHAIN ancestors upwards from the LLM span
                agent_name = metadata.get("langgraph_node")
                valid_agents.add(agent_name)
                # Ensure initialization in dependency
                _ = agent_dependency[agent_name]

        # 2. Build the hierarchy relationship between agents
        # For each valid agent, find its direct child agents
        for agent_name in valid_agents:
            # Find the span corresponding to this agent
            agent_span = None
            for span in self.trace:
                if (
                    span.get("attributes", {}).get("openinference.span.kind") == "CHAIN"
                    and span.get("name") == agent_name
                ):
                    agent_span = span
                    break

            if not agent_span:
                continue

            agent_span_id = agent_span.get("span_id")

            # Find other valid agents among the direct child spans
            for span in self.trace:
                if span.get("parent_span_id") == agent_span_id:
                    span_kind = span.get("attributes", {}).get("openinference.span.kind")
                    span_name = span.get("name")

                    # If it is a CHAIN span and in valid_agents, then it is a child agent
                    if span_kind == "CHAIN" and span_name in valid_agents:
                        agent_dependency[agent_name]["agent"].append(span_name)

        # 3. Find the tools used by each agent
        """
        Filter spans whose openinference.span.kind is LLM or TOOL, then sort these spans by start_time_unix_nano,
        and iterate over the spans in chronological order.

        If the LLM span has a tool_call attribute, then the agent corresponding to this LLM span owns that tool.

        For TOOL spans, assign the tool to the agent corresponding to the previous LLM span.
        """
        # Filter LLM and TOOL spans, and sort by time
        llm_and_tool_spans = [
            span for span in self.trace if span.get("attributes", {}).get("openinference.span.kind") in ["LLM", "TOOL"]
        ]
        sorted_spans = sorted(llm_and_tool_spans, key=lambda x: x["start_time_unix_nano"])
        last_llm_agent = None  # Track the most recent LLM agent

        i = 0
        while i < len(sorted_spans):
            span = sorted_spans[i]
            span_kind = span.get("attributes", {}).get("openinference.span.kind")

            if span_kind == "LLM":
                attrs = span.get("attributes", {})

                # Parse basic info from the LLM span
                # Get agent name from metadata
                metadata = self.parse_json_value(span, "metadata")
                agent_name = metadata.get("langgraph_node")
                if not agent_name or agent_name not in valid_agents:
                    i += 1
                    continue

                last_llm_agent = agent_name
                # Parse user input from input.value
                input_data = self.parse_json_value(span, "input.value")

                # First check whether input.data contains tool_calls
                tool_calls_found = False

                # Extract tool_calls from input.data
                if "tool_calls" in input_data:
                    tool_calls = input_data.get("tool_calls", [])
                    if tool_calls:
                        for tool_call in tool_calls:
                            if isinstance(tool_call, dict):
                                tool_name = tool_call.get("function", {}).get("name", "")
                                if tool_name:
                                    tool_calls_found = True
                                    agent_dependency[agent_name]["tool"].append(tool_name)

                # # If the LLM span has no tool_call info, check for TOOL spans with same parent
                # if not tool_calls_found:
                #     llm_parent_id = span.get("parent_span_id")
                #     for tool_span in self.trace:
                #         if tool_span.get("attributes", {}).get("openinference.span.kind") == "TOOL":
                #             if tool_span.get("parent_span_id") == llm_parent_id:
                #                 tool_name = tool_span.get("attributes", {}).get("tool.name", "")
                #                 if tool_name:
                #                     agent_dependency[agent_name]["tool"].append(tool_name)

            elif span_kind == "TOOL":
                # Extract tool info directly from TOOL span's metadata
                metadata = self.parse_json_value(span, "metadata")
                node_name = metadata.get("langgraph_node")
                tool_name = span.get("attributes", {}).get("tool.name", "")
                # Assign tool to valid agent or fall back to last LLM agent
                if tool_name:
                    target_agent = node_name if node_name in valid_agents else last_llm_agent
                    if target_agent:
                        agent_dependency[target_agent]["tool"].append(tool_name)

            i += 1

        # 4. Deduplicate and sort
        return {
            k: {"agent": sorted(list(set(v["agent"]))), "tool": sorted(list(set(v["tool"])))}
            for k, v in agent_dependency.items()
        }


class AutoGenPhoenixParser(BaseTraceParser):
    def extract_agent_steps(self) -> List[Dict]:
        """Extract agent_steps"""
        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        # Filter ChatCompletion spans and sort by time
        chat_completion_spans = [span for span in self.trace if span.get("name") == "ChatCompletion"]
        sorted_spans = sorted(chat_completion_spans, key=lambda x: x["start_time_unix_nano"])

        for span in sorted_spans:
            step_num += 1
            attrs = span.get("attributes", {})

            # Trace upwards from the ChatCompletion span to get agent name
            agent_name = self._get_agent_name_from_span(span)

            # Parse user input from input.value
            input_data = self.parse_json_value(span, "input.value")
            user_input = []

            # Parse messages: filter out system and tool messages, keep user and assistant messages
            messages = input_data.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        user_input.append({"role": role, "content": content})
            # Keep the last one
            user_input = user_input[-1:] if user_input else []

            # Parse output text from output.value
            output_data = self.parse_json_value(span, "output.value")
            output_text = ""

            choices = output_data.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                output_text = message.get("content", "")

            # Parse tool calls
            tools_called = []

            # First check whether output.value contains tool_calls
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_name = function_info.get("name", "")
                        tool_args_str = function_info.get("arguments", "{}")
                        tool_call_id = tool_call.get("id", "")

                        try:
                            tool_args = json.loads(tool_args_str) if tool_args_str else {}
                        except:
                            tool_args = {}

                        if tool_name:
                            # Find the corresponding tool response (look in the next ChatCompletion span input)
                            tool_response = self._get_tool_response(tool_call_id, sorted_spans, span)

                            tools_called.append(
                                {
                                    "tool_name": tool_name,
                                    "tool_args": tool_args,
                                    "tool_response": tool_response,
                                }
                            )

            # Compute usage
            usage = output_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            exec_time = (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"])) / 1e9

            accumulated["input_tokens"] += input_tokens
            accumulated["output_tokens"] += output_tokens
            accumulated["time"] += exec_time
            if last_agent_name != agent_name:
                last_agent_name = agent_name
                accumulated["transfers"] += 1

            agent_steps.append(
                {
                    "step": step_num,
                    "agent_name": agent_name,
                    "agent": {"input": user_input, "output": output_text, "tools_called": tools_called},
                    "environment": None,
                    "step_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_inference_time": exec_time,
                        "model": output_data.get("model", attrs.get("llm.model_name", "")),
                        "step_execution_time": exec_time,
                    },
                    "accumulated_usage": {
                        "accumulated_input_tokens": accumulated["input_tokens"],
                        "accumulated_output_tokens": accumulated["output_tokens"],
                        "accumulated_time": accumulated["time"],
                        "accumulated_transferred_times": accumulated["transfers"],
                    },
                    "start_time": span["start_time_unix_nano"],
                    "end_time": span["end_time_unix_nano"],
                    "span_id": span["span_id"],
                    "trace_id": span["trace_id"],
                    "parent_span_id": span.get("parent_span_id", ""),
                }
            )

        return agent_steps

    def extract_agent_settings(self) -> Dict:
        """Extract agent settings and tool definitions"""
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()

        # Iterate over ChatCompletion spans to get system prompts and tool definitions
        for span in self.trace:
            if span.get("name") != "ChatCompletion":
                continue

            attrs = span.get("attributes", {})
            agent_name = self._get_agent_name_from_span(span)

            # Extract system prompt from input.value
            input_data = self.parse_json_value(span, "input.value")
            messages = input_data.get("messages", [])

            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    if system_content and agent_name not in agent_settings["prompt"]:
                        agent_settings["prompt"][agent_name] = system_content
                    break

            # Extract tool definitions from input.value
            tools = input_data.get("tools", [])
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    function_info = tool.get("function", {})
                    tool_name = function_info.get("name", "")

                    if tool_name and tool_name not in seen_tools:
                        seen_tools.add(tool_name)
                        description = function_info.get("description", "")
                        parameters = function_info.get("parameters", {})

                        agent_settings["tool"].append(
                            {
                                "name": tool_name,
                                "description": description,
                                "parameters": parameters,
                            }
                        )

        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        """Extract agent dependency relationships"""
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})

        # Collect all agent names
        all_agents = set()
        for span in self.trace:
            if span.get("name") == "ChatCompletion":
                agent_name = self._get_agent_name_from_span(span)
                all_agents.add(agent_name)
                # Ensure initialization in dependency
                _ = agent_dependency[agent_name]

        # Analyze tool dependencies
        for span in self.trace:
            if span.get("name") != "ChatCompletion":
                continue

            agent_name = self._get_agent_name_from_span(span)

            # Extract tool calls from output.value
            output_data = self.parse_json_value(span, "output.value")
            choices = output_data.get("choices", [])

            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_name = function_info.get("name", "")
                        if tool_name:
                            agent_dependency[agent_name]["tool"].append(tool_name)

        # Deduplicate and sort
        return {
            k: {"agent": sorted(list(set(v["agent"]))), "tool": sorted(list(set(v["tool"])))}
            for k, v in agent_dependency.items()
        }

    def _get_agent_name_from_span(self, span: Dict) -> str:
        """Get agent name by tracing upwards from a ChatCompletion span"""
        current_span = span

        while current_span:
            parent_span_id = current_span.get("parent_span_id")
            if not parent_span_id:
                break

            parent_span = self.span_map.get(parent_span_id)
            if not parent_span:
                break

            # Find a span name like "autogen process xxx.(default)-A"
            parent_name = parent_span.get("name", "")
            if parent_name.startswith("autogen process ") and ".(default)-" in parent_name:
                # Extract agent name: "autogen process agent_name.(default)-X"
                match = re.search(r"autogen process (.+?)\.\(default\)-.", parent_name)
                if match:
                    return match.group(1)

            current_span = parent_span

        return "unknown"

    def _get_tool_response(self, tool_call_id: str, sorted_spans: List[Dict], current_span: Dict) -> Any:
        """Get tool response"""
        # Search for tool response messages in spans after the current span
        current_index = sorted_spans.index(current_span)

        for i in range(current_index + 1, len(sorted_spans)):
            next_span = sorted_spans[i]
            input_data = self.parse_json_value(next_span, "input.value")
            messages = input_data.get("messages", [])

            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id") == tool_call_id:
                    return msg.get("content", "")

        return None


class AgnoPhoenixParser(BaseTraceParser):
    def extract_agent_steps(self) -> List[Dict]:
        """Extract agent_steps"""
        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        # Get ChatCompletion spans under the openai scope (LLM calls)
        openai_llm_spans = [span for span in self.trace if span.get("name") == "ChatCompletion"]

        # Sort by time
        sorted_spans = sorted(openai_llm_spans, key=lambda x: x["start_time_unix_nano"])

        for span in sorted_spans:
            step_num += 1
            attrs = span.get("attributes", {})

            # Get agent name from the parent span chain
            agent_name = self._get_agent_name_from_agno_span(span)

            # Parse user input from input.value
            input_data = self.parse_json_value(span, "input.value")
            user_input = []

            # Parse messages, extract user and assistant messages (exclude system)
            messages = input_data.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["user", "assistant"] and content:
                        user_input.append({"role": role, "content": content})
            # Keep the last one
            user_input = user_input[-1:] if user_input else []

            # Parse output text from output.value
            output_data = self.parse_json_value(span, "output.value")
            output_text = ""

            choices = output_data.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                output_text = message.get("content", "")

            # Parse tool calls
            tools_called = []

            # Check tool_calls in output.value
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_name = function_info.get("name", "")
                        tool_args_str = function_info.get("arguments", "{}")

                        try:
                            tool_args = json.loads(tool_args_str) if tool_args_str else {}
                        except:
                            tool_args = {}

                        # Find tool response from agno scope spans
                        tool_response = self._get_tool_response_from_agno(tool_name, span)

                        if tool_name:
                            tools_called.append(
                                {
                                    "tool_name": tool_name,
                                    "arguments": tool_args,
                                    "result": tool_response,
                                }
                            )

            # Compute usage
            usage = output_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            exec_time = (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"])) / 1e9

            accumulated["input_tokens"] += input_tokens
            accumulated["output_tokens"] += output_tokens
            accumulated["time"] += exec_time
            if last_agent_name != agent_name:
                last_agent_name = agent_name
                accumulated["transfers"] += 1

            agent_steps.append(
                {
                    "step": step_num,
                    "agent_name": agent_name,
                    "agent": {"input": user_input, "output": output_text, "tools_called": tools_called},
                    "environment": None,
                    "step_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_inference_time": exec_time,
                        "model": output_data.get("model", attrs.get("llm.model_name", "")),
                        "step_execution_time": exec_time,
                    },
                    "accumulated_usage": {
                        "accumulated_input_tokens": accumulated["input_tokens"],
                        "accumulated_output_tokens": accumulated["output_tokens"],
                        "accumulated_time": accumulated["time"],
                        "accumulated_transferred_times": accumulated["transfers"],
                    },
                    "start_time": span["start_time_unix_nano"],
                    "end_time": span["end_time_unix_nano"],
                    "span_id": span["span_id"],
                    "trace_id": span["trace_id"],
                    "parent_span_id": span.get("parent_span_id", ""),
                }
            )

        return agent_steps

    def extract_agent_settings(self) -> Dict:
        """Extract agent settings and tool definitions"""
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()
        seen_agents = set()
        # Iterate over ChatCompletion spans to get system prompts and tool definitions
        for span in self.trace:
            if span.get("name") != "ChatCompletion":
                continue
            agent_name = self._get_agent_name_from_agno_span(span)
            # Extract system prompt from input.value
            input_data = self.parse_json_value(span, "input.value")
            messages = input_data.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    if system_content and agent_name not in seen_agents:
                        agent_settings["prompt"][agent_name] = system_content
                        seen_agents.add(agent_name)
                    break
            # Extract tool definitions from input.value
            tools = input_data.get("tools", [])
            for tool in tools:
                if isinstance(tool, dict) and tool.get("type") == "function":
                    function_info = tool.get("function", {})
                    tool_name = function_info.get("name", "")
                    if tool_name and tool_name not in seen_tools:
                        seen_tools.add(tool_name)
                        description = function_info.get("description", "")
                        parameters = function_info.get("parameters", {})
                        agent_settings["tool"].append(
                            {
                                "name": tool_name,
                                "description": description,
                                "parameters": parameters,
                            }
                        )
            # Extract tool definitions from actual tool_calls in output.value
            output_data = self.parse_json_value(span, "output.value")
            choices = output_data.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_name = function_info.get("name", "")
                        if tool_name and tool_name not in seen_tools:
                            seen_tools.add(tool_name)
                            # Infer parameters from arguments
                            tool_args_str = function_info.get("arguments", "{}")
                            try:
                                tool_args = json.loads(tool_args_str) if tool_args_str else {}
                                parameters = {
                                    "type": "object",
                                    "properties": {k: {"type": "string"} for k in tool_args.keys()},
                                }
                            except:
                                parameters = {}
                            agent_settings["tool"].append(
                                {
                                    "name": tool_name,
                                    "description": "",
                                    "parameters": parameters,
                                }
                            )
        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        """Extract agent dependency relationships based on parent-child spans"""
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})

        # Collect all agent names
        all_agents = set()
        for span in self.trace:
            if span.get("name") == "ChatCompletion":
                agent_name = self._get_agent_name_from_agno_span(span)
                all_agents.add(agent_name)
                # Ensure initialization in dependency
                _ = agent_dependency[agent_name]

        # Analyze tool and agent dependencies
        for span in self.trace:
            if span.get("name") != "ChatCompletion":
                continue

            agent_name = self._get_agent_name_from_agno_span(span)

            # Extract tool calls from output.value
            output_data = self.parse_json_value(span, "output.value")
            choices = output_data.get("choices", [])

            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                tool_calls = message.get("tool_calls", [])

                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        function_info = tool_call.get("function", {})
                        tool_name = function_info.get("name", "")
                        if tool_name:
                            agent_dependency[agent_name]["tool"].append(tool_name)

            # Check whether child agents are called (via parent-child span relationships)
            span_id = span.get("span_id")
            for potential_child in self.trace:
                if potential_child.get("parent_span_id") == span_id:
                    # If the child span is also an agent call
                    if potential_child.get("name") == "ChatCompletion":
                        child_agent_name = self._get_agent_name_from_agno_span(potential_child)
                        if child_agent_name != agent_name and child_agent_name in all_agents:
                            agent_dependency[agent_name]["agent"].append(child_agent_name)
                    # Find called child agents from agno scope
                    elif potential_child.get("attributes", {}).get("openinference.span.kind") == "AGENT":
                        # Extract agent name from span name
                        child_agent_name = self._extract_agent_name_from_agno_span_name(potential_child)
                        if child_agent_name and child_agent_name in all_agents:
                            agent_dependency[agent_name]["agent"].append(child_agent_name)

        # Deduplicate and sort
        return {
            k: {"agent": sorted(list(set(v["agent"]))), "tool": sorted(list(set(v["tool"])))}
            for k, v in agent_dependency.items()
        }

    def _get_agent_name_from_agno_span(self, span: Dict) -> str:
        """Get agent name from an Agno span"""
        # Walk up parent spans to find an AGENT span under agno scope
        current_span = span

        while current_span:
            parent_span_id = current_span.get("parent_span_id")
            if not parent_span_id:
                break

            parent_span = self.span_map.get(parent_span_id)
            if not parent_span:
                break

            # Check if this is an agno AGENT span
            if parent_span.get("attributes", {}).get("openinference.span.kind") == "AGENT":
                return self._extract_agent_name_from_agno_span_name(parent_span)

            current_span = parent_span
        # Fallback: infer agent name from system prompt content
        return self._infer_agent_name_from_prompt(span)

    def _infer_agent_name_from_prompt(self, span: Dict) -> str:
        """Infer agent name from system prompt when parent span is unavailable"""
        input_data = self.parse_json_value(span, "input.value")
        messages = input_data.get("messages", [])
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                # Try to identify agent type from prompt content
                content_lower = content.lower()
                if "devops assistant" in content_lower and "python code" in content_lower:
                    return "ExecutionAgent"
                if "administrator" in content_lower or "reasoning" in content_lower:
                    return "ReasoningAgent"
                if "final" in content_lower and "answer" in content_lower:
                    return "FinalAgent"
                break
        return "unknown"

    def _extract_agent_name_from_agno_span_name(self, span: Dict) -> str:
        """Extract agent name from an agno span name"""
        # agno span name format: "ck_plan_agent.run", "ck_action_agent.run", "web_agent.run"
        span_name = span.get("name", "")
        if ".run" in span_name:
            return span_name.replace(".run", "")
        return span_name

    def _get_tool_response_from_agno(self, tool_name: str, llm_span: Dict) -> Any:
        """Get tool response from agno scope spans"""
        # Find tool execution spans related to the current LLM span
        span_id = llm_span.get("span_id")
        trace_id = llm_span.get("trace_id")

        # Find tool execution results in the same trace
        for span in self.trace:
            # Find TOOL spans in agno scope
            if span.get("trace_id") == trace_id and span.get("attributes", {}).get("openinference.span.kind") == "TOOL":

                # Check whether the tool name matches
                attrs = span.get("attributes", {})
                span_tool_name = attrs.get("tool.name", "")

                if span_tool_name == tool_name:
                    # Get tool output
                    output_value = attrs.get("output.value", "")
                    return output_value

        return None


class CrewAIPhoenixParser(BaseTraceParser):
    def extract_agent_steps(self) -> List[Dict]:
        """Extract agent_steps"""
        agent_steps = []
        step_num = 0
        accumulated = {"input_tokens": 0, "output_tokens": 0, "time": 0, "transfers": -1}
        last_agent_name = None

        # Sort all spans by time
        sorted_spans = sorted(self.trace, key=lambda x: x["start_time_unix_nano"])

        i = 0
        while i < len(sorted_spans):
            span = sorted_spans[i]

            # Only process ChatCompletion spans
            if span.get("name") != "ChatCompletion":
                i += 1
                continue

            step_num += 1
            attrs = span.get("attributes", {})

            # Parse user input from input.value
            input_data = self.parse_json_value(span, "input.value")
            user_input = []

            # Parse messages, extract system and user messages
            messages = input_data.get("messages", [])
            for msg in messages:
                if isinstance(msg, dict):
                    role = msg.get("role", "")
                    content = msg.get("content", "")
                    if role in ["system", "user"] and content:
                        user_input.append({"role": role, "content": content})
            # Keep the last one
            user_input = user_input[-1:] if user_input else []

            # Parse output text from output.value
            output_data = self.parse_json_value(span, "output.value")
            output_text = ""

            choices = output_data.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                output_text = message.get("content", "")

            # Find TOOL spans right after this ChatCompletion
            tools_called = []

            j = i + 1
            while j < len(sorted_spans):
                next_span = sorted_spans[j]

                # Stop when encountering the next ChatCompletion
                if next_span.get("name") == "ChatCompletion":
                    break

                # Check whether it is a TOOL span
                if next_span.get("attributes", {}).get("openinference.span.kind") == "TOOL":
                    tool_attrs = next_span.get("attributes", {})
                    tool_name = tool_attrs.get("tool.name", "")

                    # Get tool params from input.value
                    tool_input_value = self.parse_json_value(next_span, "input.value")

                    # Try to extract arguments from the calling field
                    tool_arguments = {}
                    if isinstance(tool_input_value, dict):
                        calling_str = tool_input_value.get("calling", "")
                        if calling_str:
                            # calling format: "tool_name='xxx' arguments={...}"
                            # extract arguments part
                            import ast

                            try:
                                args_match = re.search(r"arguments=(\{.+\})$", calling_str)
                                if args_match:
                                    args_str = args_match.group(1)
                                    # Safely parse with ast.literal_eval
                                    tool_arguments = ast.literal_eval(args_str)
                            except Exception as e:
                                logger.warning(f"Failed to parse tool arguments from calling: {e}")
                                tool_arguments = {}

                        # If calling parsing fails, try using tool_input_value directly
                        if not tool_arguments and tool_input_value:
                            # Tool args might be at the top level of input_value
                            tool_arguments = {
                                k: v for k, v in tool_input_value.items() if k not in ["tool_string", "tool", "calling"]
                            }

                    # Get tool result from output.value
                    tool_output = next_span.get("attributes", {}).get("output.value", "{}")

                    if tool_name:
                        tools_called.append(
                            {"tool_name": tool_name, "arguments": tool_arguments, "result": tool_output}
                        )

                j += 1

            # Get agent name from the span parent chain
            agent_name = self._get_agent_name_from_crewai_span(span)

            # Compute usage
            usage = output_data.get("usage", {})
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            exec_time = (int(span["end_time_unix_nano"]) - int(span["start_time_unix_nano"])) / 1e9

            accumulated["input_tokens"] += input_tokens
            accumulated["output_tokens"] += output_tokens
            accumulated["time"] += exec_time
            if last_agent_name != agent_name:
                last_agent_name = agent_name
                accumulated["transfers"] += 1

            agent_steps.append(
                {
                    "step": step_num,
                    "agent_name": agent_name,
                    "agent": {"input": user_input, "output": output_text, "tools_called": tools_called},
                    "environment": None,
                    "step_usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "llm_inference_time": exec_time,
                        "model": output_data.get("model", attrs.get("llm.model_name", "")),
                        "step_execution_time": exec_time,
                    },
                    "accumulated_usage": {
                        "accumulated_input_tokens": accumulated["input_tokens"],
                        "accumulated_output_tokens": accumulated["output_tokens"],
                        "accumulated_time": accumulated["time"],
                        "accumulated_transferred_times": accumulated["transfers"],
                    },
                    "start_time": span["start_time_unix_nano"],
                    "end_time": span["end_time_unix_nano"],
                    "span_id": span["span_id"],
                    "trace_id": span["trace_id"],
                    "parent_span_id": span.get("parent_span_id", ""),
                }
            )

            i += 1

        return agent_steps

    def extract_agent_settings(self) -> Dict:
        """Extract agent settings and tool definitions"""
        agent_settings = {"prompt": {}, "tool": []}
        seen_tools = set()
        seen_agents = set()

        # Iterate over ChatCompletion spans to get system prompts
        for span in self.trace:
            if span.get("name") != "ChatCompletion":
                continue

            agent_name = self._get_agent_name_from_crewai_span(span)

            # Extract system prompt from input.value
            input_data = self.parse_json_value(span, "input.value")
            messages = input_data.get("messages", [])

            for msg in messages:
                if isinstance(msg, dict) and msg.get("role") == "system":
                    system_content = msg.get("content", "")
                    if system_content and agent_name not in seen_agents:
                        agent_settings["prompt"][agent_name] = system_content
                        seen_agents.add(agent_name)
                    break

        # Extract tool definitions from xxx._use spans' input.value
        for span in self.trace:
            span_name = span.get("name", "")

            # Check whether it is a tool usage span
            if span_name.endswith("._use"):
                attrs = span.get("attributes", {})
                input_value_str = attrs.get("input.value", "")

                if not input_value_str:
                    continue

                # Parse input.value to get tool info
                try:
                    # Try JSON parsing first
                    input_value = json.loads(input_value_str) if isinstance(input_value_str, str) else input_value_str

                    # input_value should contain a tool field as a string
                    tool_str = input_value.get("tool", "")
                    if not tool_str:
                        continue

                    # tool_str looks like: "CrewStructuredTool(name='Bash Command Executor', description='...')"
                    # We need to parse this string to extract name and description

                    # Extract tool name
                    tool_name = span_name.replace("._use", "")
                    name_match = re.search(r"name=['\"]([^'\"]+)['\"]", tool_str)
                    if name_match:
                        tool_name = name_match.group(1)

                    if tool_name and tool_name not in seen_tools:
                        # Extract tool description (improved): find everything after description=' until the final ')
                        # Strategy: find the quote after description=, then match to the last same quote before the last ')'
                        tool_description = f"Tool: {tool_name}"  # default

                        # Try extracting description value
                        if "description=" in tool_str:
                            # Locate description=
                            desc_start_idx = tool_str.find("description=")
                            desc_content = tool_str[desc_start_idx:]

                            # Determine quote type
                            if desc_content.startswith("description='"):
                                quote_char = "'"
                                desc_value_start = desc_start_idx + len("description='")
                            elif desc_content.startswith('description="'):
                                quote_char = '"'
                                desc_value_start = desc_start_idx + len('description="')
                            else:
                                quote_char = None

                            if quote_char:
                                # Search backwards for the last matching quote (before the final ')')
                                end_paren_idx = tool_str.rfind(")")
                                last_quote_idx = tool_str.rfind(quote_char, desc_value_start, end_paren_idx)

                                if last_quote_idx > desc_value_start:
                                    tool_description = tool_str[desc_value_start:last_quote_idx]

                        # Extract more detailed info from description
                        # description format: "Tool Name: xxx\nTool Arguments: {...}\nTool Description: ..."
                        tool_args_dict = {}
                        full_description = tool_description

                        if "Tool Arguments:" in tool_description and "Tool Description:" in tool_description:
                            # Extract Tool Arguments part
                            args_start = tool_description.find("Tool Arguments:") + len("Tool Arguments:")
                            args_end = tool_description.find("Tool Description:")
                            args_str = tool_description[args_start:args_end].strip()

                            # Extract Tool Description part
                            desc_start = tool_description.find("Tool Description:") + len("Tool Description:")
                            full_description = tool_description[desc_start:].strip()
                            # Remove leading escaped newline
                            if full_description.startswith("\\n"):
                                full_description = full_description[2:].strip()

                            # Parse arguments
                            try:
                                # args_str looks like: "{'command': {'description': '...', 'type': 'str'}}"
                                # Handle escape chars
                                args_str = args_str.replace("\\'", "'").replace('\\"', '"')
                                tool_args_dict = eval(args_str) if args_str.startswith("{") else {}
                            except Exception as e:
                                logger.warning(f"Failed to parse tool arguments for {tool_name}: {e}")
                                tool_args_dict = {}

                        # Build parameters
                        parameters = {"type": "object", "properties": {}, "required": []}

                        for arg_name, arg_info in tool_args_dict.items():
                            if isinstance(arg_info, dict):
                                param_type = arg_info.get("type", "string")
                                param_desc = arg_info.get("description", "")

                                parameters["properties"][arg_name] = {"type": param_type, "description": param_desc}
                                parameters["required"].append(arg_name)

                        # Build tool definition
                        tool_def = {
                            "type": "function",
                            "function": {"name": tool_name, "description": full_description, "parameters": parameters},
                        }

                        agent_settings["tool"].append(tool_def)
                        seen_tools.add(tool_name)

                except Exception as e:
                    logger.warning(f"Failed to parse tool info from span {span_name}: {e}")
                    # Fallback: use a simple tool name
                    tool_name = span_name.replace("._use", "")
                    if tool_name and tool_name not in seen_tools:
                        tool_def = {
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "description": f"Tool: {tool_name}",
                                "parameters": {"type": "object", "properties": {}, "required": []},
                            },
                        }
                        agent_settings["tool"].append(tool_def)
                        seen_tools.add(tool_name)

        return agent_settings

    def extract_agent_dependency(self) -> Dict:
        """Extract agent dependency relationships"""
        agent_dependency = defaultdict(lambda: {"agent": [], "tool": []})

        # Collect all agent names
        all_agents = set()
        for span in self.trace:
            if span.get("name") == "ChatCompletion":
                agent_name = self._get_agent_name_from_crewai_span(span)
                all_agents.add(agent_name)
                # Ensure initialization in dependency
                _ = agent_dependency[agent_name]

        # Sort all spans by time
        sorted_spans = sorted(self.trace, key=lambda x: x["start_time_unix_nano"])

        i = 0
        while i < len(sorted_spans):
            span = sorted_spans[i]

            # If it is a ChatCompletion span
            if span.get("name") == "ChatCompletion":
                agent_name = self._get_agent_name_from_crewai_span(span)

                # Look for subsequent tool calls
                j = i + 1
                while j < len(sorted_spans):
                    next_span = sorted_spans[j]
                    next_span_name = next_span.get("name", "")

                    # Stop when encountering the next ChatCompletion
                    if next_span.get("name") == "ChatCompletion":
                        break

                    # If it is a tool usage span
                    if (
                        next_span_name.endswith("._use")
                        or next_span.get("attributes", {}).get("openinference.span.kind") == "TOOL"
                    ):
                        tool_name = (
                            next_span_name.replace("._use", "")
                            if next_span_name.endswith("._use")
                            else next_span.get("attributes", {}).get("tool.name", next_span_name)
                        )
                        if tool_name and tool_name not in agent_dependency[agent_name]["tool"]:
                            agent_dependency[agent_name]["tool"].append(tool_name)

                    j += 1

            i += 1

        # Deduplicate and sort
        return {
            k: {"agent": sorted(list(set(v["agent"]))), "tool": sorted(list(set(v["tool"])))}
            for k, v in agent_dependency.items()
        }

    def _get_agent_name_from_crewai_span(self, span: Dict) -> str:
        """Get agent name from a CrewAI span"""
        # Extract agent name from system message
        input_data = self.parse_json_value(span, "input.value")
        messages = input_data.get("messages", [])

        agent_name = "Software Engineer Assistant"  # default
        system_content = ""
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                system_content = msg.get("content", "")
                # Try extracting agent name from system message
                if "You are" in system_content:
                    lines = system_content.split("\n")
                    for line in lines:
                        if line.startswith("You are "):
                            agent_name = line.replace("You are ", "").strip().rstrip(".")
                            break
                break

        return agent_name
