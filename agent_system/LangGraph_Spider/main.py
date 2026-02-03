# main.py
"""
Adapted from https://python.langchain.com/docs/tutorials/sql_qa/
as well as https://langchain-ai.github.io/langgraph/tutorials/sql-agent/
"""

import os
import sys
import argparse

os.chdir(os.path.dirname(os.path.abspath(__file__)))
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from dotenv import load_dotenv

load_dotenv()

# ============ workdir ==============
parser = argparse.ArgumentParser()
parser.add_argument(
    "--workdir",
    type=str,
    default="./workdir-dev/spider-0014_20251104113447",
    help="working directory",
)
args, _ = parser.parse_known_args()
# ==================================

# ========== metric, log, trace =============
from evoagentops.observability.metrics import CustomResourceMonitor
from evoagentops.observability.trace import CustomHTTPInterceptor

monitor = CustomResourceMonitor(output_dir=f"{args.workdir}/metrics")
custom_http_interceptor = CustomHTTPInterceptor(output_dir=f"{args.workdir}/trace")

from evoagentops.observability.log import agent_logger

agent_logger(level="INFO", log_dir=f"{args.workdir}/logs")

import asyncio
from phoenix.otel import register

# optimize batch size, make sure all spans are sent in one batch
os.environ["OTEL_BSP_SCHEDULE_DELAY"] = "999999"
os.environ["OTEL_BSP_MAX_EXPORT_BATCH_SIZE"] = "100000"
os.environ["OTEL_BSP_MAX_QUEUE_SIZE"] = "200000"
os.environ["OTEL_BSP_EXPORT_TIMEOUT"] = "60000"
os.environ["OTEL_BSP_FORCE_FLUSH_TIMEOUT"] = "60000"

# Phoenix tracing
tracer_provider = register(
    endpoint="http://localhost:6006",
    project_name="sql-agents",
    batch=True,
    auto_instrument=True,
)
# =====================================

import json
import re
import tempfile
import time
import shutil
from typing import Any, Literal, Optional, Dict, List

import termcolor
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.utilities import SQLDatabase
from langchain.chat_models import init_chat_model
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from spider_eval.exec_eval import eval_exec_match
from pydantic import BaseModel, Field
from prompt import (
    WRITE_QUERY_SYSTEM,
    WRITE_QUERY_USER,
    CHECK_QUERY_SYSTEM,
    CHECK_QUERY_USER,
    REWRITE_QUERY_SYSTEM,
    REWRITE_QUERY_USER,
)


class LLM(BaseModel):
    """Provide an LLM endpoint and model name as a resource."""

    resource_type: Literal["llm"] = "llm"
    endpoint: str
    model: str
    sampling_parameters: Dict[str, Any] = Field(default_factory=dict)


WRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        ("system", WRITE_QUERY_SYSTEM),
        ("user", WRITE_QUERY_USER),
    ]
)

CHECK_QUERY_PROMPT = ChatPromptTemplate(
    [
        ("system", CHECK_QUERY_SYSTEM),
        ("user", CHECK_QUERY_USER),
    ]
)

REWRITE_QUERY_PROMPT = ChatPromptTemplate(
    [
        ("system", REWRITE_QUERY_SYSTEM),
        ("user", REWRITE_QUERY_USER),
    ]
)


class State(MessagesState):
    question: str
    query: str
    execution: str
    answer: str
    feedback: str
    num_turns: int
    messages: list[BaseMessage]


# Check if model is a reasoning model that doesn't support temperature
def is_reasoning_model(name: str) -> bool:
    return any(x in name.lower() for x in ["o1", "o3", "gpt-5.2"])


class SQLAgent:
    def __init__(
        self,
        db: str,
        max_turns: int = 5,
        debug: bool = False,
        db_schema: str | None = None,
        endpoint: str | None = None,
        verl_replacement: dict | None = None,
        table_info_truncate: int = 2048,
        execution_truncate: int = 2048,
    ):
        self.db = SQLDatabase.from_uri(db)
        self.db_schema = db_schema
        self.debug = debug
        self.max_turns = max_turns
        self.table_info_truncate = table_info_truncate
        self.execution_truncate = execution_truncate

        if verl_replacement is not None:
            self.model_name = verl_replacement["model"]
            assert endpoint is not None
            llm_kwargs = {
                "model_provider": "openai",
                "openai_api_base": endpoint,
                "openai_api_key": os.getenv("OPENAI_API_KEY", "dummy"),
                "max_retries": 0,
                "max_tokens": 2048,
                "model_kwargs": {
                    "extra_body": (
                        {"reasoning_effort": "medium"}
                        if "gpt" in os.getenv("OPENAI_MODEL").lower()
                        else {"thinking": {"type": "disabled"}}
                    )
                },
            }
            if not is_reasoning_model(self.model_name):
                llm_kwargs["temperature"] = verl_replacement["temperature"]
            self.llm = init_chat_model(
                self.model_name,
                **llm_kwargs,
            )
        else:
            self.model_name = os.getenv("OPENAI_MODEL")
            llm_kwargs = {
                "model_provider": "openai",
                "openai_api_base": endpoint or os.getenv("OPENAI_BASE_URL"),
                "openai_api_key": os.getenv("OPENAI_API_KEY"),
                "max_retries": 1,
                "max_tokens": 2048,
                "model_kwargs": {
                    "extra_body": (
                        {"reasoning_effort": "medium"}
                        if "gpt" in os.getenv("OPENAI_MODEL").lower()
                        else {"thinking": {"type": "disabled"}}
                    )
                },
            }
            if not is_reasoning_model(self.model_name):
                llm_kwargs["temperature"] = 0
            self.llm = init_chat_model(
                self.model_name,
                **llm_kwargs,
            )

    def get_table_info(self) -> str:
        try:
            table_info = self.db.get_table_info()
            # if len(table_info) > self.table_info_truncate:
            # table_info = table_info[: self.table_info_truncate] + "\n... (truncated)"
            return table_info
        except Exception as e:
            print(f"Failed to get table info: {e}")
            if self.db_schema:
                # if len(self.db_schema) > self.table_info_truncate:
                # return self.db_schema[: self.table_info_truncate] + "\n... (truncated)"
                return self.db_schema
            return "No schema available."

    def invoke_prompt(self, prompt: Any) -> BaseMessage:
        if self.debug:
            for message in prompt.messages:
                termcolor.cprint(message.pretty_repr(), "blue")
        try:
            result = self.llm.invoke(prompt)
        except Exception as e:
            print(f"Failed to invoke prompt: {e}")
            result = self.llm.invoke([HumanMessage(content="Please create a random SQL query as an example.")])
        if self.debug:
            termcolor.cprint(result.pretty_repr(), "green")
        return result

    def truncate_execuion(self, execution: str) -> str:
        """Truncate the execution result to a reasonable length."""
        # if len(execution) > self.execution_truncate:
        # return execution[: self.execution_truncate] + "\n... (truncated)"
        return execution

    def parse_query(self, message: BaseMessage) -> str | None:
        result = None
        for match in re.finditer(r".*```\w*\n(.*?)\n```.*", message.content, re.DOTALL):
            result = match.group(1).strip()
        return result

    def write_query(self, state: State):
        prompt = WRITE_QUERY_PROMPT.invoke(
            {"dialect": self.db.dialect, "input": state["question"], "table_info": self.get_table_info()}
        )
        result = self.invoke_prompt(prompt)
        query = self.parse_query(result) or result.content
        return {**state, "query": query, "num_turns": 1, "messages": [*prompt.messages, result]}

    def execute_query(self, state: State) -> State:
        execute_query_tool = QuerySQLDatabaseTool(db=self.db)
        execution_result = execute_query_tool.invoke(state["query"])
        if not isinstance(execution_result, str):
            execution_result = str(execution_result)
        if self.debug:
            termcolor.cprint(execution_result, "yellow")
        return {**state, "execution": execution_result}

    def check_query(self, state: State) -> State:
        prompt = CHECK_QUERY_PROMPT.invoke(
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)
        return {**state, "feedback": result.content, "messages": [*state.get("messages", []), *prompt.messages, result]}

    def rewrite_query(self, state: State) -> State:
        prompt = REWRITE_QUERY_PROMPT.invoke(
            {
                "dialect": self.db.dialect,
                "input": state["question"],
                "query": state["query"],
                "execution": self.truncate_execuion(state["execution"]),
                "feedback": state["feedback"],
                "table_info": self.get_table_info(),
            }
        )
        result = self.invoke_prompt(prompt)
        rewritten_query = self.parse_query(result)
        return {
            **state,
            "query": rewritten_query or state["query"],
            "num_turns": state.get("num_turns", 0) + 1,
            "messages": [*prompt.messages, result],
        }

    def should_continue(self, state: State) -> Literal[END, "rewrite_query"]:
        if state["messages"] and isinstance(state["messages"][-1], BaseMessage):
            last_message = state["messages"][-1]
            if "THE QUERY IS CORRECT" in last_message.content:
                if "THE QUERY IS INCORRECT" in last_message.content:
                    correct_index = last_message.content.rfind("THE QUERY IS CORRECT")
                    incorrect_index = last_message.content.rfind("THE QUERY IS INCORRECT")
                    if correct_index > incorrect_index:
                        return END
                else:
                    return END
        if state.get("num_turns", 0) >= self.max_turns:
            return END
        return "rewrite_query"

    def graph(self) -> CompiledStateGraph[State]:
        builder = StateGraph(State)
        builder.add_node(self.write_query)
        builder.add_node(self.execute_query)
        builder.add_node(self.check_query)
        builder.add_node(self.rewrite_query)
        builder.add_edge(START, "write_query")
        builder.add_edge("write_query", "execute_query")
        builder.add_edge("execute_query", "check_query")
        builder.add_conditional_edges("check_query", self.should_continue)
        builder.add_edge("rewrite_query", "execute_query")
        return builder.compile()


def evaluate_query(query: str, ground_truth: str, database: str, raise_on_error: bool = True) -> float:
    try:
        database = os.path.abspath(database)
        if not os.path.exists(database):
            raise FileNotFoundError(f"Database file {database} does not exist.")
        exec_score = eval_exec_match(
            db=database,
            p_str=query,
            g_str=ground_truth,
            plug_value=False,
            keep_distinct=False,
            progress_bar_for_each_datapoint=False,
        )
        return 1.0 if exec_score == 1 else 0.0
    except Exception as e:
        if raise_on_error:
            raise
        print(f"Error evaluating query: {e}")
        return 0.0


def get_query_from_qa(qa_path: str) -> tuple:
    """Extract query info from qa.json, return (task_id, question, ground_truth, db_id)"""
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    info = data[0] if isinstance(data, list) else data
    info = info.get("info", info)
    task_id = data[0].get("task_id", "task0000") if isinstance(data, list) else data.get("task_id", "task0000")
    question = None
    for key in ["question", "Question", "task", "Task", "query", "Query"]:
        if key in info and info[key]:
            question = info[key]
            break
    ground_truth = info.get("query", info.get("ground_truth", ""))
    db_id = info.get("db_id", "")
    return task_id, question, ground_truth, db_id


if __name__ == "__main__":
    is_training = False
    end_point = os.getenv("OPENAI_BASE_URL")
    model_name = os.getenv("OPENAI_MODEL")
    max_turns = 3
    table_info_truncate = 2048
    execution_truncate = 2048
    val_temperature = None

    input_path = os.path.join(args.workdir, "qa.json")
    task_id, question, ground_truth, db_id = get_query_from_qa(input_path)

    if not question:
        print("No question found in qa.json")
        sys.exit(1)

    llm = LLM(resource_type="llm", endpoint=end_point, model=model_name, sampling_parameters={"temperature": 0.1})

    # Data directory is under script directory, not workdir
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Find database file
    original_db_path = None
    if db_id:
        # Try data/database/{db_id}/{db_id}.sqlite first
        for sub in ["database", "test_database", ""]:
            candidate = (
                os.path.join(script_dir, "data", sub, db_id, db_id + ".sqlite")
                if sub
                else os.path.join(script_dir, "data", db_id, db_id + ".sqlite")
            )
            if os.path.exists(candidate):
                original_db_path = candidate
                break
    if not original_db_path:
        print(f"Database not found for db_id={db_id}")
        sys.exit(1)

    # Schema file in same directory as database
    schema_path = os.path.join(os.path.dirname(original_db_path), "schema.sql")
    schema = "No schema available."
    if os.path.exists(schema_path):
        with open(schema_path, "r") as f:
            schema = f.read()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)

            agent = SQLAgent(
                "sqlite:///" + db_path,
                max_turns=max_turns,
                table_info_truncate=table_info_truncate,
                execution_truncate=execution_truncate,
                debug=False,
                db_schema=schema,
                endpoint=llm.endpoint,
                verl_replacement=(
                    {"model": llm.model, **llm.sampling_parameters}
                    if is_training
                    else {
                        "model": llm.model,
                        "temperature": (
                            val_temperature
                            if val_temperature is not None
                            else llm.sampling_parameters.get("temperature", 0.0)
                        ),
                    }
                ),
            ).graph()

            result = agent.invoke({"question": question})

        print(f"Question: {question}")
        print(f"Generated Query: {result['query']}")
        print(f"Ground Truth: {ground_truth}")

        # Evaluate with fresh db copy
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, os.path.basename(original_db_path))
            shutil.copyfile(original_db_path, db_path)
            reward = evaluate_query(result["query"], ground_truth, db_path, raise_on_error=False)

        _this_corr = int(reward)

        # Save result to output.json
        output_path = os.path.join(args.workdir, "output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "success" if _this_corr else "fail",
                    "task_id": task_id,
                    "task": question,
                    "answer_pred": result["query"],
                    "answer_gold": ground_truth,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(f"Reward: {reward}")
        print(
            "RESULT: " + json.dumps({"task_id": task_id, "_this_corr": _this_corr}, ensure_ascii=True),
            flush=True,
        )

    except Exception as e:
        print(f"Error during agent invocation: {e}")
        # Save error result
        output_path = os.path.join(args.workdir, "output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "fail",
                    "task_id": task_id,
                    "task": question,
                    "answer_pred": "error",
                    "answer_gold": ground_truth,
                    "error": str(e),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("RESULT: " + json.dumps({"task_id": task_id, "_this_corr": 0}, ensure_ascii=True), flush=True)
