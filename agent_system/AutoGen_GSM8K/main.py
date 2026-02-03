# main.py
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
    default="./workdir-dev/gsm8k-0000_20251103231550",
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
    project_name="math-agents",
    batch=True,
    auto_instrument=True,
)
# =====================================

import json


import re
from dataclasses import dataclass
from typing import Dict, List
from autogen_core.models import ModelFamily
from autogen_core import (
    DefaultTopicId,
    MessageContext,
    RoutedAgent,
    SingleThreadedAgentRuntime,
    TypeSubscription,
    default_subscription,
    message_handler,
)
from autogen_core.models import (
    AssistantMessage,
    ChatCompletionClient,
    LLMMessage,
    SystemMessage,
    UserMessage,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from datasets import load_dataset

from prompt import (
    MATHSOLVERA_SYSTEM,
    MATHSOLVERB_SYSTEM,
    MATHSOLVERC_SYSTEM,
    MATHSOLVERD_SYSTEM,
)

VIS = True


@dataclass
class Question:
    content: str


@dataclass
class Answer:
    content: str


@dataclass
class SolverRequest:
    content: str
    question: str


@dataclass
class IntermediateSolverResponse:
    content: str
    question: str
    answer: str
    round: int


@dataclass
class FinalSolverResponse:
    answer: str


@default_subscription
class MathSolver(RoutedAgent):
    def __init__(
        self,
        model_client: ChatCompletionClient,
        topic_type: str,
        num_neighbors: int,
        max_round: int,
        system_messages: List[SystemMessage],
    ) -> None:
        super().__init__("A debator.")
        self._topic_type = topic_type
        self._model_client = model_client
        self._num_neighbors = num_neighbors
        self._history: List[LLMMessage] = []
        self._buffer: Dict[int, List[IntermediateSolverResponse]] = {}
        self._system_messages = system_messages
        self._round = 0
        self._max_round = max_round

    @message_handler
    async def handle_request(self, message: SolverRequest, ctx: MessageContext) -> None:
        # Add the question to the memory.
        self._history.append(UserMessage(content=message.content, source="user"))
        # Make an inference using the model.
        model_result = await self._model_client.create(
            self._system_messages + self._history,
            extra_create_args=(
                {
                    "extra_body": (
                        {"reasoning_effort": "medium"}
                        if "gpt" in os.getenv("OPENAI_MODEL").lower()
                        else {"thinking": {"type": "disabled"}}
                    )
                }
            ),
        )
        assert isinstance(model_result.content, str)
        # Add the response to the memory.
        self._history.append(AssistantMessage(content=model_result.content, source=self.metadata["type"]))
        if VIS:
            print(f"{'-'*80}\nSolver {self.id} round {self._round}:\n{model_result.content}")
        # Extract the answer from the response.
        match = re.search(r"\{\{(\-?\d+(\.\d+)?)\}\}", model_result.content)
        # if match is None:
        #     raise ValueError("The model response does not contain the answer.")
        # answer = match.group(1)
        if match is None:
            # If primary format not found, try fallback patterns
            print(f"Warning: Solver {self.id} couldn't extract answer from: {model_result.content[:100]}...")
            # Try alternative formats like "answer is 42" or Chinese "答案是 42"

            fallback_match = re.search(
                r"(?:答案是|answer is|答案为|the answer is)\s*(\-?\d+(?:\.\d+)?)", model_result.content, re.IGNORECASE
            )
            if fallback_match:
                answer = fallback_match.group(1)
            else:
                # If still not found, use default value
                answer = "Unable to parse and obtain the answer"
        else:
            answer = match.group(1)
        # Increment the counter.
        self._round += 1
        if self._round == self._max_round:
            # If the counter reaches the maximum round, publishes a final response.
            await self.publish_message(FinalSolverResponse(answer=answer), topic_id=DefaultTopicId())
        else:
            # Publish intermediate response to the topic associated with this solver.
            await self.publish_message(
                IntermediateSolverResponse(
                    content=model_result.content,
                    question=message.question,
                    answer=answer,
                    round=self._round,
                ),
                topic_id=DefaultTopicId(type=self._topic_type),
            )

    @message_handler
    async def handle_response(self, message: IntermediateSolverResponse, ctx: MessageContext) -> None:
        # Add neighbor's response to the buffer.
        self._buffer.setdefault(message.round, []).append(message)
        # Check if all neighbors have responded.
        if len(self._buffer[message.round]) == self._num_neighbors:
            if VIS:
                print(
                    f"{'-'*80}\nSolver {self.id} round {message.round}:\nReceived all responses from {self._num_neighbors} neighbors."
                )
            # Prepare the prompt for the next question.
            prompt = "These are the solutions to the problem from other agents:\n"
            for resp in self._buffer[message.round]:
                prompt += f"One agent solution: {resp.content}\n"
            prompt += (
                "Using the solutions from other agents as additional information, "
                "can you provide your answer to the math problem? "
                "The original math problem is {question}. "
                "Your final answer should be a single numerical number, "
                "in the form of {{answer}}, at the end of your response."
            ).format(question=message.question)
            # Send the question to the agent itself to solve.
            await self.send_message(SolverRequest(content=prompt, question=message.question), self.id)
            # Clear the buffer.
            self._buffer.pop(message.round)


@default_subscription
class MathAggregator(RoutedAgent):
    def __init__(self, num_solvers: int, result_future: asyncio.Future) -> None:
        super().__init__("Math Aggregator")
        self._num_solvers = num_solvers
        self._buffer: List[FinalSolverResponse] = []
        self._result_future = result_future

    @message_handler
    async def handle_question(self, message: Question, ctx: MessageContext) -> None:
        if VIS:
            print(f"{'-'*80}\nAggregator {self.id} received question:\n{message.content}")
        prompt = (
            "Can you solve the following math problem?\n{question}\n"
            "Explain your reasoning. Your final answer should be a single numerical number, "
            "in the form of {{answer}}, at the end of your response."
        ).format(question=message.content)
        if VIS:
            print(f"{'-'*80}\nAggregator {self.id} publishes initial solver request.")
        await self.publish_message(SolverRequest(content=prompt, question=message.content), topic_id=DefaultTopicId())

    @message_handler
    async def handle_final_solver_response(self, message: FinalSolverResponse, ctx: MessageContext) -> None:
        self._buffer.append(message)
        if len(self._buffer) == self._num_solvers:
            if VIS:
                print(f"{'-'*80}\nAggregator {self.id} received all final answers from {self._num_solvers} solvers.")
            # Find the majority answer.
            answers = [resp.answer for resp in self._buffer]
            majority_answer = max(set(answers), key=answers.count)

            if not self._result_future.done():
                self._result_future.set_result(majority_answer)

            # Publish the aggregated response.
            await self.publish_message(Answer(content=majority_answer), topic_id=DefaultTopicId())
            # Clear the responses.
            self._buffer.clear()
            if VIS:
                print(f"{'-'*80}\nAggregator {self.id} publishes final answer:\n{majority_answer}")


"""
The solver agents will be connected in a sparse manner as illustrated in the figure below:
A --- B
|     |
|     |
D --- C
"""


async def main(question):
    runtime = SingleThreadedAgentRuntime()
    result_future = asyncio.Future()

    model_client = OpenAIChatCompletionClient(
        model_info={
            "vision": False,
            "function_calling": False,
            "json_output": False,
            "family": ModelFamily.UNKNOWN,
            "structured_output": True,
        },
        model=model_name,
        api_key=api_key,
        base_url=base_url,
    )

    await MathSolver.register(
        runtime,
        "MathSolverA",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverA",
            system_messages=[SystemMessage(content=MATHSOLVERA_SYSTEM)],
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathSolver.register(
        runtime,
        "MathSolverB",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverB",
            system_messages=[SystemMessage(content=MATHSOLVERB_SYSTEM)],
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathSolver.register(
        runtime,
        "MathSolverC",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverC",
            system_messages=[SystemMessage(content=MATHSOLVERC_SYSTEM)],
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathSolver.register(
        runtime,
        "MathSolverD",
        lambda: MathSolver(
            model_client=model_client,
            topic_type="MathSolverD",
            system_messages=[SystemMessage(content=MATHSOLVERD_SYSTEM)],
            num_neighbors=2,
            max_round=3,
        ),
    )
    await MathAggregator.register(
        runtime, "MathAggregator", lambda: MathAggregator(num_solvers=4, result_future=result_future)
    )

    # Subscriptions for topic published to by MathSolverA.
    await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverD"))
    await runtime.add_subscription(TypeSubscription("MathSolverA", "MathSolverB"))

    # Subscriptions for topic published to by MathSolverB.
    await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverA"))
    await runtime.add_subscription(TypeSubscription("MathSolverB", "MathSolverC"))

    # Subscriptions for topic published to by MathSolverC.
    await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverB"))
    await runtime.add_subscription(TypeSubscription("MathSolverC", "MathSolverD"))

    # Subscriptions for topic published to by MathSolverD.
    await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverC"))
    await runtime.add_subscription(TypeSubscription("MathSolverD", "MathSolverA"))

    # All solvers and the aggregator subscribe to the default topic.

    runtime.start()
    await runtime.publish_message(Question(content=question), DefaultTopicId())

    majority_answer = await result_future
    if VIS:
        print(f"{'='*80}\nFinal aggregated answer: {majority_answer}")

    # Wait for the runtime to stop when idle.
    await runtime.stop_when_idle()
    # Close the connection to the model client.
    await model_client.close()

    return majority_answer


def extract_ground_truth(text):
    return text.split("####")[-1].strip()


def get_query_from_qa(qa_path: str) -> tuple:
    """Extract query info from qa.json, return (task_id, question, answer)"""
    with open(qa_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    info = data[0] if isinstance(data, list) else data
    task_id = info.get("task_id", "task0000")
    question = info.get("question", "")
    answer = info.get("answer", "")
    return task_id, question, answer


if __name__ == "__main__":
    base_url = os.getenv("OPENAI_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3/")
    model_name = os.getenv("OPENAI_MODEL", "doubao-seed-1.6-250615")
    api_key = os.getenv("OPENAI_API_KEY", "")

    input_path = os.path.join(args.workdir, "qa.json")
    task_id, question, answer_raw = get_query_from_qa(input_path)

    if not question:
        print("No question found in qa.json")
        sys.exit(1)

    ground_truth_answer = extract_ground_truth(answer_raw)

    try:
        majority_answer = asyncio.run(main(question))
        correct = (majority_answer == ground_truth_answer) if majority_answer is not None else False
        _this_corr = 1 if correct else 0

        print(f"Question: {question}")
        print(f"Ground Truth: {ground_truth_answer}, Predicted: {majority_answer}, Correct: {correct}")

        # Save result to output.json
        output_path = os.path.join(args.workdir, "output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "success" if _this_corr else "fail",
                    "task_id": task_id,
                    "task": question,
                    "answer_pred": majority_answer,
                    "answer_gold": ground_truth_answer,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

        print(
            "RESULT: " + json.dumps({"task_id": task_id, "_this_corr": _this_corr}, ensure_ascii=True),
            flush=True,
        )
    except Exception as e:
        print(f"Error during agent invocation: {e}")
        output_path = os.path.join(args.workdir, "output.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "status": "fail",
                    "task_id": task_id,
                    "task": question,
                    "answer_pred": "error",
                    "answer_gold": ground_truth_answer,
                    "error": str(e),
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print("RESULT: " + json.dumps({"task_id": task_id, "_this_corr": 0}, ensure_ascii=True), flush=True)
